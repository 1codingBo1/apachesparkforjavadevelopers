package com.apachesparkforjavadevelopers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;

public class FreeTrialDecisionTree {

    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("LogisticCustomerChurnModel")
                .master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///C:/Users/mheinecke/spark_tmp")
                .getOrCreate();

        spark.udf().register("countryGrouping", (String country) -> {
            List<String> topCountries = Arrays.asList("GB", "US", "IN", "UNKNOWN");
            List<String> europeanCountries = Arrays.asList("BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR",
                    "HR", "IT", "CY", "LV", "LT", "LU", "HU", "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI",
                    "SE", "CH", "IS", "NO", "LI", "EU");
            if (topCountries.contains(country)) return country;
            if (europeanCountries.contains(country)) return "EUROPE";
            else return "OTHER";
        }, DataTypes.StringType);

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/vppFreeTrials.csv")
                .withColumn("country", callUDF("countryGrouping", col("country")))
                // 1: paying customer, 0: non-paying customer
                .withColumn("label", when(col("payments_made").geq(1), 1).otherwise(0));

        // categorical data has to be numerical but not encoded for decision tree
        StringIndexer countryIndexer = new StringIndexer();
        csvData = countryIndexer
                .setInputCol("country")
                .setOutputCol("countryIndex")
                .fit(csvData)
                .transform(csvData);

        // create table with country indexes and country names to print to console
        // to help with later interpretation of output
        new IndexToString()
                .setInputCol("countryIndex")
                .setOutputCol("value")
                .transform(csvData.select("countryIndex").dropDuplicates())
                .show();

        Dataset<Row> inputData = new VectorAssembler()
                .setInputCols(new String[]{"countryIndex", "rebill_period", "chapter_access_count", "seconds_watched"})
                .setOutputCol("features")
                .transform(csvData)
                .select("label", "features");

//        inputData.show();

        Dataset<Row>[] trainingAndHoldoutData = inputData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingData = trainingAndHoldoutData[0];
        Dataset<Row> holdoutData = trainingAndHoldoutData[1];

        DecisionTreeClassifier dtClassifier = new DecisionTreeClassifier();
        dtClassifier.setMaxDepth(3);

        DecisionTreeClassificationModel model = dtClassifier.fit(trainingData);

        Dataset<Row> predictions = model.transform(holdoutData);
        predictions.show();

        System.out.println(model.toDebugString());

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy");
        System.out.printf("The model accuracy is %f.%n", evaluator.evaluate(predictions));

        // random forest
        RandomForestClassifier rfClassifier = new RandomForestClassifier()
                .setMaxDepth(3);
        RandomForestClassificationModel rfModel = rfClassifier.fit(trainingData);
        Dataset<Row> rfPredictions = rfModel.transform(holdoutData);
        rfPredictions.show();

        System.out.println(rfModel.toDebugString());
        System.out.printf("The forest model accuracy is %f.%n", evaluator.evaluate(rfPredictions));

    }

}
