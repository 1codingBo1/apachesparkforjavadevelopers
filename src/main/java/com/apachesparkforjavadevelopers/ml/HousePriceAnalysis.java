package com.apachesparkforjavadevelopers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

public class HousePriceAnalysis {

    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("HousePriceAnalysis").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///C:/Users/mheinecke/spark_tmp")
                .getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/kc_house_data.csv")
                .drop("id", "date", "view", "yr_renovated", "lat", "long",
                        "sqft_lot", "sqft_lot15", "yr_built", "sqft_living15",
                        "sqft_basement")
                .withColumn("sqft_above_share", col("sqft_above").divide(col("sqft_living")));

        // create vectors for categorical columns condition, grade, and zipcode
        // index columns
        StringIndexer conditionIndexer = new StringIndexer()
                .setInputCol("condition")
                .setOutputCol("condition_index");
        csvData = conditionIndexer.fit(csvData).transform(csvData);

        StringIndexer gradeIndexer = new StringIndexer()
                .setInputCol("grade")
                .setOutputCol("grade_index");
        csvData = gradeIndexer.fit(csvData).transform(csvData);

        StringIndexer zipcodeIndexer = new StringIndexer()
                .setInputCol("zipcode")
                .setOutputCol("zipcode_index");
        csvData = zipcodeIndexer.fit(csvData).transform(csvData);

        // encode indexed columns
        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[]{"condition_index", "grade_index", "zipcode_index"})
                .setOutputCols(new String[]{"condition_vector", "grade_vector", "zipcode_vector"});
        csvData = encoder.fit(csvData).transform(csvData);

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"bedrooms", "bathrooms", "sqft_living", "sqft_above_share", "floors",
                        "condition_vector", "grade_vector", "zipcode_vector", "waterfront"})
                .setOutputCol("features");
        Dataset<Row> csvDataWithFeatures = vectorAssembler.transform(csvData);

        Dataset<Row> modelInputData = csvDataWithFeatures
                .select("price", "features")
                .withColumnRenamed("price", "label");

        Dataset<Row>[] dataSplits = modelInputData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingAndTestData = dataSplits[0];
        Dataset<Row> holdOutData = dataSplits[1];

        LinearRegression linearRegression = new LinearRegression();

        // helper object to build ParamMap object
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();

        // ParamMap is a grid of parameter combinations for Spark to test which one yields the best model
        ParamMap[] paramMap = paramGridBuilder
                .addGrid(linearRegression.regParam(), new double[]{0.01, 0.1, 0.5})
                .addGrid(linearRegression.elasticNetParam(), new double[]{0, 0.5, 1})
                .build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(linearRegression) // param is type of model builder
                .setEvaluator(new RegressionEvaluator().setMetricName("r2")) // set metric for model evaluation
                .setEstimatorParamMaps(paramMap)
                .setTrainRatio(0.8); // ratio for training and test data split

        TrainValidationSplitModel model = trainValidationSplit.fit(trainingAndTestData);
        LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();

        System.out.println("The training data r2 value is " + lrModel.summary().r2());
        System.out.println("The training data RMSE value is " + lrModel.summary().rootMeanSquaredError());

        System.out.println("The test data r2 value is " + lrModel.evaluate(holdOutData).r2());
        System.out.println("The test data RMSE value is " + lrModel.evaluate(holdOutData).rootMeanSquaredError());

        System.out.println("Intercept: " + lrModel.intercept() + "; coefficient: " + lrModel.coefficients());
        System.out.println("regPram: " + lrModel.getRegParam() + "; elastic net param: " + lrModel.getElasticNetParam());

    }

}
