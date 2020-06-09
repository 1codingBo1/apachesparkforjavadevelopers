package com.apachesparkforjavadevelopers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

public class LogisticCustomerChurnModel {
    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("LogisticCustomerChurnModel").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///C:/Users/mheinecke/spark_tmp")
                .getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/vppChapterViews/*.csv")
                .filter(col("is_cancelled").equalTo("false"))
                .selectExpr(
                        "payment_method_type",
                        "country",
                        "rebill_period_in_months",
                        "cast(firstSub as INT) as first_sub",
                        "age",
                        "all_time_views",
                        "last_month_views",
                        "next_month_views"
                )
                .na().fill(0, new String[]{"first_sub", "all_time_views", "last_month_views", "next_month_views"})
                // 1: customer will watch no videos, 0: customer will watch videos
                .withColumn("next_month_views", when(col("next_month_views").$greater(0), 0).otherwise(1))
                .withColumnRenamed("next_month_views", "label");

        Dataset<Row>[] dataSplits = csvData.randomSplit(new double[]{0.9, 0.1});
        Dataset<Row> trainingAndTestData = dataSplits[0];
        Dataset<Row> holdOutData = dataSplits[1];

        // indexers for categorical columns payment_method_type, country, rebill_period_in_months
        StringIndexer paymentMethodIndexer = new StringIndexer()
                .setInputCol("payment_method_type")
                .setOutputCol("payment_method_type_index");

        StringIndexer countryIndexer = new StringIndexer()
                .setInputCol("country")
                .setOutputCol("country_index");

        StringIndexer rebillPeriodIndexer = new StringIndexer()
                .setInputCol("rebill_period_in_months")
                .setOutputCol("rebill_period_in_months_index");

        // encoder for indexed columns
        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[]{"payment_method_type_index", "country_index", "rebill_period_in_months_index"})
                .setOutputCols(new String[]{"payment_method_type_vector", "country_vector", "rebill_period_in_months_vector"});

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"payment_method_type_vector", "country_vector", "rebill_period_in_months_vector",
                        "first_sub", "age", "all_time_views", "last_month_views"})
                .setOutputCol("features");

        LogisticRegression logisticRegression = new LogisticRegression();

        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        ParamMap[] paramMap = paramGridBuilder
                .addGrid(logisticRegression.regParam(), new double[]{0.01, 0.1, 0.5})
                .addGrid(logisticRegression.elasticNetParam(), new double[]{0, 0.5, 1})
                .build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(logisticRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setTrainRatio(0.9);

        // create pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{paymentMethodIndexer, countryIndexer, rebillPeriodIndexer, encoder,
                        vectorAssembler, trainValidationSplit});
        PipelineModel pipelineModel = pipeline.fit(trainingAndTestData);
        TrainValidationSplitModel model = (TrainValidationSplitModel) pipelineModel.stages()[5];
        LogisticRegressionModel lrModel = (LogisticRegressionModel) model.bestModel();

        System.out.println("The training data accuracy value is " + lrModel.summary().accuracy());

        Dataset<Row> holdOutResults = pipelineModel.transform(holdOutData).select("label", "features");

        LogisticRegressionSummary summary = lrModel.evaluate(holdOutResults);

        double truePositives = summary.truePositiveRateByLabel()[1];
        double falsePositives = summary.falsePositiveRateByLabel()[0];

        System.out.println("The holdout data accuracy value is " + summary.accuracy());
        System.out.printf("For the houldout data, the likelihood of a positive being correct is %f.%n",
                truePositives / (truePositives + falsePositives));

        lrModel.transform(holdOutResults).groupBy("label", "prediction").count().show();

    }

}
