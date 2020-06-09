package com.apachesparkforjavadevelopers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
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

public class CustomerChurnModel {

    /*
    Requirement
    Build model to predict how many videos a customer will watch next months

    Guidance
    1. Filter out records where customer has cancelled their subscription
    2. Replace nulls in columns firstSub, all_time_views, last_month_views, next_month_views
    3. Build model using all fields but observation_date
    3.1 Encode categorical data in columns payment_method_type, country, rebill_period_in_months
    3.2 Use 90/10 data split
     */

    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("CustomerChurnModel").master("local[*]")
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
                .withColumnRenamed("next_month_views", "label");

        csvData.printSchema();
        csvData.show();

        Dataset<Row>[] dataSplits = csvData.randomSplit(new double[]{0.9, 0.1});
        Dataset<Row> trainingAndTestData = dataSplits[0];
        Dataset<Row> holdOutData = dataSplits[1];

        // create vectors for categorical columns payment_method_type, country, rebill_period_in_months
        // index columns
        StringIndexer paymentMethodIndexer = new StringIndexer()
                .setInputCol("payment_method_type")
                .setOutputCol("payment_method_type_index");

        StringIndexer countryIndexer = new StringIndexer()
                .setInputCol("country")
                .setOutputCol("country_index");

        StringIndexer rebillPeriodIndexer = new StringIndexer()
                .setInputCol("rebill_period_in_months")
                .setOutputCol("rebill_period_in_months_index");

        // encode indexed columns
        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[]{"payment_method_type_index", "country_index", "rebill_period_in_months_index"})
                .setOutputCols(new String[]{"payment_method_type_vector", "country_vector", "rebill_period_in_months_vector"});

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"payment_method_type_vector", "country_vector", "rebill_period_in_months_vector",
                        "first_sub", "age", "all_time_views", "last_month_views"})
                .setOutputCol("features");

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
                .setTrainRatio(0.9); // ratio for training and test data split

        // create pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{paymentMethodIndexer, countryIndexer, rebillPeriodIndexer, encoder,
                        vectorAssembler, trainValidationSplit});
        PipelineModel pipelineModel = pipeline.fit(trainingAndTestData);
        TrainValidationSplitModel model = (TrainValidationSplitModel) pipelineModel.stages()[5];
        LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();

        Dataset<Row> holdOutResults = pipelineModel.transform(holdOutData).drop("prediction");

        System.out.println("The training data r2 value is " + lrModel.summary().r2());
        System.out.println("The training data RMSE value is " + lrModel.summary().rootMeanSquaredError());

        System.out.println("The holdout data r2 value is " + lrModel.evaluate(holdOutResults).r2());
        System.out.println("The holdout data RMSE value is " + lrModel.evaluate(holdOutResults).rootMeanSquaredError());

    }

}
