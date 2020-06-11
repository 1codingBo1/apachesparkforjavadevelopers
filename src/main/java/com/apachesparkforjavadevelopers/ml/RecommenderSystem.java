package com.apachesparkforjavadevelopers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class RecommenderSystem {

    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("RecommenderSystem")
                .master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///C:/Users/mheinecke/spark_tmp")
                .getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/VPPcourseViews.csv");

//        csvData.groupBy("userId").pivot("courseId").sum("proportionWatched").show();

//        Dataset<Row>[] trainingAndHoldoutData = csvData.randomSplit(new double[]{0.9, 0.1});
//        Dataset<Row> trainingData = trainingAndHoldoutData[0];
//        Dataset<Row> holdOotData = trainingAndHoldoutData[1];

        ALS als = new ALS()
                .setMaxIter(10)
                .setRegParam(0.1)
                .setUserCol("userId")
                .setItemCol("courseId")
                .setRatingCol("proportionWatched");

        ALSModel model = als.fit(csvData);

        // drop records in output dataset for which no recommendation can be made due to lack of data
        // default is including those records with value NaN in prediction column
        model.setColdStartStrategy("drop");

//        Dataset<Row> userRecommendations = model.recommendForAllUsers(5);

//        List<Row> userRecommendationsList = userRecommendations.takeAsList(5);
//
//        for (Row r : userRecommendationsList) {
//			int userId = r.getAs(0);
//			String recs = r.getAs(1).toString();
//			System.out.println("User " + userId + " we might want to recommend " + recs);
//			System.out.println("This user has already watched: ");
//			csvData.filter("userId = " + userId).show();
//		}

        Dataset<Row> testData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/VPPcourseViewsTest.csv");

        model.transform(testData).show();
        model.recommendForUserSubset(testData, 5).show();
    }

}
