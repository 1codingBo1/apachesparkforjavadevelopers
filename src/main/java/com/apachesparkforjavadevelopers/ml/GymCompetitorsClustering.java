package com.apachesparkforjavadevelopers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GymCompetitorsClustering {

    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("GymCompetitorsClustering").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///C:/Users/mheinecke/spark_tmp")
                .getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/gymcompetition/GymCompetition.csv");

        StringIndexer genderIndexer = new StringIndexer()
                .setInputCol("Gender")
                .setOutputCol("GenderIndex");
        csvData = genderIndexer.fit(csvData).transform(csvData);

        new IndexToString()
                .setInputCol("GenderIndex")
                .setOutputCol("Value")
                .transform(csvData.select("GenderIndex").dropDuplicates())
                .show();

        OneHotEncoderEstimator genderEncoder = new OneHotEncoderEstimator()
                .setInputCols(new String[]{"GenderIndex"})
                .setOutputCols(new String[]{"GenderVector"});
        csvData = genderEncoder.fit(csvData).transform(csvData);

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"Age", "Height", "Weight", "GenderVector", "NoOfReps"})
                .setOutputCol("features");
        Dataset<Row> inputData = vectorAssembler.transform(csvData).select("features");

//        inputData.show();

        KMeans kMeans = new KMeans();

        for (int noOfClusters = 2; noOfClusters <= 8; noOfClusters++) {
            // model can only be improved by changing number of clusters
            // plot SEE and silhouette after looping over different cluster numbers to find elbow in graphs
            // which indicate optimal number of clusters
            // 5 clusters would be optimal in this example

            // set number of clusters
            kMeans.setK(noOfClusters);

            System.out.println("Number of clusters: " + noOfClusters);

            KMeansModel model = kMeans.fit(inputData);
            Dataset<Row> predictions = model.transform(inputData);
            predictions.show();

//            Vector[] clusterCenters = model.clusterCenters();
//            Arrays.stream(clusterCenters).forEach(System.out::println);

            // check number of records per cluster
            predictions.groupBy("prediction").count().show();

            // the closer to 0, the better
            System.out.println("The sum of squared errors is " + model.computeCost(inputData));

            // the closer to 1, the better
            ClusteringEvaluator evaluator = new ClusteringEvaluator();
            System.out.println("The silhouette with squared euclidean distance is " + evaluator.evaluate(predictions));
        }

    }

}
