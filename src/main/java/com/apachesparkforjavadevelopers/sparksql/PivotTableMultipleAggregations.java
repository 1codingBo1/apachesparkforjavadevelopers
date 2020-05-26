package com.apachesparkforjavadevelopers.sparksql;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;

public class PivotTableMultipleAggregations {
    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("PivotTable").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///C:/Users/mheinecke/spark_tmp")
                .getOrCreate();

        Dataset<Row> dataset = spark.read().option("header", true).csv("src/main/resources/exams/students.csv");

        dataset.select("subject", "year", "score")
                .groupBy("subject")
                .pivot("year")
                .agg(
                        round(mean("score"), 2).alias("mean_score"),
                        round(stddev("score"), 2).alias("stddev_score")
                ).show();

        spark.stop();
    }
}
