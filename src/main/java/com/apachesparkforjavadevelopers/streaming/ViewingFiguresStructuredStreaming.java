package com.apachesparkforjavadevelopers.streaming;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.OutputMode;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;

import static org.apache.spark.sql.functions.*;

public class ViewingFiguresStructuredStreaming {

    public static void main(String[] args) throws StreamingQueryException {

        // using winutils.exe of Hadoop 3.2.1 after upgrade
        // not on PATH
        System.setProperty("hadoop.home.dir", "C:\\Users\\mheinecke\\hadoop");

        Logger.getLogger("org.apache").setLevel(Level.WARN);
        Logger.getLogger("org.apache.spark.storage").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .master("local[*]")
                .appName("ViewingFiguresStructuredStreaming")
                // set to appropriate value for data
                // empty partitions are also processed (only overhead) which adds to processing time
                .config("spark.sql.shuffle.partitions", "2")
                .getOrCreate();

        Dataset<Row> df = spark.readStream()
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "viewrecords")
                .load();

        StreamingQuery query = df
                // explicit cast to string on underlying byte array needed
                .select(col("value").cast("string").alias("course"), col("timestamp"))
                // one event for every 5 seconds viewing time
                .withColumn("seconds_viewed", lit(5L))
                .groupBy(
                        window(col("timestamp"), "30 seconds"),
                        col("course")
                )
                .agg(sum("seconds_viewed").alias("seconds_viewed"))
                .writeStream()
                .format("console")
                .outputMode(OutputMode.Update())
                .option("truncate", false)
                .start();

        query.awaitTermination();

    }
}
