package com.apachesparkforjavadevelopers.streaming;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

public class LogStreamAnalysis {

    public static void main(String[] args) throws InterruptedException {

        Logger.getLogger("org.apache").setLevel(Level.WARN);
        Logger.getLogger("org.apache.spark.storage").setLevel(Level.ERROR);

        SparkConf conf = new SparkConf().setAppName("LogStreamAnalysis").setMaster("local[*]");
        JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(5));

        JavaReceiverInputDStream<String> inputData = jssc.socketTextStream("localhost", 8989);

        JavaPairDStream<String, Long> results = inputData
                .map(item -> item)
                .mapToPair(rawLogMessage -> new Tuple2<>(rawLogMessage.split(",")[0], 1L))
                .reduceByKeyAndWindow(Long::sum, Durations.seconds(20));

        results.print();

        jssc.start(); // start processing stream
        jssc.awaitTermination(); // keep application alive until JVM is terminated

    }

}
