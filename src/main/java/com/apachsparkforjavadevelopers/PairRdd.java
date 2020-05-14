package com.apachsparkforjavadevelopers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

public class PairRdd {

    public static void main(String[] args) {
        List<String> inputData = Arrays.asList(
                "WARN: Tuesday 4 September 0405",
                "ERROR: Tuesday 4 September 0408",
                "FATAL: Wednesday 5 September 1632",
                "ERROR: Friday 7 September 1854",
                "WARN: Saturday 8 September 1942"
        );

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf conf = new SparkConf().setAppName("startingSpark").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> originalLogMessages = sc.parallelize(inputData);

        JavaPairRDD<String, Long> pairRdd = originalLogMessages.mapToPair(rawString -> {

            String[] columns = rawString.split(":");
            String level = columns[0];

            return new Tuple2<>(level, 1L);
        });

        JavaPairRDD<String, Long> sumsRdd = pairRdd.reduceByKey(Long::sum);

        sumsRdd.foreach(tuple2 -> System.out.println(tuple2._1 + " has " + tuple2._2 + " instances"));

        sc.close();
    }
}
