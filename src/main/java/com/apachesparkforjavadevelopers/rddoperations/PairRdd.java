package com.apachesparkforjavadevelopers.rddoperations;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
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

        sc.parallelize(inputData)
                .mapToPair(rawString -> (new Tuple2<>(rawString.split(":")[0], 1L)))
                .reduceByKey(Long::sum)
                .foreach(tuple2 -> System.out.println(tuple2._1 + " has " + tuple2._2 + " instances"));

        sc.close();
    }
}
