package com.apachsparkforjavadevelopers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;
import java.util.List;


public class ReduceOnRDD {
    public static void main(String[] args) {
        List<Double> inputData = Arrays.asList(2.523, 1.234, 5.123, 0.58);

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf conf = new SparkConf().setAppName("startingSpark").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<Double> javaRdd = sc.parallelize(inputData);

        Double result = javaRdd.reduce((v1, v2) -> v1 + v2);

        System.out.println(result);

        sc.close();
    }
}
