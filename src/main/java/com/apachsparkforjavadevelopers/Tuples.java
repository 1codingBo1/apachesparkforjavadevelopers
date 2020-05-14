package com.apachsparkforjavadevelopers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;


public class Tuples {
    public static void main(String[] args) {
        List<Integer> inputData = Arrays.asList(2, 5, 7, 8);

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf conf = new SparkConf().setAppName("startingSpark").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // tuple
        Tuple2<Integer, Double> myValue = new Tuple2<>(9, 3.0);
        System.out.println(myValue);

        JavaRDD<Integer> originalIntegers = sc.parallelize(inputData);
        // tuple RDD
        JavaRDD<Tuple2<Integer, Double>> sqrtRdd = originalIntegers.map(v -> new Tuple2<>(v, Math.sqrt(v)));
        sqrtRdd.collect().forEach(System.out::println);

        sc.close();
    }
}
