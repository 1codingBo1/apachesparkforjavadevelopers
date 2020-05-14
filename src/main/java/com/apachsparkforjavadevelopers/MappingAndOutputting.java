package com.apachsparkforjavadevelopers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;
import java.util.List;


public class MappingAndOutputting {
    public static void main(String[] args) {
        List<Integer> inputData = Arrays.asList(2, 5, 7, 8);

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf conf = new SparkConf().setAppName("startingSpark").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<Integer> javaRdd = sc.parallelize(inputData);

        JavaRDD<Double> sqrtRdd = javaRdd.map(Math::sqrt); // lambda equivalent: v1 -> Math.sqrt(v1)

        sqrtRdd.collect().forEach(System.out::println);

        // counting elements in RDD with method; returns long
        long result1 = sqrtRdd.count();
        System.out.println(result1);

        // counting elements in RDD with map and reduce
        // long needs to be used in map if result exceeds int range
        long result2 = sqrtRdd.map(v1 -> 1L).reduce(Long::sum); // lambda equivalent: (v1, v2) -> v1 + v2
        System.out.println(result2);

        sc.close();
    }
}
