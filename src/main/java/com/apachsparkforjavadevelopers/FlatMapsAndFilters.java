package com.apachsparkforjavadevelopers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;
import java.util.List;

public class FlatMapsAndFilters {

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
                .flatMap(sentence -> Arrays.asList(sentence.split(" ")).iterator())
                .filter(word -> word.matches("\\D+")) // filtering out numerals
                .collect()
                .forEach(System.out::println);

        sc.close();
    }
}
