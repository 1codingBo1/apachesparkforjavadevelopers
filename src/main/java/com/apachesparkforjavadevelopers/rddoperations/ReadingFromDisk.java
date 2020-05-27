package com.apachesparkforjavadevelopers.rddoperations;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;

public class ReadingFromDisk {

    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf conf = new SparkConf().setAppName("startingSpark").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> fromDiskRdd = sc.textFile("src/main/resources/subtitles/input.txt");

        fromDiskRdd.flatMap(line -> Arrays.asList(line.split(" ")).iterator())
                .collect()
                .forEach(System.out::println);

        sc.close();
    }
}
