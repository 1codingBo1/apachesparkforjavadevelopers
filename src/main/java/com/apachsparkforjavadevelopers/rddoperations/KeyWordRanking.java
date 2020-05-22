package com.apachsparkforjavadevelopers.rddoperations;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.Arrays;

public class KeyWordRanking {

    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf conf = new SparkConf().setAppName("KeyWordRankingExercise").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

//        String idPattern = "^\\D+$"; // e.g. 25
//        String timestampPattern = "\\D{2}:\\D{2}:\\D{2},\\D{3}";
//        String timeFramePattern = "^" + timestampPattern + "\\s{1}-->\\s{1}" + timestampPattern + "$"; // e.g. 00:10:43,289 --> 00:10:47,777
//        String idAndTimeFramePattern = idPattern + "|" + timeFramePattern;

//        JavaPairRDD<String, Long> wordCountRdd = sc.textFile("src/main/resources/subtitles/input.txt")
//                .map(String::trim)
//                .filter(line -> line.matches(idAndTimeFramePattern))
//                .flatMap(line -> Arrays.asList(line.split(" ")).iterator())
//                .map(word -> word.replaceAll("\\W", "")) // strip words of non-letters
//                .map(String::toLowerCase)
//                .filter(Util::isNotBoring)
//                .mapToPair(word -> new Tuple2<>(word, 1L))
//                .reduceByKey(Long::sum);

        JavaPairRDD<String, Long> wordCountRdd = sc.textFile("src/main/resources/subtitles/input.txt")
                .map(line -> line.replaceAll("[^a-zA-Z\\s]", "").toLowerCase()) // strip words of non-letters
                .flatMap(line -> Arrays.asList(line.split(" ")).iterator())
                .filter(word -> word.trim().length() > 0)
                .filter(Util::isNotBoring)
                .mapToPair(word -> new Tuple2<>(word, 1L))
                .reduceByKey(Long::sum);

        System.out.println(wordCountRdd.getNumPartitions());

        wordCountRdd.mapToPair(Tuple2::swap) // swap tuple elements to be able to use sortByKey on the count
                .sortByKey(false)
                .take(10)
                .forEach(System.out::println);
    }

}
