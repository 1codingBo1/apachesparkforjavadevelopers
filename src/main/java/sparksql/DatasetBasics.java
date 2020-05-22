package sparksql;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class DatasetBasics {
    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("datasetBasics").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///C:/Users/mheinecke/spark_tmp")
                .getOrCreate();

        Dataset<Row> dataset = spark.read().option("header", true).csv("src/main/resources/exams/students.csv");
        dataset.show();

        long numRows = dataset.count();
        System.out.println("There are " + numRows + " rows in the dataset.");

        Row firstRow = dataset.first();

        String subject = firstRow.get(2).toString();
        int score = Integer.parseInt(firstRow.getAs("score"));
        String grade = firstRow.getAs("grade").toString();

        System.out.println("Subject: " + subject);
        System.out.println("Score: " + score);
        System.out.println("Grade: " + grade);

        spark.close();
    }
}
