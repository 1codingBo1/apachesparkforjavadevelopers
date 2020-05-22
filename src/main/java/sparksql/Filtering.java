package sparksql;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Filtering {
    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("datasetBasics").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///C:/Users/mheinecke/spark_tmp")
                .getOrCreate();

        Dataset<Row> dataset = spark.read().option("header", true).csv("src/main/resources/exams/students.csv");

        // filtering with expressions
        Dataset<Row> filterExpression = dataset.filter("subject = 'Modern Art' AND year >= 2007");
        filterExpression.show();

        // filtering with lambdas
        Dataset<Row> filterLambda = dataset.filter((FilterFunction<Row>) row -> row.getAs("subject").equals("Modern Art")
                && Integer.parseInt(row.getAs("year")) >= 2007);
        filterLambda.show();

        spark.close();
    }
}
