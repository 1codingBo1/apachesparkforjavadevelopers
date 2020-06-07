package com.apachesparkforjavadevelopers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceFields {
    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("HousePriceFields").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///C:/Users/mheinecke/spark_tmp")
                .getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/kc_house_data.csv")
                .drop("id", "date", "waterfront", "view", "condition",
                        "grade", "yr_renovated", "zipcode", "lat", "long",
                        "sqft_lot", "sqft_lot15", "yr_built", "sqft_living15");

        /*
        1. Eliminate dependent variables
        2. Check whether range of continuous variables is sufficiently wide
        3. Check for clear correlations of variables with outcome to be predicted
        4. Eliminate duplicate variables (representing the same)
         */

        // Step 2
//        csvData.describe().show(false);

        // Step 3
//        for (String col : csvData.columns()) {
//            System.out.printf("Correlation between price and %s: %f %n", col, csvData.stat().corr("price", col));
//        }

        // variables to use based on correlation: bedrooms, bathrooms, sqft_living, sqft_above, sqft_living15
        // sqft_living and sqft_living15 are dependent; choose sqft_living as correlation coefficient is higher

        // Step 4
        for (String col1 : csvData.columns()) {
            for (String col2 : csvData.columns()) {
                System.out.printf("Correlation between %s and %s: %f %n", col1, col2, csvData.stat().corr(col1, col2));
            }
        }

        // high correlation between sqft_living and sqft_above; drop sqft_above because of lower correlation with price
        // sqft_above and sqft_basement is sqft_living; instead of dropping sqft_above, calculate share of sqft_above
        // out of sqft_living as additional variable

    }
}
