# ==============================================================
# NAME OF EXPERIMENT : Stock Price Prediction Using PySpark
# STUDENT NAME       : Pratik Kumar Yadav
# ROLL NO.           : 43
# SECTION            : EM301
# SUBJECT            : Cluster Computing (INT315)
#
# AIM :
# To predict the closing price of a stock using real historical
# stock market data and trading volume with the help of
# Linear Regression in PySpark.
#
# SOFTWARE REQUIRED :
# 1. Python
# 2. PySpark
# 3. Pandas
# 4. Matplotlib
# 5. Command Prompt (CMD)
#
# DATASET USED :
# Real stock dataset (AAPL.csv from Kaggle), renamed as stock_data.csv
#
# ALGORITHM / STEPS :
# 1. Import required libraries.
# 2. Create Spark Session.
# 3. Load stock market CSV dataset into Spark DataFrame.
# 4. Select useful columns required for prediction.
# 5. Convert Date column into proper date format.
# 6. Check and handle missing values.
# 7. Sort the dataset by Date.
# 8. Generate lag-based features and moving average feature.
# 9. Drop null rows created due to lag features.
# 10. Assemble all input features into a feature vector.
# 11. Split data into training set and testing set.
# 12. Train Linear Regression model.
# 13. Predict stock closing prices.
# 14. Evaluate model using RMSE and R2 score.
# 15. Create three attractive visualizations:
#     (a) Actual price trend
#     (b) Predicted price trend
#     (c) Actual vs Predicted overlay
# 16. Save graphs and stop Spark session.
# ==============================================================


# --------------------------------------------------------------
# STEP 1 : Import Required Libraries
# --------------------------------------------------------------
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, lag, to_date
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------
# STEP 2 : Create Spark Session
# --------------------------------------------------------------
spark = SparkSession.builder \
    .appName("StockPricePredictionRealData") \
    .getOrCreate()

print("===================================================")
print("         STOCK PRICE PREDICTION USING PYSPARK      ")
print("===================================================")
print("Spark Session created successfully.")


# --------------------------------------------------------------
# STEP 3 : Load Real Stock Dataset
# --------------------------------------------------------------
# The CSV file stock_data.csv must be present in the same folder
# as this Python script.
df = spark.read.csv("stock_data.csv", header=True, inferSchema=True)

print("\nDataset loaded successfully.")


# --------------------------------------------------------------
# STEP 4 : Display Original Schema
# --------------------------------------------------------------
print("\nOriginal Schema of Dataset:")
df.printSchema()


# --------------------------------------------------------------
# STEP 5 : Select Required Columns
# --------------------------------------------------------------
# We keep only those columns which are needed for prediction.
required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
df = df.select(*required_cols)


# --------------------------------------------------------------
# STEP 6 : Convert Date Column into Proper Date Format
# --------------------------------------------------------------
df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))


# --------------------------------------------------------------
# STEP 7 : Remove Invalid Date Rows
# --------------------------------------------------------------
df = df.dropna(subset=["Date"])

print("\nSchema After Selecting Required Columns:")
df.printSchema()

print("\nFirst 5 Rows of Dataset:")
df.show(5)


# --------------------------------------------------------------
# STEP 8 : Check Missing Values
# --------------------------------------------------------------
print("\nMissing Values Before Handling:")
for column in df.columns:
    missing_count = df.filter(col(column).isNull()).count()
    print(f"{column}: {missing_count}")


# --------------------------------------------------------------
# STEP 9 : Handle Missing Values
# --------------------------------------------------------------
# Missing numeric values are filled using mean of each column.
numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

for column_name in numeric_columns:
    mean_value = df.select(avg(col(column_name))).collect()[0][0]
    if mean_value is not None:
        df = df.fillna({column_name: mean_value})

print("\nMissing Values After Handling:")
for column in df.columns:
    missing_count = df.filter(col(column).isNull()).count()
    print(f"{column}: {missing_count}")


# --------------------------------------------------------------
# STEP 10 : Sort Dataset by Date
# --------------------------------------------------------------
df = df.orderBy("Date")


# --------------------------------------------------------------
# STEP 11 : Feature Engineering
# --------------------------------------------------------------
# Lag features are created to use previous day information.
# Moving average feature is created to capture short-term trend.
window_spec = Window.orderBy("Date")

# Previous day closing price
df = df.withColumn("Close_lag_1", lag("Close", 1).over(window_spec))

# Two days previous closing price
df = df.withColumn("Close_lag_2", lag("Close", 2).over(window_spec))

# Previous day trading volume
df = df.withColumn("Volume_lag_1", lag("Volume", 1).over(window_spec))

# 3-day moving average of closing price
moving_avg_window = Window.orderBy("Date").rowsBetween(-2, 0)
df = df.withColumn("Close_MA_3", avg("Close").over(moving_avg_window))

print("\nDataset After Feature Engineering:")
df.show(10)


# --------------------------------------------------------------
# STEP 12 : Drop Null Rows Generated by Lag Features
# --------------------------------------------------------------
df = df.dropna()

print("\nDataset After Dropping Null Rows:")
df.show(10)


# --------------------------------------------------------------
# STEP 13 : Prepare Feature Vector
# --------------------------------------------------------------
# These columns are used as input features for model training.
feature_columns = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Close_lag_1",
    "Close_lag_2",
    "Volume_lag_1",
    "Close_MA_3"
]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

final_df = assembler.transform(df).select(
    "Date",
    "features",
    col("Close").alias("label")
)

print("\nFinal Dataset Used for Machine Learning:")
final_df.show(5, truncate=False)


# --------------------------------------------------------------
# STEP 14 : Split Dataset into Training and Testing Data
# --------------------------------------------------------------
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

print(f"\nNumber of Training Rows : {train_data.count()}")
print(f"Number of Testing Rows  : {test_data.count()}")


# --------------------------------------------------------------
# STEP 15 : Build Linear Regression Model
# --------------------------------------------------------------
lr = LinearRegression(featuresCol="features", labelCol="label")
model = lr.fit(train_data)

print("\nLinear Regression model trained successfully.")


# --------------------------------------------------------------
# STEP 16 : Predict Stock Closing Price
# --------------------------------------------------------------
predictions = model.transform(test_data).orderBy("Date")

print("\nPredicted vs Actual Closing Prices:")
predictions.select("Date", "label", "prediction").show(20, truncate=False)


# --------------------------------------------------------------
# STEP 17 : Evaluate Model Performance
# --------------------------------------------------------------
# RMSE = Root Mean Squared Error
# R2   = Coefficient of Determination
rmse_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

r2_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)

rmse = rmse_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)

print("\n===================================================")
print("                 MODEL EVALUATION                  ")
print("===================================================")
print(f"RMSE     : {rmse}")
print(f"R2 Score : {r2}")


# --------------------------------------------------------------
# STEP 18 : Attractive Visualization (3 Graphs)
# --------------------------------------------------------------
# Convert prediction result into Pandas DataFrame for plotting.
plot_df = predictions.select("Date", "label", "prediction").orderBy("Date").toPandas()
plot_df["Date"] = pd.to_datetime(plot_df["Date"])

# ==============================
# GRAPH 1 : ACTUAL STOCK PRICE
# ==============================
plt.figure(figsize=(14, 6))

plt.plot(
    plot_df["Date"],
    plot_df["label"],
    color="blue",
    marker="o",
    linewidth=2.5,
    markersize=5
)

plt.title("Actual Stock Closing Price Trend", fontsize=18, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Actual Closing Price", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("actual_stock_price.png", dpi=300)
print("\nActual stock price graph saved as actual_stock_price.png")


# ==============================
# GRAPH 2 : PREDICTED STOCK PRICE
# ==============================
plt.figure(figsize=(14, 6))

plt.plot(
    plot_df["Date"],
    plot_df["prediction"],
    color="red",
    marker="D",
    linestyle="--",
    linewidth=2.5,
    markersize=5
)

plt.title("Predicted Stock Closing Price Trend", fontsize=18, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Predicted Closing Price", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("predicted_stock_price.png", dpi=300)
print("Predicted stock price graph saved as predicted_stock_price.png")


# =========================================
# GRAPH 3 : ACTUAL VS PREDICTED (OVERLAY)
# =========================================
plt.figure(figsize=(15, 7))

plt.plot(
    plot_df["Date"],
    plot_df["label"],
    color="blue",
    marker="o",
    linewidth=2.8,
    markersize=5,
    label="Actual Price"
)

plt.plot(
    plot_df["Date"],
    plot_df["prediction"],
    color="red",
    marker="D",
    linestyle="--",
    linewidth=2.8,
    markersize=5,
    label="Predicted Price"
)

plt.title("Actual vs Predicted Stock Closing Price", fontsize=19, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Closing Price", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig("actual_vs_predicted_overlay.png", dpi=300)
print("Overlay graph saved as actual_vs_predicted_overlay.png")


# --------------------------------------------------------------
# STEP 19 : Stop Spark Session
# --------------------------------------------------------------
spark.stop()
print("\nSpark Session stopped successfully.")
print("Project completed successfully.")


# ==============================================================
# RESULT :
# The stock closing price was predicted successfully using
# PySpark Linear Regression. The model was evaluated using
# RMSE and R2 score, and three attractive graphs were created:
# 1. Actual stock price trend
# 2. Predicted stock price trend
# 3. Actual vs Predicted overlay graph
# ==============================================================