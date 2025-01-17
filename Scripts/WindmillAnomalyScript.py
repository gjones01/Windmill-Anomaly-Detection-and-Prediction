import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths for datasets and combined file
datasets_folder_path = "/Users/gerryjr/Downloads/archive-6/Wind Farm A/datasets"
combined_save_path = os.path.join(datasets_folder_path, "combined_data_farm_a.csv")

# See if combined file already exists
if os.path.exists(combined_save_path):
    print("Combined data file found. Loading...")
    combined_df = pd.read_csv(combined_save_path)
else:
    print("No combined data file found. Processing raw files...")

    combined_data = []

    # Only process CSV files
    for file_name in os.listdir(datasets_folder_path):
        if file_name.endswith(".csv") and file_name.startswith("comma_"):
            file_path = os.path.join(datasets_folder_path, file_name)
            print(f"Processing file: {file_path}")

            try:
                df = pd.read_csv(file_path, engine='python')

                # Add columns to identify farm and windmill ID
                df["farm"] = "Wind Farm A"
                df["windmill_id"] = file_name.split(".")[0]
                combined_data.append(df)

            except Exception as e:
                print(f"Error reading file {file_name}: {e}")

    # Combine all DataFrames into one
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)

        # Save CSV
        combined_df.to_csv(combined_save_path, index=False)
        print(f"Combined data saved to {combined_save_path}")
    else:
        print("No data found to combine.")

#Visualization

df = pd.read_csv('/Users/gerryjr/Downloads/archive-6/Wind Farm A/datasets/combined_data_farm_a.csv')

#Scatter Plot
scatter_df = df[['reactive_power_27_avg', 'wind_speed_3_avg']].dropna()

plt.figure(figsize=(10, 6))
plt.scatter(scatter_df['wind_speed_3_avg'], scatter_df['reactive_power_27_avg'], alpha=0.3, s=10, edgecolor='k')
plt.title('Scatter Plot: Reactive Power (27) vs Wind Speed (3)', fontsize=16)
plt.xlabel('Wind Speed 3 (Average)', fontsize=12)
plt.ylabel('Reactive Power 27 (Average)', fontsize=12)
plt.grid(True)
plt.show()

#Box Plot (Sensor 2)
box_df = df[['windmill_id', 'sensor_2_avg']]
box_df = box_df.dropna(subset=['sensor_2_avg'])
plt.figure(figsize=(14,8))
sns.boxplot(data=box_df, x='windmill_id', y='sensor_2_avg')
plt.title('Box Plot: Sensor 2 Across Windmills', fontsize=16)
plt.xlabel("Windmill ID", fontsize=12)
plt.ylabel("Sensor 2 (average)", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#3D Scatter 

# Ensure 'time_stamp' is numeric for 3D plotting
df['time_stamp'] = pd.to_datetime(df['time_stamp'], errors='coerce')  # Convert to datetime
df = df.dropna(subset=['time_stamp'])  # Remove rows with invalid dates

# Convert datetime to a numeric format (epoch time)
df['time_stamp_numeric'] = df['time_stamp'].apply(lambda x: x.timestamp())

# Ensure other columns are numeric
df['wind_speed_3_avg'] = pd.to_numeric(df['wind_speed_3_avg'], errors='coerce')
df['reactive_power_27_avg'] = pd.to_numeric(df['reactive_power_27_avg'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['wind_speed_3_avg', 'reactive_power_27_avg', 'time_stamp_numeric'])

# Downsample the data for visualization
sampler = df.sample(n=500) 
sampler = sampler.sort_values('time_stamp_numeric')  

# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
figure = plt.figure(figsize=(12, 8))
axes = figure.add_subplot(111, projection='3d')
scatter = axes.scatter(
    sampler['time_stamp_numeric'],
    sampler['wind_speed_3_avg'],
    sampler['reactive_power_27_avg'],
    c=sampler['wind_speed_3_avg'],  # Color based on wind speed
    cmap='viridis'
)
cbar = plt.colorbar(scatter, ax=axes, pad=0.1)
cbar.set_label('Wind Speed (Average)', rotation=270, labelpad=15)
axes.set_xlabel('Time (Epoch)')
axes.set_ylabel('Wind Speed (Average)')
axes.set_zlabel('Reactive Power 27 (Average)')
axes.set_title('3D Time-Series Trend: Wind Speed vs Reactive Power')
plt.show()

#Machine Learning

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import numpy as np
import os
import matplotlib.pyplot as plt

# SparkSession
spark = SparkSession.builder \
    .appName("Windmill Anomaly Detection") \
    .getOrCreate()

# Load data
combined_csv_path = "/Users/gerryjr/Downloads/archive-6/Wind Farm A/datasets/combined_data_farm_a.csv"
df_spark = spark.read.csv(combined_csv_path, header=True, inferSchema=True)

# Define features for clustering. NOTE: Too many features can cause overfitting, so only relevant readings are featured here. 
# This prevents data sparsing and issues of a model not learning meangful patterns.
features = [
    "wind_speed_3_avg", 
    "reactive_power_27_avg", 
    "sensor_2_avg", 
    "sensor_5_avg", 
    "sensor_18_avg"
]

# Reload anomaly files
anomalies_save_path = "/Users/gerryjr/Downloads/archive-6/Wind Farm A/datasets/anomalies"
top_5_anomalies_all_path = "/Users/gerryjr/Downloads/archive-6/Wind Farm A/datasets/top_5_anomalies_farm_a"

# Visualize distance from centroids (this is for ML)
print("Generating histogram to visualize the distribution of distances...")
if os.path.exists(anomalies_save_path):
    anomalies_df = spark.read.csv(anomalies_save_path, header=True, inferSchema=True)
    distances = anomalies_df.select('distance_from_centroid').toPandas()

    # Plot the histogram
    plt.hist(distances['distance_from_centroid'], bins=50, edgecolor='black')
    plt.xlabel('Distance from Centroid')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution (Preliminary Analysis)')
    plt.show()
else:
    print("No anomalies file found. Skipping histogram for saved anomalies.")

# Process each windmill
def process_windmill(windmill_id, df):
    print(f"Processing Windmill: {windmill_id} please stand by...")
    windmill_df = df.filter(df['windmill_id'] == windmill_id)
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    assembler_df = assembler.transform(windmill_df)

    # Standardize features NOTE: This is important so all sensor readings are scaled properly. This prevents innacurate pattern recognition
    # and one feature dominating others.
    scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withMean=False)
    scaler_model = scaler.fit(assembler_df)
    scaled_df = scaler_model.transform(assembler_df)

    # Train K-Means
    kmeans = KMeans(featuresCol='scaled_features', predictionCol='cluster', k=3)
    kmeans_model = kmeans.fit(scaled_df)
    clustered_df = kmeans_model.transform(scaled_df)

    # Evaluate clustering
    evaluator = ClusteringEvaluator(featuresCol='scaled_features', predictionCol='cluster', metricName='silhouette')
    silhouette_score = evaluator.evaluate(clustered_df)
    print(f"Silhouette Score for Windmill {windmill_id}: {silhouette_score}")

    # Define distance calculation for anomalies
    centroids = np.array(kmeans_model.clusterCenters())

    def calculate_distance(features, cluster):
        return float(np.linalg.norm(features - centroids[cluster]))

    distance_udf = udf(calculate_distance, DoubleType())
    clustered_df = clustered_df.withColumn('distance_from_centroid', distance_udf('scaled_features', 'cluster'))

    # Filter anomalies based on threshold
    threshold = 5.25
    anomalies_df = clustered_df.filter(clustered_df['distance_from_centroid'] > threshold)

    # Select top 5 anomalies for this windmill
    top_5_anomalies_df = anomalies_df.orderBy("distance_from_centroid", ascending=False).limit(5)

    return clustered_df, anomalies_df, top_5_anomalies_df


# Unique windmill IDs from the DataFrame
windmills = df_spark.select("windmill_id").distinct().rdd.flatMap(lambda x: x).collect()

# Check if anomaly file exists
if os.path.exists(anomalies_save_path):
    print("Anomalies file found. Loading anomalies...")
    anomalies_df = spark.read.csv(anomalies_save_path, header=True, inferSchema=True)
    anomalies_df.show()  # Display top 10 anomalies
else:
    print("No anomalies file found. Processing windmills...")

    all_anomalies = []
    top_5_anomalies = []

    # Process each windmill
    for windmill_id in windmills:
        clustered_df, anomalies_df, top_5_anomalies_df = process_windmill(windmill_id, df_spark)
        all_anomalies.append(anomalies_df)
        top_5_anomalies.append(top_5_anomalies_df)

        # Show anomalies for this windmill
        print(f"Top 5 Anomalies for Windmill {windmill_id}:")
        top_5_anomalies_df.select(
            "windmill_id", 
            "wind_speed_3_avg", 
            "reactive_power_27_avg", 
            "sensor_2_avg", 
            "sensor_5_avg", 
            "sensor_18_avg", 
            "distance_from_centroid"
        ).show(5)

    # Combine all anomalies into one DataFrame
    if all_anomalies:
        combined_anomalies = all_anomalies[0]
        for anomalies_df in all_anomalies[1:]:
            combined_anomalies = combined_anomalies.union(anomalies_df)

        # Drop unsupported columns 
        combined_anomalies = combined_anomalies.select(
            "windmill_id", "wind_speed_3_avg", "reactive_power_27_avg", "sensor_2_avg", 
            "sensor_5_avg", "sensor_18_avg", "distance_from_centroid"
        )

        # Save combined anomalies 
        combined_anomalies.coalesce(1).write.csv(anomalies_save_path, header=True, mode="overwrite")
        print(f"Anomalies saved to {anomalies_save_path}")
    else:
        print("No anomalies found.")

    # Combine top 5 anomalies for all windmills into one DataFrame
    if top_5_anomalies:
        combined_top_5_anomalies = top_5_anomalies[0]
        for anomaly_df in top_5_anomalies[1:]:
            combined_top_5_anomalies = combined_top_5_anomalies.union(anomaly_df)

        # Drop unsupported columns 
        combined_top_5_anomalies = combined_top_5_anomalies.select(
            "windmill_id", "wind_speed_3_avg", "reactive_power_27_avg", "sensor_2_avg", 
            "sensor_5_avg", "sensor_18_avg", "distance_from_centroid"
        )

        # Save top 5 anomalies for all windmills to CSV
        combined_top_5_anomalies.coalesce(1).write.csv(top_5_anomalies_all_path, header=True, mode="overwrite")
        print(f"Top 5 anomalies per windmill saved to {top_5_anomalies_all_path}")
    else:
        print("No top 5 anomalies found for any windmills.")

#LOGISTIC REGRESSION
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import when, col
from pyspark.sql.types import DoubleType
import numpy as np

# Threshold for anomalies
threshold = 5.25

# Ensure distance_from_centroid exists
if "distance_from_centroid" not in df_spark.columns:
    print("Generating distance_from_centroid using K-Means clustering...")

    # Feature assembly
    assembler = VectorAssembler(inputCols=features, outputCol="assembled_features")
    df_with_features = assembler.transform(df_spark)

    # Standardize features
    scaler = StandardScaler(inputCol="assembled_features", outputCol="scaled_features_kmeans", withMean=False)
    scaler_model = scaler.fit(df_with_features)
    scaled_df = scaler_model.transform(df_with_features)

    # Train K-Means
    kmeans = KMeans(featuresCol="scaled_features_kmeans", predictionCol="cluster", k=3)
    kmeans_model = kmeans.fit(scaled_df)
    clustered_df = kmeans_model.transform(scaled_df)

    # Define distance calculation
    centroids = np.array(kmeans_model.clusterCenters())

    def calculate_distance(features, cluster):
        return float(np.linalg.norm(features - centroids[cluster]))

    distance_udf = udf(calculate_distance, DoubleType())
    df_spark = clustered_df.withColumn("distance_from_centroid", distance_udf("scaled_features_kmeans", "cluster"))

# Add label column for logistic regression
df_spark = df_spark.withColumn(
    "label", when(df_spark["distance_from_centroid"] > threshold, 1).otherwise(0)
)

# Assemble features for logistic regression
assembler = VectorAssembler(inputCols=features, outputCol="assembled_features_lr")
df_with_features = assembler.transform(df_spark)

# Standardize features for logistic regression
scaler = StandardScaler(inputCol="assembled_features_lr", outputCol="scaled_features_lr", withMean=False)
scaler_model = scaler.fit(df_with_features)
df_scaled = scaler_model.transform(df_with_features)

# Split the data into training and testing sets
train_data, test_data = df_scaled.randomSplit([0.7, 0.3], seed=42)

# Initialize Logistic Regression
log_reg = LogisticRegression(featuresCol="scaled_features_lr", labelCol="label")

# Train the model
log_reg_model = log_reg.fit(train_data)

# Make predictions for model
predictions = log_reg_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
auc_roc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
print(f"Model Accuracy (AUC-ROC): {auc_roc}")

# Get unique windmill IDs
windmills = df_spark.select("windmill_id").distinct().rdd.flatMap(lambda x: x).collect()

# Display top 5 anomalies for each windmill
print("\nTop 5 anomalies for each windmill:")
for windmill_id in windmills:
    windmill_anomalies = predictions.filter(
        (col("windmill_id") == windmill_id) & (col("label") == 1)
    ).orderBy(col("probability").desc())
    top_5_anomalies = windmill_anomalies.select(
        "windmill_id", "time_stamp", "distance_from_centroid", "probability"
    ).limit(5)
    print(f"\nWindmill {windmill_id}:")
    top_5_anomalies.show()
