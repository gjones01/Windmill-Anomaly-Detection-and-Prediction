Windmill Anomaly Detection and Prediction

PROJECT OBJECTIVE:
- Detect anomalies in windmill sensor readings.
- Predict the likelihood of anomalies being valid instead of false positives.
- Showcase proficiency in PySpark, cluster modeling, and predictive modeling.
- Visualize insights with Matplotlib for insightful figures.

TOOLS:
- Language: Python
- - Framework: PySpark
- Machine Learning Models: K-Means and Logistic Regression
- Visualization Library: Matplotlib
- Other: Pandas, NumPy

DATASET INFO:
- The dataset contains sensor readings from over 20 windmills in a wind farm.
- Each windmill gathers data every 10 minutes from over 50 sensors. These sensors measure:
Wind speed
Reactive power
Timestamps for recorded readings
Temperature of various components
Rotor RPM
Voltage
Grid frequency
Average current

Data Source: This dataset was obtained from the Fraunhofer Society, a German publicly-owned research organization focused on applied science.

PROCESSING STEPS:
- Handle missing values.
- Standardize sensor readings.
- Aggregate windmill ID data for analysis.

METHOD:
1. DATA PROCESSING
- Clean the dataset and combine raw CSV files into a single dataset for easier analysis.
- Add columns to identify windmill IDs and timestamps.

2. VISUALIZATION
- Use Matplotlib to gain general insights into windmill behavior:

I. Scatter Plot (ScatterPlotReactVSWind.png):
A scatter plot of average wind speed and reactive power for all windmills helps identify nominal operating windows. For instance, as wind speed increases to around 5 m/s, reactive power increases and operates between 5-10 m/s. Beyond this range, reactive power is limited for energy efficiency. Notable clusters below 0.2 kVAr at higher wind speeds indicate potential anomalies.

II. Box Plot (BoxPlotSensor2.png):
A box plot reveals if sensor 2 behaves consistently across all windmills. This sensor measures relative wind speed, considering yaw angle, and the small readings confirm consistent behavior.

III. 3D Time Series Trend (3DTimeSeriesTrend.mov):
This visualization adds a time dimension to the scatter plot, showing interactions between average wind speed, reactive power, and time. Points at the bottom or intermediate levels of the graph indicate potential anomalies, which can be explored further using ML models.

IV. Centroid Distance Plot (CentroidDist.png):
This visualization identifies a threshold for the distance from centroids, with 5.25 being a reasonable value to balance anomaly detection and energy efficiency.

3. ANOMALY DETECTION
- Use PySpark’s ML library to build a clustering model with K-Means.
- Calculate Euclidean distances from centroids. Points with distances greater than 5.25 are identified as potential anomalies.

4. LOGISTIC REGRESSION (AnomalyProbabilities.png)
- Apply Logistic Regression to classify data points as normal (0) or anomalous (1).
- Model accuracy: 93.62% (0.9362), confirming that the threshold and predictions are reliable.
- The model outputs a table of the top five anomalies per windmill, indicating windmill ID, timestamp, distance from centroid, and probability.

5. CONCLUSION
- Successfully identified anomalies in windmill sensor readings and predicted their state with high accuracy.
- Demonstrated the effectiveness of K-Means and Logistic Regression in addressing the problem.
- Created insightful visualizations to guide model engineering.

6. POTENTIAL IMPROVEMENTS
- Implement PySpark Streaming for real-time anomaly detection (dependent on real-time data availability).
- Engineer additional features, such as humidity or external temperature, to enhance model accuracy.

7. ACKNOWLEDGEMENTS
Dataset: Provided by Christian Güück, Cyriana M.A. Roelofs, and Stefan Faulstich from Fraunhofer IEE via Zenodo.


