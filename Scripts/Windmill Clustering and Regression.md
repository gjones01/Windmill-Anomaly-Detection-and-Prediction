Windmill Anomalies: Detection and Prediction of Sensor Failures

PROJECT OBJECTIVE:
- Detect anomalies in windmill sensor readings
- Make predictions on the likelihood said anomalies being valid, instead of false positives.
- Showcase proficiency in PySpark, cluster modeling and predictive modeling.
- Visualizing insights with Matplotlib to create static and interactive figures.

TOOLS:
Language: Python
Framework: PySpark
Machine Learning Models: K-Means and Logistic Regression
Visualization Library: Matplotlib
Other: Pandas, Numpy

DATASET INFO:
- This dataset contains sensor readings from over 20 windmills inside of a farm.
- Each windmill gathers data every 10 minutes from over 50 sensors. Such sensors measure:
    - Wind speed 
    - Reactive Power 
    - Timestamps for the date and time of the recorded readings
    - Temperature of different components
    - Rotor RPM 
    - Voltage
    - Grid frequency
    - Average current


Data Source: This dataset was obtained from the Fraunhofer Society, a German publicly owned research organization
 focused on applied science. 

 PROCESSING STEPS:
 - Handling of missing values
 - Standardization of sensor readings
 - Aggregate windmill ID for analysis

 METHOD:
 1. DATA PROCESSING
    - Clean the dataset and combine the raw CSV files into a singular dataset for easier analysis.
    - Add columns to identify the windmills and time stamp

2. VISUALIZATION
    - I leverage a few of Matplotlib's abilities to get some general insights on the behavior of these windmills.

    I. I create a scatter plot of the average wind speed and reactive power for all of the windmills. The reasoning behind this is to
    grasp an understanding as to what the nominal operating window seems to be for this particular reactive power. In this case we can see as the wind speed increases from to around 5 m/s, the reactive power increases and operates between 5-10m/s. Beyond that the windmill's reactive power limits the power for energy efficiency. We can also see that, on average, the windmills don't drop below
    0.2 kVAr in the wind speed window of 11-24m/s.
        - Now taking a look at that same window we can see a cluster of data points below 0.2 kVAr and they disperse as the wind speed progresses. This is what we want to look at from a ML engineering standpoint to determine which of these are anomalies and the likelihood they are incorrect readings.

    II. The next visualization is a simple box plot to see if sensor 2 behaviors in a similar fashion across all of the windmills. As we can see it appears so with the this particular sensor having small readings. This specific sensor feasure relative windspeed and considers the yaw angle. In other words, it measures how the windmill interacts with incoming wind.

    III. This visualization adds another dimension of analysis to the first scatter plot. In this case we are interested in seeing how the average wind speed and average reactive power interact over the course course of time. The x-axis is time, y-axis is average wind speed and z-axis is average reactive power. Data points that are greenish-yellow indicate higher wind speeds and those readings should be showing up on the tail end of the distribution (which they do). However, what we are interested in are the greenish-blue data points that are near the bottom of the graph, underneath the distribution, because these are most likely anomalies. Concurrently we want to especially take a look at the greenish-blue data points that are intermediate between the distribution at the top and the blatant anomalies at the bottom. We will be engineering a ML model to see if these intermediate points, especially, are correct readings or anomalies in sensor reading. 

    IV. This final visualization will is crucial for how we set particular parameters for the model we will be producing. Why is this the case? In the next section


3. ANOMALY DETECTION
    - Since this is a larger dataset we would be building a ML model from, we decide to leverage PySpark's ML library. Given that
    we want to identify anomalies, it makes sense to engineer a clustering model. In this case, K-Means works well. K-Means is a form of vector quantization and will be ideal for the scenario.

    - By calculating the Euclidean distance from centroids, K-Means can be versatile in identifying outliers. In doing so I set the
    anomaly threshold at 5.25. The reasoning behind this stems from the distribution of the



