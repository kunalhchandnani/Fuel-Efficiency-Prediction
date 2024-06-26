# **Fuel-K-Prediction**

This project leverages machine learning techniques to predict fuel efficiences using a dataset containing various automobile features. The workflow includes data preprocessing, exploratory data analysis (EDA), and building predictive models with deep learning. The project aims to provide a robust model for predicting fuel prices based on the characteristics of vehicles.

**Features**

1. **Data Loading and Cleaning:**
* Handles missing values and incorrect data types.
* Removes anomalies and converts data types to ensure consistency.
2. **Exploratory Data Analysis (EDA):**
* Provides visual insights into the relationships between various features and fuel efficiency.
* Utilizes bar plots and heatmaps for data visualization.
3. **Predictive Modeling:**
* Implements a deep learning model using TensorFlow and Keras.
* Model architecture includes multiple dense layers with ReLU activation, batch normalization, and dropout for regularization.
4.** Model Training and Evaluation:**
* Uses MAE (Mean Absolute Error) and MAPE (Mean Absolute Percentage Error) as evaluation metrics.
* Training history is plotted to visualize loss and performance over epochs.

**Data**

The project utilizes the **auto-mpg.csv** dataset, which includes features such as:
* mpg: Miles per gallon
* cylinders: Number of cylinders
* displacement: Engine displacement
* horsepower: Engine horsepower
* weight: Vehicle weight
* acceleration: Time to accelerate from 0 to 60 mph
* model_year: Model year of the car
* origin: Country of origin
* name: Car name
  
**Modeling and Evaluation**

1. **Model Architecture**

* **Input Layer:** Accepts 6 features after dropping mpg and car name.
* **Dense Layers:** Two dense layers with 256 units and ReLU activation.
* **Batch Normalization:** Applied after each dense layer for normalization.
* **Dropout:** A dropout layer with a 0.3 rate to prevent overfitting.
* **Output Layer:** A single unit with ReLU activation to predict the mpg.

  
2. **Model Training**

* **Data Split:** The dataset is split into training and validation sets using an 80-20 split.
* **Batch Processing:** Utilizes TensorFlow’s Dataset API for efficient batch processing.
* **Training:** The model is trained over 50 epochs, optimizing with the Adam optimizer and minimizing MAE.

  
3. **Evaluation Metrics**

* **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions.
* **Mean Absolute Percentage Error (MAPE):** Evaluates the percentage error in predictions.


**Training Results**

* The training history is visualized to show the model's performance across epochs.
* Plots include loss and MAPE for both training and validation sets.

![image](https://github.com/kunalhchandnani/Fuel-Price-Prediction/assets/88874426/38c0ac96-8451-45cb-b750-3e6147e2fd94)
![image](https://github.com/kunalhchandnani/Fuel-Price-Prediction/assets/88874426/9aedc0e9-908b-48da-a517-87da9c144a5d)

