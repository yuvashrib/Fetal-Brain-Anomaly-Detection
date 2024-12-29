# Real-Time ECG Anomaly Detection Using Deep Neural Networks  

This repository presents a deep learning-based system for detecting anomalies in ECG (electrocardiogram) data. The project focuses on real-time classification of heartbeats as either "normal" or "abnormal" using various deep learning models and real-time data processing. The system is designed to assist in early detection of cardiac anomalies, facilitating timely medical interventions.  

---

## Project Objectives  

1. **Data Preparation**  
   - Load and preprocess ECG time series data from a publicly available dataset.  
   - Handle missing values and format data to ensure consistency across all records.  
   - Normalize data to improve model performance on multidimensional features.  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize the ECG signals to understand patterns in normal and abnormal heartbeats.  
   - Analyze data distribution and address imbalances between normal and anomalous samples.  

3. **Model Development**  
   - Develop and train four different deep learning models tailored for specific tasks:  
     - **Model 1**: Classifies individual time instances of ECG data.  
     - **Model 2**: Analyzes ECG data as a time series for sequence-based classification.  
     - **Model 3**: Implements anomaly detection using an autoencoder to identify irregular patterns.  
     - **Model 4**: Combines time series classification and anomaly detection for enhanced accuracy.  

4. **Model Evaluation**  
   - Assess model performance using key metrics such as accuracy, precision, recall, F1-score, and confusion matrix.  
   - Perform a comparative analysis to identify the best-performing model for real-time anomaly detection.  

5. **Real-Time Anomaly Detection**  
   - Develop a Python-based system for processing and classifying real-time ECG data in intervals of 5 seconds.  
   - Build an interactive web or mobile application to visualize detection results and notify users in case of anomalies.  

---

## Dataset  

The project utilizes a publicly available ECG dataset that contains labeled time series data for normal and abnormal heartbeats. The dataset includes features such as heartbeat rhythm, signal amplitude, and timing intervals.  
- [ECG Dataset Documentation](https://www.physionet.org/)  

---

## Implementation  

The implementation is divided into the following key stages:  

1. **Data Ingestion and Preprocessing**  
   - Load the dataset and perform necessary preprocessing steps.  
   - Normalize the ECG signals and split the data into training, validation, and test sets.  

2. **Model Training**  
   - Train each model using TensorFlow/Keras frameworks with appropriate hyperparameter tuning.  
   - Monitor training using early stopping and validation loss metrics to prevent overfitting.  

3. **Model Testing and Evaluation**  
   - Evaluate models on the test dataset using classification metrics and compare performance.  

4. **Deployment**  
   - Deploy the best-performing model as a REST API for real-time anomaly detection.  
   - Integrate the API into an interactive application for end-user accessibility.  

---

## Tools and Technologies  

- **Frameworks**: TensorFlow, Keras  
- **Programming Language**: Python  
- **Visualization**: Matplotlib, Seaborn  
- **Deployment**: Flask or FastAPI, Streamlit for the interactive interface  

---

## Key Takeaways  

This project demonstrates the potential of deep learning in real-time ECG anomaly detection. By leveraging multiple models and comparing their effectiveness, it provides insights into the best practices for processing and analyzing time series data in critical healthcare applications.  
