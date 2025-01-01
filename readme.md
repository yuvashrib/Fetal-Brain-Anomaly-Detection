
# Fetal Brain Anomaly Detection Using Deep Learning  

## Project Overview  
Identifying abnormalities in the prenatal brain is crucial for timely diagnosis and improved prognoses. This project aims to leverage **Convolutional Neural Networks (CNNs)** with **Separable Convolution Layers** to accurately classify fetal brain anomalies from ultrasound images.  

Using a dataset of 1,786 labeled images spanning 16 categories of fetal brain abnormalities—including Arnold-Chiari malformation, ventriculomegaly (mild, moderate, severe), intracranial tumors, and normal cases—we explore the potential of advanced deep learning techniques to improve diagnostic accuracy and efficiency.  

## Objectives  
1. **Develop robust Deep Neural Network (DNN) architectures**:  
   - Traditional CNNs.  
   - CNNs with Separable Convolution Layers.  
   - Xception architecture for enhanced performance.  
2. **Preprocess ultrasound images**:  
   - Data augmentation, normalization, and noise reduction.  
   - Splitting the dataset into training, validation, and test sets.  
3. **Train and optimize models**:  
   - Fine-tune hyperparameters for optimal performance.  
   - Compare evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.  
4. **Develop a user-friendly GUI**:  
   - Allow users to upload ultrasound images for classification.  
   - Provide confidence scores for predictions.  

## Key Features  
- **Innovative Architecture**: Comparison of traditional Conv2D, separable Conv2D, and Xception models.  
- **High Efficiency**: Use of Separable Convolution Layers to improve computational performance and spatial feature extraction.  
- **End-to-End Pipeline**: From data preprocessing to GUI deployment for real-world usability.  

## Technology and Resources  

### Tools and Libraries  
- **Programming Language**: Python  
- **Deep Learning Frameworks**:  
  - TensorFlow  
  - Keras  
- **Data Manipulation**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **Model Evaluation**: Scikit-learn  
- **Signal Processing**: SciPy  

### Computing Environment  
- **IDE**: Spyder  
- **Hardware**: High-performance GPUs for efficient training of deep learning models.  

## Dataset  
The dataset, sourced from an open repository, includes 1,786 ultrasound images labeled across 16 categories of fetal brain anomalies.
Link: https://universe.roboflow.com/hritwik-trivedi-gkgrv/fetal-brain-abnormalities-ultrasound

## Methodology  
1. **Data Preprocessing**:  
   - Data augmentation: Flip, rotate, and scale images.  
   - Normalization and noise reduction for consistent quality.  
   - Filtering and reshaping images for model compatibility.  

2. **Model Development**:  
   - Design of three DNN models:  
     - Traditional CNN with Conv2D layers.  
     - Separable CNN with depthwise separable convolutions.  
     - Xception model (pre-trained) leveraging separable convolutions.  
   - Transfer learning to enhance feature extraction.  
   - Iterative hyperparameter tuning for optimization.  

3. **Validation and Evaluation**:  
   - Metrics: Accuracy, precision, recall, F1-score, cross-validation, ROC, and AUC.  
   - ROC-AUC to optimize classification thresholds.  

4. **GUI Development**:  
   - An interactive interface for image upload and anomaly classification.  
   - Displays predicted anomaly and confidence level.  

## Results  
Through extensive evaluation, the models were compared to determine:  
- **Accuracy improvements**: Separable CNN and Xception outperformed traditional CNNs.  
- **Efficiency**: Separable convolutions demonstrated superior computational performance.  
- **Practical Utility**: The GUI enables real-time anomaly detection for clinical use.  

## Conclusion  
This project highlights the potential of deep learning in advancing prenatal care through early and accurate detection of fetal brain abnormalities. By leveraging ultrasound imaging—a cost-effective and accessible alternative to MRI—this solution aims to improve diagnostic capabilities for broader populations.  
