import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,SeparableConv2D, MaxPooling2D, Flatten, Dense, Input, LeakyReLU
import cv2
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
import json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Clear session
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

image_size = (64, 64)

# Load the CSV file
current_directory = os.getcwd()

# Load CSV file containing image labels and filenames
csv_file_path = os.path.join(current_directory, 'Required Files/DataSet/Main/Dataset.csv')
df = pd.read_csv(csv_file_path)

# Define the folder path where images are stored
image_folder_path = os.path.join(current_directory, 'Required Files/DataSet/Main')


# Function to load and preprocess images
def load_and_preprocess_image(filename):
    img_path = os.path.join(image_folder_path, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found at {img_path}")
        return None
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, image_size)
    img_array = resized_img / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array



# Data augmentation 
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Load and preprocess all images
images = np.array([load_and_preprocess_image(filename) for filename in df['filename']])

# Extract labels and one-hot encode them
labels = df.drop(columns=['filename']).values
labels_int = np.argmax(labels, axis=1) 
labels = to_categorical(labels_int)

# Automate label mapping using column names
column_names = df.columns[1:].tolist()
label_map = {index: column_names[index] for index in range(len(column_names))}
print("Label mapping:")
print(label_map)

# Check class distribution before duplication
label_counts_before = pd.Series(np.argmax(labels, axis=1)).value_counts()
print("\nClass distribution before duplication:")
print(label_counts_before)

# Handle classes with very few samples by duplicating
min_samples_per_class = 50
for class_index in np.unique(labels_int):
    class_samples = np.sum(labels_int == class_index)
    if class_samples < min_samples_per_class:

        class_indices = np.where(labels_int == class_index)[0]
        num_samples_needed = min_samples_per_class - class_samples
        duplicated_indices = np.random.choice(class_indices, num_samples_needed, replace=True)
        images = np.concatenate([images, images[duplicated_indices]])
        
        labels = np.concatenate([labels, labels[duplicated_indices]])
        labels_int = np.concatenate([labels_int, labels_int[duplicated_indices]])

# Check class distribution after duplication
label_counts_after = pd.Series(np.argmax(labels, axis=1)).value_counts()
print("/nClass distribution after duplication:")
print(label_counts_after)

# Split data into training, validation, and test sets using StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

#Splitting for standard cnn and seprable cnn
for train_index, temp_index in sss.split(images, labels_int):
    x_train, x_temp = images[train_index], images[temp_index]
    y_train, y_temp = labels[train_index], labels[temp_index]
    labels_int_train, labels_int_temp = labels_int[train_index], labels_int[temp_index]

sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_index, test_index in sss_val_test.split(x_temp, labels_int_temp):
    x_val, x_test = x_temp[val_index], x_temp[test_index]
    y_val, y_test = y_temp[val_index], y_temp[test_index]
    

# Reshape images for model input
x_train = x_train.reshape(-1, image_size[0], image_size[1], 1)
x_val = x_val.reshape(-1, image_size[0], image_size[1], 1)
x_test = x_test.reshape(-1, image_size[0], image_size[1], 1)



# Determine the maximum number of samples in any category
max_samples = label_counts_before.max()

# Augment the data to balance the dataset
augmented_images = []
augmented_labels = []

# Define the number of batches to generate per class
batches_per_class = 10  

for class_index in range(labels.shape[1]):
    class_images = x_train[np.argmax(y_train, axis=1) == class_index]
    class_labels = y_train[np.argmax(y_train, axis=1) == class_index]
    current_samples = len(class_images)
    samples_needed = 300
    
    if samples_needed > 0:
        batches_needed = (samples_needed + current_samples - 1) // current_samples
        
        for _ in range(batches_needed * batches_per_class): 
            batch = next(datagen.flow(class_images, class_labels, batch_size=len(class_images)))
            augmented_images.extend(batch[0])
            augmented_labels.extend(batch[1])

# Convert lists to numpy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Concatenate augmented data with original data
x_train = np.concatenate([x_train, augmented_images])
y_train = np.concatenate([y_train, augmented_labels])

# Shuffle the data after augmentation
shuffle_index = np.random.permutation(len(x_train))
x_train = x_train[shuffle_index]
y_train = y_train[shuffle_index]

#-------------------------xception model---------------------
image_size = (64, 64)

# Load the CSV file
current_directory = os.getcwd()

# Load CSV file containing image labels and filenames
csv_file_path = os.path.join(current_directory, 'Required Files/DataSet/Main/Dataset.csv')
df = pd.read_csv(csv_file_path)

# Define the folder path where images are stored
image_folder_path = os.path.join(current_directory, 'Required Files/DataSet/Main')


def load_and_preprocess_image_for_xcnn(filename):
    img_path = os.path.join(image_folder_path, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image not found at {img_path}")
        return None
    resized_img = cv2.resize(img, image_size)
    img_array = resized_img / 255.0
    img_array = np.stack((img_array,) * 3, axis=-1)  
    return img_array

images_xcnn = np.array([load_and_preprocess_image_for_xcnn(filename) for filename in df['filename']])
labels_xcnn = df.drop(columns=['filename']).values
labels_int_xcnn = np.argmax(labels_xcnn, axis=1) 
labels_xcnn = to_categorical(labels_int_xcnn)

min_samples_per_class = 20
for class_index in np.unique(labels_int_xcnn):
    class_samples = np.sum(labels_int_xcnn == class_index)
    if (class_samples < min_samples_per_class):
        class_indices = np.where(labels_int_xcnn == class_index)[0]
        num_samples_needed = min_samples_per_class - class_samples
        duplicated_indices = np.random.choice(class_indices, num_samples_needed, replace=True)
        images_xcnn = np.concatenate([images_xcnn, images_xcnn[duplicated_indices]])
        labels_xcnn = np.concatenate([labels_xcnn, labels_xcnn[duplicated_indices]])
        labels_int_xcnn = np.concatenate([labels_int_xcnn, labels_int_xcnn[duplicated_indices]])
        

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

for train_index, temp_index in sss.split(images_xcnn, labels_int_xcnn):
    x_train_xcnn, x_temp_xcnn = images_xcnn[train_index], images_xcnn[temp_index]
    y_train_xcnn, y_temp_xcnn = labels_xcnn[train_index], labels_xcnn[temp_index]
    labels_int_train_xcnn, labels_int_temp_xcnn = labels_int_xcnn[train_index], labels_int_xcnn[temp_index]

sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_index, test_index in sss_val_test.split(x_temp_xcnn, labels_int_temp_xcnn):
    x_val_xcnn, x_test_xcnn = x_temp_xcnn[val_index], x_temp_xcnn[test_index]
    y_val_xcnn, y_test_xcnn = y_temp_xcnn[val_index], y_temp_xcnn[test_index]
    

x_train_xcnn= x_train_xcnn.reshape(-1, image_size[0], image_size[1], 3)
x_val_xcnn = x_val_xcnn.reshape(-1, image_size[0], image_size[1], 3)
x_test_xcnn = x_test_xcnn.reshape(-1, image_size[0], image_size[1], 3)

augmented_images_xcnn = []
augmented_labels_xcnn= []

# Define the number of batches to generate per class
batches_per_class = 10  

for class_index in range(labels_xcnn.shape[1]):
    class_images = x_train_xcnn[np.argmax(y_train_xcnn, axis=1) == class_index]
    class_labels = y_train_xcnn[np.argmax(y_train_xcnn, axis=1) == class_index]
    current_samples = len(class_images)
    samples_needed = 100
    
    if samples_needed > 0:
        batches_needed = (samples_needed + current_samples - 1) // current_samples
        
        for _ in range(batches_needed * batches_per_class):  
            batch = next(datagen.flow(class_images, class_labels, batch_size=len(class_images)))
            augmented_images_xcnn.extend(batch[0])
            augmented_labels_xcnn.extend(batch[1])

# Convert lists to numpy arrays
augmented_images_xcnn = np.array(augmented_images_xcnn)
augmented_labels_xcnn = np.array(augmented_labels_xcnn)

# Concatenate augmented data with original data
x_train_xcnn = np.concatenate([x_train_xcnn, augmented_images_xcnn])
y_train_xcnn = np.concatenate([y_train_xcnn, augmented_labels_xcnn])

# Shuffle the data after augmentation
shuffle_index = np.random.permutation(len(x_train_xcnn))
x_train_xcnn = x_train_xcnn[shuffle_index]
y_train_xcnn = y_train_xcnn[shuffle_index]
#-----------------------------------------------------------------------
folder_path = os.path.join(os.getcwd(), 'Required Files')

# Load Model and History for Model 1
print('--------------Model 1 - Standard CNN Model----------------')
model1 = tf.keras.models.load_model(os.path.join(folder_path, 'Standard_CNN_Trained_Model.keras'))
with open(os.path.join(folder_path, 'history_Standard_CNN.json'), 'r') as file:
    history1 = json.load(file)

# Load Model and History for Model 2
print('--------------Model 2 - CNN Model using SeparableConv2D----------------')
model2 = tf.keras.models.load_model(os.path.join(folder_path, 'Separable_CNN_trained_Model.keras'))
with open(os.path.join(folder_path, 'history_Separable_CNN.json'), 'r') as file:
    history2 = json.load(file)

# Load Model and History for Model 3
print('--------------Model 3 - CNN Model using Xception----------------')
model3 = tf.keras.models.load_model(os.path.join(folder_path, 'Xception_trained_model.keras'))
with open(os.path.join(folder_path, 'history_Xception_trained_Model.json'), 'r') as file:
    history3 = json.load(file)
#---------------------- Evaluation and Comparison of models---------------------
accuracy1 = history1['accuracy']
accuracy2 = history2['accuracy']
accuracy3 = history3['accuracy']

loss1 = history1['loss']
loss2 = history2['loss']
loss3 = history3['loss']

# Create subplots for training accuracy and loss
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

# Plotting training accuracy
ax1.plot(accuracy1, label='Standard CNN - Training Accuracy')
ax1.plot(accuracy2, label='Separable CNN - Training Accuracy')
ax1.plot(accuracy3, label='Xception - Training Accuracy')
ax1.set_title('Model Training Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Plotting training loss
ax2.plot(loss1, label='Standard CNN - Training Loss')
ax2.plot(loss2, label='Separable CNN - Training Loss')
ax2.plot(loss3, label='Xception - Training Loss')
ax2.set_title('Model Training Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()



#--------------------------------Accuracy-----------------------------
y_test = np.argmax(y_test,axis=1)  

# Predictions for each model 
y_pred1 = np.argmax(model1.predict(x_test), axis=1)
y_pred2 = np.argmax(model2.predict(x_test), axis=1)
y_pred3 = np.argmax(model3.predict(x_test_xcnn),axis=1)
y_test_xcnn = np.argmax(y_test_xcnn,axis=1)

# Calculate metrics for each model
accuracy1 = accuracy_score(y_test, y_pred1)
precision1 = precision_score(y_test, y_pred1, average='weighted')
recall1 = recall_score(y_test, y_pred1, average='weighted')
f1_1 = f1_score(y_test, y_pred1, average='weighted')

accuracy2 = accuracy_score(y_test, y_pred2)
precision2 = precision_score(y_test, y_pred2, average='weighted')
recall2 = recall_score(y_test, y_pred2, average='weighted')
f1_2 = f1_score(y_test, y_pred2, average='weighted')

accuracy3 = accuracy_score(y_test_xcnn, y_pred3)
precision3 = precision_score(y_test_xcnn, y_pred3, average='weighted')
recall3 = recall_score(y_test_xcnn, y_pred3, average='weighted')
f1_3 = f1_score(y_test_xcnn, y_pred3, average='weighted')


metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
model1_metrics = [accuracy1, precision1, recall1, f1_1]
model2_metrics = [accuracy2, precision2, recall2, f1_2]
model3_metrics = [accuracy3, precision3, recall3, f1_3]


x = np.arange(len(metrics))  
width = 0.2  
# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x - width, model1_metrics, width, label='Standard CNN')
rects2 = ax.bar(x, model2_metrics, width, label='Separable CNN')
rects3 = ax.bar(x + width, model3_metrics, width, label='Xception')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 3)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()


#----------------------------Roc Curves----------------------------------------------

y_pred_prob1 = model1.predict(x_test)
y_pred_prob2 = model2.predict(x_test)
y_pred_prob3 = model3.predict(x_test_xcnn)

# Convert y_test to categorical 
n_classes = y_pred_prob1.shape[1]
y_test_categorical = to_categorical(y_test, num_classes=n_classes)
y_test_categorical_xcnn=to_categorical(y_test_xcnn, num_classes=n_classes)

# Compute ROC curve and ROC area for each model
fpr1, tpr1, _ = roc_curve(y_test_categorical.ravel(), y_pred_prob1.ravel())
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, _ = roc_curve(y_test_categorical.ravel(), y_pred_prob2.ravel())
roc_auc2 = auc(fpr2, tpr2)

fpr3, tpr3, _ = roc_curve(y_test_categorical_xcnn.ravel(), y_pred_prob3.ravel())
roc_auc3 = auc(fpr3, tpr3)

# Plotting ROC curves
plt.figure(figsize=(10, 8))

plt.plot(fpr1, tpr1, label=f'Standard CNN (AUC = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, label=f'Separable CNN (AUC = {roc_auc2:.2f})')
plt.plot(fpr3, tpr3, label=f'Xception (AUC = {roc_auc3:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)

plt.show()