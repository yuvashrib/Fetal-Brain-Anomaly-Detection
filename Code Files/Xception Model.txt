#Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, LeakyReLU,GlobalAveragePooling2D
import cv2
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy import interp
from itertools import cycle
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
import json


# Clear session
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
# Load the CSV file
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
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image not found at {img_path}")
        return None
    resized_img = cv2.resize(img, image_size)
    img_array = resized_img / 255.0
    img_array = np.stack((img_array,) * 3, axis=-1)  
    return img_array

# Data augmentation Function
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

# Define labels and one-hot encode them
labels = df.drop(columns=['filename']).values
labels_int = np.argmax(labels, axis=1)  # Convert to single label for each sample
labels = to_categorical(labels_int)

# label mapping using column names
column_names = df.columns[1:].tolist()
label_map = {index: column_names[index] for index in range(len(column_names))}
print("Label mapping:")
print(label_map)


# Distribution before duplication
label_counts_before = pd.Series(np.argmax(labels, axis=1)).value_counts()
print("Class distribution before duplication:")
print(label_counts_before)

# Handle classes with very few samples by duplicating
min_samples_per_class = 20
for class_index in np.unique(labels_int):
    class_samples = np.sum(labels_int == class_index)
    if (class_samples < min_samples_per_class):
        class_indices = np.where(labels_int == class_index)[0]
        num_samples_needed = min_samples_per_class - class_samples
        duplicated_indices = np.random.choice(class_indices, num_samples_needed, replace=True)
        images = np.concatenate([images, images[duplicated_indices]])
        labels = np.concatenate([labels, labels[duplicated_indices]])
        labels_int = np.concatenate([labels_int, labels_int[duplicated_indices]])

# Distribution after duplication
label_counts_after = pd.Series(np.argmax(labels, axis=1)).value_counts()
print("Class distribution after duplication:")
print(label_counts_after)

# Split data into training, validation, and test sets using StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

for train_index, temp_index in sss.split(images, labels_int):
    x_train, x_temp = images[train_index], images[temp_index]
    y_train, y_temp = labels[train_index], labels[temp_index]
    labels_int_train, labels_int_temp = labels_int[train_index], labels_int[temp_index]

sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_index, test_index in sss_val_test.split(x_temp, labels_int_temp):
    x_val, x_test = x_temp[val_index], x_temp[test_index]
    y_val, y_test = y_temp[val_index], y_temp[test_index]

# Reshape images for model input
x_train = x_train.reshape(-1, image_size[0], image_size[1], 3)
x_val = x_val.reshape(-1, image_size[0], image_size[1], 3)
x_test = x_test.reshape(-1, image_size[0], image_size[1], 3)

# Display label distribution in test set
print("Label distribution in y_test:")
print(pd.Series(np.argmax(y_test, axis=1)).value_counts())

# Display No. of images in each set
print(f"No. of images in training set: {x_train.shape[0]}")
print(f"No. of images in validation set: {x_val.shape[0]}")
print(f"No. of images in test set: {x_test.shape[0]}")

# Display No. of samples in each category before augmentation
label_counts_before = pd.Series(np.argmax(y_train, axis=1)).value_counts()
# Create a bar plot
plt.figure(figsize=(10, 5))
bars = label_counts_before.plot(kind='bar')

for bar in bars.patches:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, int(height), ha='center', va='bottom')

plt.title('Number of Samples in Each Category Before Augmentation')
plt.xlabel('Category')
plt.ylabel('Number of Samples')
plt.show()

# Determine the maximum number of samples in each category
max_samples = label_counts_before.max()


augmented_images = []
augmented_labels = []

# Define the number of batches to generate per class
batches_per_class = 10  

for class_index in range(labels.shape[1]):
    class_images = x_train[np.argmax(y_train, axis=1) == class_index]
    class_labels = y_train[np.argmax(y_train, axis=1) == class_index]
    current_samples = len(class_images)
    samples_needed = 150
    
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

# Display number of samples in each category after augmentation
label_counts_after = pd.Series(np.argmax(y_train, axis=1)).value_counts()
plt.figure(figsize=(10, 5))
ax = label_counts_after.plot(kind='bar')

plt.title('Number of Samples in Each Category After Augmentation')
plt.xlabel('Category')
plt.ylabel('Number of Samples')

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='center', 
                xytext=(0, 3),  
                textcoords='offset points')

plt.show()

# Define the input layer
input_layer = Input(shape=(image_size[0], image_size[1], 3))  # 3 channels for RGB

# Load the Xception model without the top layer and with weights pre-trained on ImageNet
base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_layer)

# Freeze the layers of the base model
#base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = Dense(128)(x)
#x = LeakyReLU(negative_slope=0.1)(x)


output_layer = Dense(labels.shape[1], activation='softmax')(x)


model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weights = dict(enumerate(class_weights))

# Define callbacks for training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Train the model with validation data
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), 
                    epochs=30, 
                    validation_data=(x_val, y_val),
                    class_weight=class_weights,
                    callbacks=[reduce_lr, early_stopping])

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')



# Generate classification report with label names
y_pred_prob = model.predict(x_test)
y_pred_labels = np.argmax(y_pred_prob, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

target_names = [label_map[i] for i in range(len(column_names))]

# Generate and print classification report
report = classification_report(y_true_labels, y_pred_labels, target_names=target_names)
print('Classification Report:')
print(report)
unique_pred_labels = np.unique(y_pred_labels)
print("Unique predicted labels:", unique_pred_labels)

unique_true_labels = np.unique(y_true_labels)
print("Unique true labels:", unique_true_labels)

# Calculate precision, recall, and F1 score
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Accuracy Score: {accuracy}')

# Plot confusion matrix with label names
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot ROC curves for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure(figsize=(12, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blue', 'red', 'green', 'black', 'pink', 'purple', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'lime', 'indigo'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {target_names[i]} (area = {roc_auc[i]:0.2f})')

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

    
save_directory = os.path.join(current_directory, 'Required Files')

# Ensure the save directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save the trained model
model_save_path = os.path.join(save_directory, 'Xception_trained_Model.keras')

model.save(model_save_path)

# Save the training history to a JSON file
history_save_path = os.path.join(save_directory, 'history_Xception_trained_Model.json')
with open(history_save_path, 'w') as file:
    json.dump(history.history, file)
