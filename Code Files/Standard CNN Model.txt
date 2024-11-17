#Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, ReLU,LeakyReLU
import cv2
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from itertools import cycle
from keras.regularizers import l1, l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import json

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

# Function for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess the images
images = np.array([load_and_preprocess_image(filename) for filename in df['filename']])

# One hot encoding the Labels
labels = df.drop(columns=['filename']).values
labels_int = np.argmax(labels, axis=1)  
labels = to_categorical(labels_int)

# Defining label mapping using column names
column_names = df.columns[1:].tolist()
label_map = {index: column_names[index] for index in range(len(column_names))}
print("Label mapping:")
print(label_map)

# Distribution before duplication
label_counts_before = pd.Series(np.argmax(labels, axis=1)).value_counts()
label_names_before = label_counts_before.index.map(label_map)

# Data frame to display the counts for each category
label_counts_before_df = pd.DataFrame({
    'Label Name': label_names_before,
    'Count': label_counts_before.values
})

print("Class distribution before duplication:")
print(label_counts_before_df)

# Handle classes with very few samples by duplicating
min_samples_per_class = 50
for class_index in np.unique(labels_int):
    class_samples = np.sum(labels_int == class_index)
    if class_samples < min_samples_per_class:
        # Duplicate samples for this class
        class_indices = np.where(labels_int == class_index)[0]
        num_samples_needed = min_samples_per_class - class_samples
        duplicated_indices = np.random.choice(class_indices, num_samples_needed, replace=True)
        images = np.concatenate([images, images[duplicated_indices]])
        labels = np.concatenate([labels, labels[duplicated_indices]])
        labels_int = np.concatenate([labels_int, labels_int[duplicated_indices]])

# Distribution after duplication
label_counts_after = pd.Series(np.argmax(labels, axis=1)).value_counts()
label_names_after = label_counts_after.index.map(label_map)


label_counts_after_df = pd.DataFrame({
    'Label Name': label_names_after,
    'Count': label_counts_after.values
})

print("Class distribution after duplication:")
print(label_counts_after_df)

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
x_train = x_train.reshape(-1, image_size[0], image_size[1], 1)
x_val = x_val.reshape(-1, image_size[0], image_size[1], 1)
x_test = x_test.reshape(-1, image_size[0], image_size[1], 1)


# Display number of images in each set
print(f"No. of in the training set: {x_train.shape[0]}")
print(f"No. of images in  the validation set: {x_val.shape[0]}")
print(f"No. of images in  the test set: {x_test.shape[0]}")

# Display number of samples in each category before augmentation using bar plot
plt.figure(figsize=(10, 5))
bars = label_counts_after.plot(kind='bar', color='skyblue')
for bar in bars.patches:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, int(height), ha='center', va='bottom')
plt.xticks(ticks=range(len(label_names_after)), labels=label_names_after, rotation=45, ha='right')
plt.title('Number of Samples in Each Category before Augmentation')
plt.xlabel('Category')
plt.ylabel('Number of Samples')
plt.tight_layout()  # Adjust layout to make room for x-axis labels
plt.show()

# The maximum number of samples in each category
max_samples = label_counts_before.max()


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
x_train = np.concatenate([x_train, augmented_images])
y_train = np.concatenate([y_train, augmented_labels])

# Shuffle the data after augmentation
shuffle_index = np.random.permutation(len(x_train))
x_train = x_train[shuffle_index]
y_train = y_train[shuffle_index]

# Compute label counts in the training set after augmentation
label_counts_after_augmentation = pd.Series(np.argmax(y_train, axis=1)).value_counts()
label_names_after_augmentation = label_counts_after_augmentation.index.map(label_map)

# DataFrame for label counts after augmentation
label_counts_after_augmentation_df = pd.DataFrame({
    'Label Name': label_names_after_augmentation,
    'Count': label_counts_after_augmentation.values
})

print("Class distribution after augmentation:")
print(label_counts_after_augmentation_df)

# Plotting
plt.figure(figsize=(10, 5))
bars = label_counts_after_augmentation.plot(kind='bar', color='lightcoral')

# Add labels on top of the bars
for bar in bars.patches:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, int(height), ha='center', va='bottom')

# Set x-ticks to display category names
plt.xticks(ticks=range(len(label_names_after_augmentation)), labels=label_names_after_augmentation, rotation=45, ha='right')
plt.title('Number of Samples in Each Category After Augmentation')
plt.xlabel('Category')
plt.ylabel('Number of Samples')
plt.tight_layout() 
plt.show()


# Input layer
input_layer = Input(shape=(image_size[0], image_size[1], 1))


# Convolution Block 1
x = Conv2D(64, (3, 3), padding='same')(input_layer)
x = ReLU()(x)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Convolution Block 2
x = Conv2D(128, (3, 3), padding='same')(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Convolution Block 3
x = Conv2D(256, (3, 3), padding='same')(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Convolution Block 4
x = Conv2D(512, (3, 3), padding='same')(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.5)(x)

x = Flatten()(x)

# Dense layers
x = Dense(2048)(x)
x = ReLU(negative_slope=0.1)(x)
x = Dense(1024)(x)
x = ReLU(negative_slope=0.1)(x)
x = Dense(512)(x)
x = ReLU()(x)

# Output layer
output_layer = Dense(labels.shape[1], activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the summary of the model
model.summary()

# Compute class weights for handling class imabalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weights = dict(enumerate(class_weights))

# Define callbacks for training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Train the model 
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), 
                    epochs=50, 
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
model_save_path = os.path.join(save_directory, 'Standard_CNN_Trained_Model.keras')

model.save(model_save_path)

# Save the training history to a JSON file
history_save_path = os.path.join(save_directory, 'history_Standard_CNN.json')
with open(history_save_path, 'w') as file:
    json.dump(history.history, file)


