import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# Define the label map
label_map = {
    0: 'anold-chiari-malformation',
    1: 'arachnoid-cyst',
    2: 'cerebellah-hypoplasia',
    3: 'colphocephaly',
    4: 'encephalocele',
    5: 'holoprosencephaly',
    6: 'hydracenphaly',
    7: 'intracranial-hemorrdge',
    8: 'intracranial-tumor',
    9: 'm-magna',
    10: 'mild-ventriculomegaly',
    11: 'moderate-ventriculomegaly',
    12: 'normal',
    13: 'polencephaly',
    14: 'severe-ventriculomegaly',
    15: 'vein-of-galen'
}

# Load the model
folder_path = os.path.join(os.getcwd(), 'Required Files')

try:
    model = tf.keras.models.load_model(os.path.join(folder_path, 'Xception_trained_model.keras'))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Function to preprocess the image
def preprocess_image(image):
    try:
        image = image.resize((64, 64), Image.LANCZOS)  
        image_array = np.array(image) / 255.0  
        if image_array.shape[-1] != 3:
            image_array = np.stack([image_array] * 3, axis=-1)  
        image_array = np.expand_dims(image_array, axis=0)  
        print(f"Preprocessed image array shape: {image_array.shape}")
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to classify the image
def classify_image(image_array):
    try:
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions)
        accuracy = np.max(predictions) 
        return label_map[predicted_class], accuracy
    except Exception as e:
        print(f"Error classifying image: {e}")
        return "Unknown", 0.0

# Create the GUI
root = tk.Tk()
root.title("Fetal Brain Anomaly Detection")

# Set the size of the initial popup frame
root.geometry("600x565")

# Create a heading label
heading_label = tk.Label(root, text="Fetal Brain Anomaly Detection", font=("Arial", 24))
heading_label.pack(pady=20)

# Create a frame
image_frame = tk.Frame(root, width=1500, height=2000)
image_frame.pack(pady=20)

# Create a label to display the image
image_label = tk.Label(image_frame)
image_label.pack()

# Create a button to load the image
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((300, 300)) 
        image_array = preprocess_image(image)
        if image_array is not None:
            classified_label, accuracy = classify_image(image_array)
            image_tk = ImageTk.PhotoImage(image)
            image_label.config(image=image_tk)
            image_label.image = image_tk
            result_label.config(text=f"Classified as: {classified_label} Confidence: {accuracy:.2f}%", font=("Arial", 18))  
            load_button.pack_forget() 

load_button = tk.Button(root, text="Load Image", command=load_image, font=("Arial", 12), height=1, width=12)
load_button.pack(pady=10)

# Create a label to display the classification result
result_label = tk.Label(root, text="", wraplength=400)
result_label.pack(pady=10)

root.mainloop()
