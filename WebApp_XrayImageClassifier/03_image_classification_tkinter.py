import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the saved model
last_model = tf.keras.models.load_model('model.h5')

def classify_image(image_path, model):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = classify_image(file_path, last_model)
        if prediction[0][0] >= 0.5:
            label.config(text="Prediction: COVID-19")
        else:
            label.config(text="Prediction: Normal")
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

# Create Tkinter GUI
root = tk.Tk()
root.title("Chest X-ray Image Classifier")

# Browse button
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Panel to display image
panel = tk.Label(root)
panel.pack(pady=10)

# Label to display prediction
label = tk.Label(root, text="")
label.pack(pady=10)

root.mainloop()
