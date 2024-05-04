from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = load_model("T:\\pycharm projects\\pythonProject\\flask_deployment\\potatoes.h5")

# Define class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


def preprocess_image(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array


@app.route('/')
def index():
    return render_template("index1.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['file']
        if image_file:
            # Save the image
            image_path = "temp.jpg"
            image_file.save(image_path)

            # Preprocess the image
            processed_image = preprocess_image(image_path)

            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class = class_names[np.argmax(prediction)]

            return jsonify({'class': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
