# app.py

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained CNN model
model = load_model('mnist_cnn.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image file
        img = request.files['image']
        
        # Open the image using PIL and convert it to grayscale
        image = Image.open(img).convert('L')
        
        # Resize the image to 28x28 pixels (the input size of the MNIST CNN model)
        image = image.resize((28, 28))
        
        # Convert the image to a NumPy array and invert the colors
        image_array = np.asarray(image)
        image_array = 255 - image_array
        
        # Scale the image pixel values to the range [0, 1]
        image_array = image_array / 255.0
        
        # Reshape the image array to match the input shape of the CNN model (1, 28, 28, 1)
        image_array = image_array.reshape(1, 28, 28, 1)
        
        # Make a prediction using the CNN model
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        
        # Redirect to the index page with the predicted digit
        return render_template('index.html', predicted_digit=predicted_digit)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

