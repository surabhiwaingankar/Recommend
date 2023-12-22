from flask import Flask, request, jsonify, render_template,send_from_directory
import os
import base64
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


# Set up Flask app
app = Flask(__name__)


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

def image_to_base64(image_path):
    # Convert image to base64 encoding
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image


# Your Flask routes go here
@app.route('/predict', methods=['POST'])
def predict():
   print("i am here")
   if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

   file = request.files['file']

   if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image to a temporary location
   image_path = 'temp.jpg'
   file.save(image_path)

    # Preprocess the image
   features = feature_extraction(image_path,model)
   print(f"features are{features}")
    # Make predictions using the loaded ML model
   indices = recommend(features,feature_list)
   print(f"indices are{indices}")
    # Convert the predictions to a format suitable for JSON response
    # This depends on the structure of your model's output
   predictions = []
   for files in indices[0]:
      predictions.append(filenames[files])
      #print(f"pathname {send_from_directory('static', filenames[files])}")

    # Convert each image to base64 and create a list
   images_data = [image_to_base64(image_path) for image_path in predictions]
   
   result = {'prediction': images_data}
  
    # Clean up the temporary image file
    # Note: In a production scenario, handle file cleanup more robustly
   import os
   os.remove(image_path)

   return jsonify(result)

@app.route('/')
def upload_file():
    print("i am here")
    # Render the HTML form for file upload
    return render_template('uploadfile.html')


"""@app.route('/images/<filename>')
def get_image(filename):
    print("i am inside getImages route")
    return send_from_directory('static', filename)"""


if __name__ == "__main__":
  app.run()
