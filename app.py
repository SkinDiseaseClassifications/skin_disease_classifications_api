from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, DenseNet201, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import joblib
import os


# Flask app initialization
app = Flask(__name__)

# Constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
SAVED_MODEL_FILENAME = './models/Hybrid_CNN_SVM_ResNet50V2_DenseNet201_DenseNet121.sav'
CDC = "Centers for Disease Control and Prevention"

# Load models
def load_models():
    model_a = ResNet50V2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    model_b = DenseNet201(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    model_c = DenseNet121(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    model_a = Model(inputs=model_a.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(model_a.output))
    model_b = Model(inputs=model_b.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(model_b.output))
    model_c = Model(inputs=model_c.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(model_c.output))

    return model_a, model_b, model_c

# Load and preprocess image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image array
    return img_array

# Combine features from different models
def combine_features(*features):
    return np.concatenate(features, axis=1)

# Get prediction details
def get_prediction_details(prediction):
    classes_info = {
        0: {
            "class": "Cacar Air",
            "signs": ["Ruam kemerahan yang muncul pada kulit di dada, punggung, dan wajah",
                      "Demam", 
                      "Kelelahan", 
                      "Kehilangan selera makan", 
                      "Sakit kepala"],
            "source": CDC
        },
        1: {
            "class": "Cacar Sapi",
            "signs": ["Munculnya lesi makula yang terasa nyeri, kemudian menjadi eschar berwarna hitam dan keras dengan pembengkakan dan kemerahan di sekelilingnya",
                      "Demam", 
                      "Kelelahan", 
                      "Muntah", 
                      "Sakit tenggorokan"],
            "source": "DermNet"
        },
        2: {
            "class": "Flu Singapura (HFMD)",
            "signs": ["Luka lepuh di mulut", 
                      "Ruam pada tangan dan kaki",
                      "Demam", 
                      "Sakit tenggorokan"],
            "source": CDC
        },
        3: {
            "class": "Sehat",
            "signs": [""],
            "source": ""
        },
        4: {
            "class": "Campak",
            "signs": [
                      "Ruam kemerahan"
                      "Demam tinggi (bisa mencapai lebih dari 40°C)", 
                      "Batuk", 
                      "Hidung berair (coryza)", 
                      "Mata merah dan berair"],
            "source": CDC
        },
        5: {
            "class": "Cacar Monyet",
            "signs": ["Muncul ruam yang muncul di berbagai anggota tubuh seperti tangan, kaki, dada, wajah, mulut atau di dekat alat kelamin",
                      "Demam", 
                      "Panas dingin", 
                      "Pembengkakan kelenjar getah bening", 
                      "Kelelahan", 
                      "Nyeri otot dan sakit punggung", 
                      "Sakit kepala", 
                      "Gejala pernapasan (misalnya sakit tenggorokan, hidung tersumbat, atau batuk)"],
            "source": CDC
        }
    }
    return classes_info.get(prediction, {"class": "Unknown", "signs": [], "source": "N/A"})

# Predict single image
def predict_single_image(img_path, svm_model, feature_extractor_models):
    img_array = load_and_preprocess_image(img_path)
    model_a, model_b, model_c = feature_extractor_models

    feature_a = model_a.predict(img_array)
    feature_b = model_b.predict(img_array)
    feature_c = model_c.predict(img_array)

    combined_features = combine_features(feature_a, feature_b, feature_c)
    prediction = svm_model.predict(combined_features)
    prediction_details = get_prediction_details(prediction[0])

    return prediction_details

# Routes
@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Save the file to ./images directory
            file_path = os.path.join('./images', file.filename)
            file.save(file_path)

            # Load the SVM model
            svm_model = joblib.load(SAVED_MODEL_FILENAME)

            # Load the feature extractor models
            model_a, model_b, model_c = load_models()

            # Predict the class of the image
            prediction_details = predict_single_image(file_path, svm_model, (model_a, model_b, model_c))
            
            return jsonify(prediction_details)
        except Exception as e:
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
