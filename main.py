from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import io
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Model
model = load_model("model_fruits.h5")
class_names = ["Anggur", "Apel", "Buah Naga", "Ceri", "Durian", "Jambu Biji", "Jeruk", "Kiwi", "Lemon", "Mangga", "Nanas", "Pir", "Pisang", "Semangka", "Stroberi"]
app = Flask(__name__)

def predict(img):
    # Mengubah gambar menjadi array numpy, normalisasikan, dan reshape
    img = np.asarray(img) / 255.0
    img = img.reshape(1, 150, 150, 3)
    pred = model.predict(img)
    result = class_names[np.argmax(pred)]
    return result

@app.route("/", methods=["GET", "POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "no file"})
    
    # Membaca gambar dari file
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    
    # Mengubah ukuran gambar menjadi 150x150 piksel
    img = img.resize((150, 150), Image.NEAREST)
    
    # Memastikan gambar menggunakan RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Membuat Prediksi
    pred_img = predict(img) 

    # Membuka Data Gizi_Buah menggunakan CSV
    df = pd.read_csv("Gizi_Buah.csv", sep=',')
    df['Buah'] = df['Buah'].str.strip()
    nutrient_info = df.loc[df['Buah'] == pred_img, [
                'Kalori', 'Lemak(g)', 'Karbohidrat(g)', 'Protein(g)', 'Ukuran']]

    # Mengubah informasi nutrisi ke dictionary
    nutrient_info_dict = nutrient_info.to_dict(orient='records')

    # Mengirimkan hasil prediksi dan informasi nutrisi
    return jsonify({"predicted_class": pred_img,
                "nutritional_info": nutrient_info_dict})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
