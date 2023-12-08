from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load Model
path = 'model_topsis.h5'
model = tf.keras.models.load_model(path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari JSON
        data = request.get_json(force=True)
        input_features = [data['IPK'], data['Sertifikasi'], data['Sertifikasi Professional'], data['Prestasi Nasional'],
                          data['Kompetisi Top 3 Nasional'], data['Prestasi Internasional'],
                          data['Kompetisi Top 3 Internasional'], data['Intern'], data['Volunteer']]

        # Ubah jadi Numpy array
        input_array = np.array([input_features])

        # Buat prediksi
        prediction = model.predict(input_array)

        # Kasih hasil prediksi
        return jsonify({'Nilai Bobot': prediction[0][0].tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)