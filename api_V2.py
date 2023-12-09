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

        scholarship_averages = {
            "Djarum": 0.1709037198,
            "BCA": 0.1265305557,
            "Baznas": 0.146413047,
            "Beasiswa Unggulan": 0.1777850383,
            "BI": 0.1351889349,
            "BRI": 0.07076699353,
            "BSI": 0.09300227797,
            "BUMN Aperi": 0.06632186284,
            "CIMB Niaga": 0.1568084538,
            "Dataprint": 0.1836811913,
            "KJMU": 0.0680858067,
            "KSE": 0.08208118504,
            "MPM Berbagi": 0.1461342038,
            "Paragon": 0.07562848414,
            "Pertamina Sobat Bumi": 0.1183803168,
            "Tanoto Foundation": 0.07585506821,
            "XL Future Leader": 0.1233061459
        }

        closest_scholarship = min(scholarship_averages, key=lambda x: abs(scholarship_averages[x] - prediction[0][0]))

        # Kasih hasil prediksi
        return jsonify({'Nilai Bobot': prediction[0][0].tolist(), 'Closest Scholarship': closest_scholarship})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)