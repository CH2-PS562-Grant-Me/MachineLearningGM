from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load Model
path = 'model_topsis_v2.1.h5'
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
            "Djarum": 0.1933799712,
            "BCA": 0.3803366748,
            "Baznas": 0.2722606404,
            "Beasiswa Unggulan": 0.3362747212,
            "BI": 0.3601408819,
            "BRI": 0.1798165235,
            "BSI": 0.2536307216,
            "BUMN Aperti": 0.1587339438,
            "CIMB Niaga": 0.2494998172,
            "Dataprint": 0.360637847,
            "KSE": 0.1828748499,
            "Pertamina Sobat Bumi": 0.2756835739,
            "Tanoto Foundation": 0.1592490954,
            "XL Future Leader": 0.3227687009
        }

        # Calculate the difference between predicted value and each scholarship average
        differences = {scholarship: abs(average - prediction[0][0]) for scholarship, average in
                       scholarship_averages.items()}

        # Sort scholarships by the closest difference and get the top 3
        closest_scholarships = sorted(differences, key=differences.get)[:3]

        # Return the closest scholarship names and the predicted value
        return jsonify(
            {'Predicted Score': prediction[0][0].tolist(), 'Recommended Scholarships': closest_scholarships})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)