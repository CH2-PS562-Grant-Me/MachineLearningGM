from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import mysql.connector
import os
from dotenv import load_dotenv, dotenv_values

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
            "Beasiswa Bakti BCA": 0.3803366748,
            "Beasiswa BAZNAS": 0.2722606404,
            "Beasiswa Unggulan": 0.3362747212,
            "Bank Indonesia": 0.3601408819,
            "BRILiaN Scholarship": 0.1798165235,
            "BSI Scholarship": 0.2536307216,
            "APERTI-BUMN": 0.1587339438,
            "CIMB Niaga": 0.2494998172,
            "Beasiswa DataPrint": 0.360637847,
            "Karya Salemba Empat": 0.1828748499,
            "Pertamina Sobat Bumi": 0.2756835739,
            "Tanoto Foundation": 0.1592490954,
            "XL Future Leader": 0.3227687009
        }

        # Calculate the difference between predicted value and each scholarship average
        differences = {scholarship: abs(average - prediction[0][0]) for scholarship, average in
                       scholarship_averages.items()}

        # Sort scholarships by the closest difference and get the top 3
        closest_scholarships = sorted(differences, key=differences.get)[:3]

        # Call get_scholarship_records to fetch the records from the database
        scholarship_records = get_scholarship_records(closest_scholarships)

        # Return the records and the predicted value
        return jsonify({'Closest Scholarships': closest_scholarships,
                        'Predicted Score': prediction[0][0].tolist(),
                        'Scholarship Records': scholarship_records})

    except Exception as e:
        return jsonify({'error': str(e)})

def get_scholarship_records(closest_scholarships):
    db_connection = get_db()
    cursor = db_connection.cursor()

    placeholders = ', '.join(['%s'] * len(closest_scholarships))
    query = f"SELECT * FROM Scholarships WHERE nama IN ({placeholders})"

    cursor.execute(query, tuple(closest_scholarships))
    results = cursor.fetchall()

    cursor.close()
    db_connection.close()

    return results

def get_db():
    try:
        connection = mysql.connector.connect(
            user=os.getenv('MYAPP_DB_USER'),
            password=os.getenv('MYAPP_DB_PASS'),
            host=os.getenv('MYAPP_DB_HOST'),
            database=os.getenv('MYAPP_DB_NAME')
        )
        return connection
    except mysql.connector.Error as err:
        print("Error connecting to database:", err)
        return None


if __name__ == '__main__':
    app.run(debug=True)