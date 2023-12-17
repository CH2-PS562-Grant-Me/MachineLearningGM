from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import mysql.connector
app = Flask(__name__)

# Load model
model = load_model('model_V2.h5')

# connect to database
db = mysql.connector.connect(
    host="34.101.114.119",
    user="root",
    passwd="9MpDdM).}ev||3D.",
    database="grant_me",
)

# Load data scholarships from database
def get_scholarships(jenis_beasiswa):
    cursor = db.cursor()
    sql = "SELECT * FROM Scholarships WHERE jenis_beasiswa = %s"
    cursor.execute(sql, (jenis_beasiswa,))
    row_headers = [x[0] for x in cursor.description]
    results = cursor.fetchall()
    json_data = []
    for result in results:
        json_data.append(dict(zip(row_headers, result)))

    return json_data

# Make prediction with model use API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data input dari request
        data = request.get_json(force=True)
        input_data = np.array([[
            data['IPK'],
            data['Sertifikasi'],
            data['SertifikasiProfesional'],
            data['prestasiNasional'],
            data['lombaNasional'],
            data['prestasiInternasional'],
            data['lombaInternasional'],
            data['internMagang'],
            data['Kepanitiaan']
        ]])
        prediction = model.predict(input_data)

        # Convert float32 values to Python floats
        prediction = prediction.astype(float)
        # Ambil nilai tertinggi dari hasil prediksi
        max_index = np.argmax(prediction)
        max_value = prediction[0, max_index].item()

    
        # Menentukan cluster berdasarkan max_value
        clusters = ["Pemerintah", "Swasta", "Organisasi", "Prestasi", "Bantuan"]
        cluster = clusters[max_index]
        hasil = get_scholarships(cluster)
        
        if prediction is not None:
            return jsonify({
                'statusCode': 200,
                'message': 'Success Predicting',
                'Persentase Akurasi': max_value,
                'Tag Beasiswa': cluster,
                'output': hasil
            }), 200
        else:
            return jsonify({
                'statusCode': 500,
                'message': 'Failed Predicting',
                'output': {}
            }), 500
        
        

    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    app.run(port=5000)