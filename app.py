from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

# Load model
model = load_model('model.h5')


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

        # Format hasil prediksi ke dalam JSON
        output = {'prediction': prediction.tolist(), 'max_value': max_value}

        # Menentukan cluster berdasarkan max_value
        clusters = ["Pemerintah", "Swasta", "Organisasi", "Prestasi", "Bantuan"]
        cluster = clusters[max_index]

        if prediction is not None:
            return jsonify({
                'statusCode': 200,
                'message': 'Success Predicting',
                'Persentase Akurasi': max_value,
                'Tag Beasiswa': cluster,
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
