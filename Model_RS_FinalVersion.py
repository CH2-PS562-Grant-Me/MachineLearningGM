# import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Read the data from the CSV file or use an existing dataframe
data = pd.read_csv('Dataset_Final.csv')

# Separate the columns using semicolons
data[['Column1', 'Volunteers', 'Internships', 'National_Honor', 'National_Top3', 'International_Honor',
      'International_Top3', 'Certification',
      'Professional_Certification', 'Scholarships', 'Scholarship_Type', 'Scholarship_Name', 'GPA']] = data['Column1;Volunteers;Internships;National_Honor;National_Top3;International_Honor;International_Top3;Certification;Professional_Certification;Scholarships;Scholarship_Type;Scholarship_Name;GPA'].str.split(';', expand=True)

# Drop the unnecessary columns
data = data.drop(columns=['Column1;Volunteers;Internships;National_Honor;National_Top3;International_Honor;International_Top3;Certification;Professional_Certification;Scholarships;Scholarship_Type;Scholarship_Name;GPA'])

# Replace empty string values with NaN
data.replace('', np.nan, inplace=True)

# Convert data types to float if needed
data[['Volunteers', 'Internships', 'National_Honor', 'National_Top3', 'International_Honor',
    'International_Top3', 'Certification', 'Professional_Certification',
    'GPA']] = data[['Volunteers', 'Internships', 'National_Honor', 'National_Top3', 'International_Honor',
    'International_Top3', 'Certification', 'Professional_Certification',
    'GPA']].astype(float)

# # Replace NaN values with mean or other replacement strategy
# data.fillna(data.mean(), inplace=True)

# Get the columns to be used for clustering
X = data[['GPA', 'Certification', 'Professional_Certification', 'National_Honor', 'National_Top3',
    'International_Honor', 'International_Top3', 'Internships',
    'Volunteers']]

# Standardize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a k-means model with the desired number of clusters
kmeans = KMeans(n_clusters=5, random_state=42)

# Perform clustering on the data
kmeans.fit(X_scaled)

# Add the clustering result column to the dataframe
data['Cluster'] = kmeans.labels_

# Display the clustering result
print(data['Cluster'])

# Import the required libraries for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data['Cluster'], test_size=0.2, random_state=42)

# Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Inputan baru untuk diprediksi
new_input = np.array([[3.8, 5, 3, 2, 1, 1, 1, 14, 13]])  # Ganti dengan inputan yang sesuai

# Lakukan standardisasi pada inputan baru
new_input_scaled = scaler.transform(new_input)

# Lakukan prediksi cluster
predicted_cluster = kmeans.predict(new_input_scaled)

# Tampilkan hasil prediksi
print("Inputan masuk ke cluster:", predicted_cluster)
if predicted_cluster == 0:
    print("Pemerintah")
elif predicted_cluster == 1:
    print("Swasta")
elif predicted_cluster == 2:
    print("Organisasi")
elif predicted_cluster == 3:
    print("Prestasi")
elif predicted_cluster == 4:
    print("Bantuan")
    
# Save the model
model.save('model_V2.h5')