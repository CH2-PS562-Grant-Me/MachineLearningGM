import pandas as pd
import tensorflow as tf
from sklearn.model_selection import  train_test_split

def model_topsis():
    df = pd.read_csv('dataset_v2.1.csv', sep=',')
    X = df.drop(['Nilai Bobot', 'Beasiswa', 'Rank'], axis=1).to_numpy()
    y = df['Nilai Bobot'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=[len(X_train[0])]),
        tf.keras.layers.Dense(1)
    ])

    customCallback = myCallback()

    model.compile(optimizer=tf.optimizers.Adam(), loss=tf.keras.losses.Huber(), metrics=['mae'])
    model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), callbacks=customCallback)

    print(model.predict([[3.83, 0., 0., 2., 3., 0., 0., 3., 14.]]))

    return model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_mae') <= 0.01):
            print("\nValidation MAE is less than 0.01, stopping...")
            self.model.stop_training = True

if __name__ == '__main__':
    model = model_topsis()
    model.save("model_topsis_v2.1.h5")