import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Lee los datos
df = pd.read_csv("mnist_784.csv")
x = np.asanyarray(df.drop(columns=['class']))
y = np.asanyarray(df['class'])

# Divide los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Escala los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define el modelo de red neuronal con capas más grandes
model = models.Sequential([
    layers.Input(shape=x_train.shape[1]),
    layers.Dense(128, activation='relu'),  # Aumenta el tamaño de la capa oculta
    layers.Dense(64, activation='relu'),   # Aumenta el tamaño de la capa oculta
    layers.Dense(10, activation='softmax')  # Capa de salida para clasificación
])

# Compila el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrena el modelo en GPU
with tf.device('/GPU:1'):  # Esto asignará el entrenamiento a la GPU 0, la gpu 0 es la gpu integrada en el procesador profe 
    history = model.fit(x_train, y_train, batch_size=100, epochs=500, validation_data=(x_test, y_test))

# Evalúa el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Realiza predicciones en el conjunto de prueba
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calcula la matriz de confusión
confusion_mtx = confusion_matrix(y_test, y_pred_classes)
print('Matriz de Confusión:')
print(confusion_mtx)

# Genera el informe de clasificación
classification_rep = classification_report(y_test, y_pred_classes)
print('Informe de Clasificación:')
print(classification_rep)

# Guarda el modelo en formato .sav
joblib.dump(model, 'modelo_num_5.sav')
