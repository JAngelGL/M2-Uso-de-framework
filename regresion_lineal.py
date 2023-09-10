import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import ops
ops.reset_default_graph()
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Establecer una semilla para replicabilidad
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

sess = tf.Session()

columns = ["Largo de sépalo", "Ancho de sépalo", "Largo de pétalo", "Ancho de pétalo", "Especies"]
df = pd.read_csv('iris.data', names=columns)

# Dividir los datos en características (x_vals) y etiquetas (y_vals)
x_vals = np.array(df["Ancho de pétalo"])
y_vals = np.array(df["Largo de sépalo"])

# Normalizar los datos
scaler = MinMaxScaler()
x_vals_normalized = scaler.fit_transform(x_vals.reshape(-1, 1))
y_vals_normalized = scaler.fit_transform(y_vals.reshape(-1, 1))

# Dividir los datos en conjuntos de entrenamiento, prueba y validación (60% entrenamiento, 20% prueba, 20% validación)
x_train, x_temp, y_train, y_temp = train_test_split(x_vals_normalized, y_vals_normalized, test_size=0.4, random_state=seed)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, random_state=seed)

x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Variables para la pendiente (A) y el intercepto (b)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Modelo de regresión lineal
model_output = tf.add(tf.matmul(x_data, A), b)

# Función de pérdida (MSE)
loss = tf.reduce_mean(tf.square(y_target - model_output))

# Declaramos el optimizador (Gradient Descent)
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(loss)

init = tf.initialize_all_variables()
sess.run(init)

# Entrenamiento del modelo
loss_vec = []
for i in range(100):
    rand_x = x_train[rand_index]
    rand_y = y_train[rand_index]
    
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    if (i + 1) % 25 == 0:
        print("Step #" + str(i + 1) + " A = " + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print("Loss = " + str(temp_loss))

# Evaluar el modelo en los conjuntos de prueba y validación
test_loss = sess.run(loss, feed_dict={x_data: x_test, y_target: y_test})
val_loss = sess.run(loss, feed_dict={x_data: x_val, y_target: y_val})

print("Pérdida en el conjunto de prueba:", test_loss)
print("Pérdida en el conjunto de validación:", val_loss)

# Calcular el Error Cuadrático Medio (MSE) en el conjunto de prueba
mse = sess.run(tf.reduce_mean(tf.square(model_output - y_target)), feed_dict={x_data: x_test, y_target: y_test})
print("Error Cuadrático Medio (MSE) en el conjunto de prueba:", mse)

[slope] = sess.run(A)
[y_intercept] = sess.run(b)

print("Valor de A (pendiente):", slope)
print("Valor de b (intercepto):", y_intercept)

# Calcular la mejor línea de ajuste
best_fit = []
for i in x_vals_normalized:
    best_fit.append(slope * i + y_intercept)

# Graficar los datos y la mejor línea de ajuste
plt.plot(x_vals_normalized, y_vals_normalized, 'o', label='Data Points')
plt.plot(x_vals_normalized, best_fit, 'r-', label='Mejor línea de ajuste', linewidth=3)
plt.legend(loc='upper left')
plt.title('Largo de sépalo vs Ancho de pétalo (Normalizado)')
plt.xlabel('Ancho de pétalo Normalizado')
plt.ylabel('Largo de sépalo Normalizado')
plt.show()

# Graficar la pérdida (loss) a lo largo de las iteraciones
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss por generación')
plt.xlabel('Generación')
plt.ylabel('L2 Loss')
plt.show()
