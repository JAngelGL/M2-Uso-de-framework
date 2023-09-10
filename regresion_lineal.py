import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import ops
ops.reset_default_graph()
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Establecer una semilla para replicabilidad
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

sess = tf.Session()

columns = ["Largo de sépalo", "Ancho de sépalo", "Largo de pétalo", "Ancho de pétalo", "Especies"]
df = pd.read_csv('iris.data', names=columns)

# Dividir los datos en conjuntos de entrenamiento, prueba y validación
x_vals = np.array(df["Ancho de pétalo"])
y_vals = np.array(df["Largo de sépalo"])
x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.2, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed)

# Tamaño de batch
batch_size = 25

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

loss_vec = []

# Entrenamiento del modelo
for i in range(100):
    rand_index = np.random.choice(len(x_train), size=batch_size)
    rand_x = np.transpose([x_train[rand_index]])
    rand_y = np.transpose([y_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    if (i + 1) % 25 == 0:
        print("Step #" + str(i + 1) + " A = " + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print("Loss = " + str(temp_loss))

# Evaluación del modelo en el conjunto de prueba
test_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_test]), y_target: np.transpose([y_test])})
print("Pérdida en el conjunto de prueba:", test_loss)

[slope] = sess.run(A)
[y_intercept] = sess.run(b)

print("Valor de A (pendiente):", slope)
print("Valor de b (intercepto):", y_intercept)

best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)

plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Mejor línea de ajuste', linewidth=3)
plt.legend(loc='upper left')
plt.title('Largo de sépalo vs Ancho de pétalo')
plt.xlabel('Ancho de pétalo')
plt.ylabel('Largo de sépalo')
plt.show()

plt.plot(loss_vec, 'k-')
plt.title('L2 Loss por generación')
plt.xlabel('Generación')
plt.ylabel('L2 Loss')
plt.show()


