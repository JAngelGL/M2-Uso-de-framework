# Regresión Lineal con TensorFlow en Python

Este repositorio contiene un ejemplo de implementación de regresión lineal utilizando TensorFlow en Python. El objetivo de este proyecto es mostrar cómo usar TensorFlow para realizar una regresión lineal simple y visualizar los resultados.

## Descripción

La regresión lineal es una técnica de aprendizaje automático que se utiliza para modelar la relación entre una variable de respuesta (dependiente) y una o más variables predictoras (independientes) mediante una línea recta. En este caso, estamos realizando una regresión lineal simple para predecir la longitud del sépalo (`Largo de sépalo`) en función del ancho del pétalo (`Ancho de pétalo`) utilizando TensorFlow.


## Funcionamiento del Código

El código `regresion_lineal.py` realiza las siguientes tareas:

1. **Carga de Datos**: Lee el conjunto de datos de ejemplo (`iris.data`) que contiene las mediciones de las flores de iris, incluyendo el largo del sépalo y el ancho del pétalo.

2. **Preprocesamiento de Datos**: Normaliza los datos utilizando Min-Max scaling para que estén en el rango [0, 1]. Luego, divide el conjunto de datos en tres partes: entrenamiento (60%), prueba (20%) y validación (20%).

3. **Definición del Modelo**: Utiliza TensorFlow para definir un modelo de regresión lineal que predice la longitud del sépalo en función del ancho del pétalo. El modelo se entrena para encontrar los valores óptimos de la pendiente (A) y el sesgo (b) que minimizan la pérdida (Loss).

4. **Entrenamiento del Modelo**: El modelo se entrena utilizando el optimizador Gradient Descent. Se ejecuta un bucle de entrenamiento durante un número especificado de iteraciones. En cada iteración, se selecciona un lote (batch) aleatorio del conjunto de entrenamiento y se actualizan los parámetros del modelo para reducir la pérdida.

5. **Evaluación del Modelo**: Después del entrenamiento, se evalúa el modelo en los conjuntos de prueba y validación. Se calcula el Error Cuadrático Medio (MSE) en el conjunto de prueba, la varianza del modelo y se muestran métricas relevantes.

6. **Visualización de Resultados**: Se genera un gráfico que muestra los datos originales y la mejor línea de ajuste del modelo en el conjunto de datos normalizado.

El código `regresion_lineal+l2.py` realiza las mismas tareas con la diferecia que se implementó **L2** el cual es un metodo de regularización que es utilizada para evitar el sobreajuste en modelos de aprendizaje profundo:

El código `funcion_regresion_lineal+l2+r2.py` agrega la metrica de R2, dicha es una métrica estadística que se utiliza para evaluar la calidad de un modelo de regresión, ademas de implementar todo en una funcion que pemite simular mas ecenarios variando algunos hiperparametros asi como el tamaños de **Train**, **Test** y **Val**


El código incluye comentarios para ayudar a comprender cada parte del proceso.
## Requisitos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas:

- `matplotlib` para visualizar los datos y los resultados.
- `numpy` para manejar matrices y cálculos numéricos.
- `tensorflow` para construir y entrenar el modelo de regresión lineal.
- `pandas` para cargar y manipular los datos.
