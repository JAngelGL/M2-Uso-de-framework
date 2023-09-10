# Regresión Lineal con TensorFlow en Python

Este repositorio contiene un ejemplo de implementación de regresión lineal utilizando TensorFlow en Python. El objetivo de este proyecto es mostrar cómo usar TensorFlow para realizar una regresión lineal simple y visualizar los resultados.

## Descripción

La regresión lineal es una técnica de aprendizaje automático que se utiliza para modelar la relación entre una variable de respuesta (dependiente) y una o más variables predictoras (independientes) mediante una línea recta. En este caso, estamos realizando una regresión lineal simple para predecir la longitud del sépalo (`Largo de sépalo`) en función del ancho del pétalo (`Ancho de pétalo`) utilizando TensorFlow.

## Funcionamiento del Código

El código consta de los siguientes pasos:

1. **Lectura de Datos:** El código carga los datos del archivo 'iris.data' en un DataFrame de Pandas. Los datos incluyen mediciones de las longitudes de sépalo y pétalo de diferentes especies de iris.

2. **Preparación de los Datos:** Se seleccionan las columnas relevantes (`Largo de sépalo` y `Ancho de pétalo`) como variables de entrada (`x_vals`) y la variable objetivo (`Largo de sépalo`) como la variable de salida (`y_vals`).

3. **Construcción del Modelo:** Se define un modelo de regresión lineal que toma `x_data` (ancho de pétalo) como entrada y predice `y_target` (largo de sépalo) utilizando una variable de pendiente (`A`) y un término de sesgo (`b`).

4. **Entrenamiento del Modelo:** Se utiliza TensorFlow para entrenar el modelo. Se minimiza la pérdida (error) entre las predicciones del modelo y los valores reales (`y_target`) utilizando el optimizador de descenso de gradiente.

5. **Visualización de Resultados:** Después del entrenamiento, el código imprime los valores finales de la pendiente (`A`) y el término de sesgo (`b`). Luego, se calcula y visualiza la línea de regresión lineal junto con los datos originales en un gráfico. También se muestra un gráfico de la pérdida a lo largo de las iteraciones de entrenamiento.

## Requisitos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas:

- `matplotlib` para visualizar los datos y los resultados.
- `numpy` para manejar matrices y cálculos numéricos.
- `tensorflow` para construir y entrenar el modelo de regresión lineal.
- `pandas` para cargar y manipular los datos.
- `seaborn` para crear una matriz de correlación visual.
