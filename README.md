# PR2-Katherine-Litzy
Proyecto Integrador Final No. 2 de la materia de Lenguaje de Programacion para Analitica. Encontrandose en este repositorio el contenido del proyecto como sus Notebooks de Google Colab y Data Sets utilizados.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTpCPc3J8GEjEzhYU-aOEevmoxVjTHbUakI03t-rNSW3HJ_W0G6ZpgoPSXTbgZiJcXI3fIP2Dq03DVO/pub?output=csv')
# Mostrar las primeras filas del conjunto de datos
df.head()

# Obtener información general sobre los datos
df.info()

df.nunique()

df.isnull().sum()

df = df.dropna(how='any',axis=0)

df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="Anxiety Status", hue="Current Year", data=df)
plt.title("Anxiety by study year")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="Depression Status", hue="Current Year", data=df)
plt.title("Depression by study year")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="Panic Attack Status", hue="Current Year", data=df)  # Corrected column name
plt.title("Panic Attack by study year")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.countplot(x='Current Year', hue='Gender', data=df)
plt.title("Students studying in a particular year")
plt.show()

# Visualizar la distribución de variables numéricas
sns.histplot(data=df, x="Age", kde=True)  # Ejemplo de gráfico de histograma para la variable 'age'
plt.show()

# Resumen estadístico del CGPA
cgpa_stats = df['CGPA'].describe()
print("Resumen estadístico del CGPA:")
cgpa_stats

# Distribución del CGPA
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='CGPA', bins=15, kde=True)
plt.title('Distribución del CGPA')
plt.xlabel('CGPA')
plt.ylabel('Frecuencia')
plt.show()

# Análisis de distribución de género
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribución de Género")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(y="Marital Status", x="Age", data=df)
plt.title("Distribution of Age by Marital Status")
plt.xlabel("Age")
plt.ylabel("Marital Status")
plt.show()

# Calculate the count of each job category
job_counts = df['Career'].value_counts()

# Create the pie chart
plt.figure(figsize=(30, 30))
plt.pie(job_counts, labels=job_counts.index, autopct='%1.1f%%')
plt.title("Distribution of Job Categories")
plt.show()

# Realizar análisis bivariados
sns.boxplot(data=df, x="Seek for specialist treatment", y="Age")  # Ejemplo de diagrama de caja para la variable 'age' en función de 'Seek for specialist treatment'
plt.show()

# Visualizar la distribución de variables numéricas
sns.histplot(data=df, x="Anxiety Status", kde=True)  # Ejemplo de gráfico de histograma para la variable 'age'
plt.show()

# Visualizar la distribución de variables numéricas
sns.histplot(data=df, x="Panic Attack Status", kde=True)  # Ejemplo de gráfico de histograma para la variable 'age'
plt.show()

# Visualizar la distribución de variables numéricas
sns.histplot(data=df, x="Depression Status", kde=True)  # Ejemplo de gráfico de histograma para la variable 'age'
plt.show()

# Cargar el conjunto de datos
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ2ENsqj3lv4pUPldYczlSyCr89BzO6uBjfIvGvri6WON_g8dEPROGhKW5yCQ8eOZrP3a2ZdsqbZg3/pub?output=csv')
# Mostrar las primeras filas del conjunto de datos
df.head()

# Obtener información general sobre los datos
df.info()

df.nunique()

df.isnull().sum()

import numpy as np

cgpa = df['Cgpa']
sleep_duration = df['How_much_time_are_you_sleeping_everyday']

correlation, p_value = stats.pearsonr(cgpa, sleep_duration)

alpha = 0.05
if p_value < alpha:
    print("Hay evidencia para rechazar la hipótesis nula.")
else:
    print("No hay suficiente evidencia para rechazar la hipótesis nula.")


import pandas as pd
import scipy.stats as stats

# Cargar el conjunto de datos
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ2ENsqj3lv4pUPldYczlSyCr89BzO6uBjfIvGvri6WON_g8dEPROGhKW5yCQ8eOZrP3a2ZdsqbZg3/pub?output=csv')

# Filtrar datos por género
male_stress = df[df['Gender'] == 'Male']['concentrate_on_your_work_when_you_are_stressed']
female_stress = df[df['Gender'] == 'Female']['concentrate_on_your_work_when_you_are_stressed']

# Realizar un análisis de frecuencia
male_stress_freq = male_stress.value_counts()
female_stress_freq = female_stress.value_counts()

print("Frecuencia de respuestas de concentración bajo estrés (Hombres):\n", male_stress_freq)
print("\nFrecuencia de respuestas de concentración bajo estrés (Mujeres):\n", female_stress_freq)

# Realizar análisis de correlación
correlation = df['How_much_time_are_you_sleeping_everyday'].astype(float).corr(df['Cgpa'])

alpha = 0.05
if correlation > alpha:
    print("Hay evidencia para rechazar la hipótesis nula.")
    print("Existe una relación significativa entre el tiempo de sueño diario y el rendimiento académico (Cgpa).")
else:
    print("No hay suficiente evidencia para rechazar la hipótesis nula.")
    print("No se encontró una relación significativa entre el tiempo de sueño diario y el rendimiento académico (Cgpa).")

import pandas as pd

# Cargar el conjunto de datos
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ2ENsqj3lv4pUPldYczlSyCr89BzO6uBjfIvGvri6WON_g8dEPROGhKW5yCQ8eOZrP3a2ZdsqbZg3/pub?output=csv')  # Reemplazar con la ruta de tu archivo CSV

# Obtener estadísticas descriptivas para columnas numéricas
numeric_columns = ['Age', 'Cgpa', 'How_much_time_are_you_sleeping_everyday']
numeric_stats = df[numeric_columns].describe()

print("Estadísticas descriptivas para columnas numéricas:")
numeric_stats

# Obtener conteo de valores únicos para columnas categóricas
categorical_columns = ['Gender', 'Department', 'Are_you_having_proper_sleep_everyday']
categorical_counts = df[categorical_columns].nunique()

print("\nConteo de valores únicos para columnas categóricas:")
categorical_counts

import pandas as pd

# Calcular correlaciones entre columnas numéricas
correlation_matrix = df[numeric_columns].corr()

print("\nMatriz de correlación entre columnas numéricas:")
correlation_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Age', bins=10, kde=True)
plt.title('Histograma de Edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

gender_year = df.groupby(['Gender', 'Current_year_you_are_studying_in']).size().unstack()
gender_year.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Gráfico de Barras Apiladas de Año de Estudio por Género')
plt.xlabel('Género')
plt.ylabel('Cantidad')
plt.xticks(rotation=0)
plt.legend(title='Año de Estudio')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ2ENsqj3lv4pUPldYczlSyCr89BzO6uBjfIvGvri6WON_g8dEPROGhKW5yCQ8eOZrP3a2ZdsqbZg3/pub?output=csv')

# Crear un scatterplot entre 'Cgpa' y 'Age'
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='Cgpa')
plt.title('Scatterplot entre Edad y CGPA')
plt.xlabel('Edad')
plt.ylabel('CGPA')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ2ENsqj3lv4pUPldYczlSyCr89BzO6uBjfIvGvri6WON_g8dEPROGhKW5yCQ8eOZrP3a2ZdsqbZg3/pub?output=csv')

# Crear un scatterplot entre 'How_much_time_are_you_sleeping_everyday' y 'Cgpa'
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='How_much_time_are_you_sleeping_everyday', y='Cgpa')
plt.title('Scatterplot entre Tiempo de Sueño Diario y CGPA')
plt.xlabel('Tiempo de Sueño Diario')
plt.ylabel('CGPA')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ2ENsqj3lv4pUPldYczlSyCr89BzO6uBjfIvGvri6WON_g8dEPROGhKW5yCQ8eOZrP3a2ZdsqbZg3/pub?output=csv')

# Crear un scatterplot con línea de regresión
plt.figure(figsize=(8, 6))
sns.regplot(data=df, x='How_much_time_are_you_sleeping_everyday', y='Cgpa')
plt.title('Scatterplot con Línea de Regresión entre Tiempo de Sueño Diario y CGPA')
plt.xlabel('Tiempo de Sueño Diario')
plt.ylabel('CGPA')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ2ENsqj3lv4pUPldYczlSyCr89BzO6uBjfIvGvri6WON_g8dEPROGhKW5yCQ8eOZrP3a2ZdsqbZg3/pub?output=csv')

# Seleccionar variables numéricas para el pairplot
numeric_columns = ['How_much_time_are_you_sleeping_everyday', 'What_is_your_screen_time', 'Age']

# Crear el pairplot
sns.pairplot(df[numeric_columns])
plt.suptitle('Pairplot de Variables Numéricas', y=1.02)
plt.show()

# MODELOS DE SOLUCION Y PREDICTIVO AL PROBLEMA

# Importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTpCPc3J8GEjEzhYU-aOEevmoxVjTHbUakI03t-rNSW3HJ_W0G6ZpgoPSXTbgZiJcXI3fIP2Dq03DVO/pub?output=csv'
df = pd.read_csv(url)

# Función para obtener las calificaciones de la encuesta
def obtener_calificaciones():
    calificaciones = {}
    for pregunta, mensaje in encuesta.items():
        calificacion = int(input(mensaje))
        calificaciones[pregunta] = calificacion
    return calificaciones

# Función para analizar las calificaciones y generar recomendaciones
def generar_recomendaciones(calificaciones):
    recomendaciones = []
    for pregunta, calificacion in calificaciones.items():
        if pregunta == "Depression Status":
            if calificacion >= 7:
                recomendaciones.append("Considera hablar con un consejero o profesional de salud mental.")
                recomendaciones.append("Prueba mantener una rutina diaria y asegúrate de descansar lo suficiente.")
            else:
                recomendaciones.append("Mantén una actitud positiva y busca actividades que te hagan feliz.")
                recomendaciones.append("Conéctate con amigos y seres queridos para obtener apoyo emocional.")
        elif pregunta == "Anxiety Status":
            if calificacion >= 8:
                recomendaciones.append("Prueba técnicas de relajación como la meditación o la respiración profunda.")
                recomendaciones.append("Limita la cafeína y otros estimulantes para reducir la ansiedad.")
            else:
                recomendaciones.append("Haz ejercicio regularmente para reducir el estrés y la ansiedad.")
                recomendaciones.append("Establece metas pequeñas y alcanzables para mantenerte motivado.")
        elif pregunta == "Panic Attack Status":
            if calificacion >= 6:
                recomendaciones.append("Aprende técnicas para manejar los ataques de pánico, como la técnica de respiración.")
                recomendaciones.append("Practica la relajación muscular progresiva para reducir la tensión.")
            else:
                recomendaciones.append("Practica la atención plena (mindfulness) para reducir la sensación de pánico.")
                recomendaciones.append("Identifica situaciones desencadenantes y desarrolla estrategias para afrontarlas.")
        elif pregunta == "Seek for specialist treatment":
            if calificacion <= 3:
                recomendaciones.append("Si sientes que necesitas ayuda, considera buscar tratamiento especializado.")
                recomendaciones.append("Explora recursos en línea sobre salud mental y terapias disponibles.")
            else:
                recomendaciones.append("Continúa cuidando tu salud mental y no dudes en buscar apoyo si lo necesitas.")
                recomendaciones.append("Habla con tu profesor o consejero académico sobre ajustes en tus responsabilidades académicas.")

    # Recomendación general basada en las calificaciones
    total_calificaciones = sum(calificaciones.values())
    if total_calificaciones <= 15:
        recomendaciones.append("Es importante que busques apoyo y cuides tu salud mental. Hablar con amigos, familiares o profesionales puede ayudarte.")
        recomendaciones.append("Prueba actividades creativas como el arte o la escritura para expresar tus emociones.")

    return recomendaciones

# Conversatorio en el chat
print("Bienvenido al chat de apoyo de salud mental para estudiantes universitarios.")
print("Por favor, cuéntanos cómo te sientes hoy y si hay algo en particular que te preocupe.")

conversacion = input("Usuario: ")

# Encuesta para calificar la salud mental
encuesta = {
    "Depression Status": "Por favor, califica tu nivel de depresión del 1 al 10, donde 1 es 'Muy bajo' y 10 es 'Muy alto': ",
    "Anxiety Status": "Por favor, califica tu nivel de ansiedad del 1 al 10, donde 1 es 'Muy bajo' y 10 es 'Muy alto': ",
    "Panic Attack Status": "Por favor, califica tu nivel de ataques de pánico del 1 al 10, donde 1 es 'Muy bajo' y 10 es 'Muy alto': ",
    "Seek for specialist treatment": "Por favor, califica si has buscado tratamiento especializado del 1 al 10, donde 1 es 'No he buscado tratamiento' y 10 es 'He buscado tratamiento especializado en varias ocasiones': "
}

# Obtener las calificaciones de la encuesta
calificaciones = obtener_calificaciones()

# Generar recomendaciones
recomendaciones = generar_recomendaciones(calificaciones)

# Mostrar recomendaciones en una tabla atractiva
print("\nRecomendaciones basadas en tus calificaciones:")
for i, recomendacion in enumerate(recomendaciones, start=1):
    print(f"{i}. {recomendacion}")

# Graficar cómo se siente el estudiante
valores = list(calificaciones.values())
categorias = list(calificaciones.keys())

plt.figure(figsize=(12, 6))

# Gráfico de pastel
plt.subplot(1, 2, 1)
plt.pie(valores, labels=categorias, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Calificación de Salud Mental del Estudiante (Pastel)')

# Gráfico de barras
plt.subplot(1, 2, 2)
plt.barh(categorias, valores, color='blue')
plt.xlabel('Calificación')
plt.ylabel('Categoría')
plt.title('Calificación de Salud Mental del Estudiante (Barras)')

plt.tight_layout()
plt.show()

# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Cargar el conjunto de datos
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSjNUGwHLsfWoxuetSvVDmTOpKK9WJJ1hRNSFXgYt8QEtZ0p6xViZd3Tn5FlsWPJjAla-BNrv7EzfE-/pub?output=csv'
df = pd.read_csv(url)

# Eliminar filas con valores faltantes
df = df.dropna()

# Codificar variables categóricas en valores numéricos
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Are_you_having_proper_sleep_everyday',
                       'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
                       'When_you_are_stressed_more']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Seleccionar características y variable objetivo
X = df[['Gender', 'Are_you_having_proper_sleep_everyday',
        'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
        'When_you_are_stressed_more']]
y = df['Age']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
report = classification_report(y_test, y_pred)
print("Informe de clasificación:\n", report)

import matplotlib.pyplot as plt

# Create a scatter plot of actual ages vs. predicted ages
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.title('Actual Age vs. Predicted Age')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.show()

# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSjNUGwHLsfWoxuetSvVDmTOpKK9WJJ1hRNSFXgYt8QEtZ0p6xViZd3Tn5FlsWPJjAla-BNrv7EzfE-/pub?output=csv'
df = pd.read_csv(url)

# Eliminar filas con valores faltantes
df = df.dropna()

# Codificar variables categóricas en valores numéricos
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Department', 'Are_you_having_proper_sleep_everyday',
                       'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
                       'When_you_are_stressed_more']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Seleccionar características y variable objetivo
X = df[['Gender', 'Department', 'Are_you_having_proper_sleep_everyday',
        'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
        'When_you_are_stressed_more']]
y = df['Age']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Crear un gráfico de dispersión de edades reales vs. edades predichas
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.title('Edad Real vs. Edad Predicha')
plt.xlabel('Edad Real')
plt.ylabel('Edad Predicha')
plt.show()

# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Cargar el conjunto de datos
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSjNUGwHLsfWoxuetSvVDmTOpKK9WJJ1hRNSFXgYt8QEtZ0p6xViZd3Tn5FlsWPJjAla-BNrv7EzfE-/pub?output=csv'
df = pd.read_csv(url)

# Eliminar filas con valores faltantes
df = df.dropna()

# Codificar variables categóricas en valores numéricos
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Are_you_having_proper_sleep_everyday',
                       'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
                       'When_you_are_stressed_more']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Seleccionar características y variable objetivo
X = df[['Gender', 'Are_you_having_proper_sleep_everyday',
        'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
        'When_you_are_stressed_more']]
y = df['Department']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
report = classification_report(y_test, y_pred)
print("Informe de clasificación:\n", report)

# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSjNUGwHLsfWoxuetSvVDmTOpKK9WJJ1hRNSFXgYt8QEtZ0p6xViZd3Tn5FlsWPJjAla-BNrv7EzfE-/pub?output=csv'
df = pd.read_csv(url)

# Eliminar filas con valores faltantes
df = df.dropna()

# Codificar variables categóricas en valores numéricos
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Are_you_having_proper_sleep_everyday',
                       'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
                       'When_you_are_stressed_more']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Seleccionar características y variable objetivo
X = df[['Gender', 'Are_you_having_proper_sleep_everyday',
        'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
        'When_you_are_stressed_more']]
y = df['Department']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Crear un DataFrame para comparar los resultados
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Contar las ocurrencias de cada combinación de valor actual y predicho
result_counts = results_df.groupby(['Actual', 'Predicted']).size().unstack(fill_value=0)

# Graficar los resultados
result_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Actual vs. Predicted Department')
plt.xlabel('Actual Department')
plt.ylabel('Count')
plt.legend(title='Predicted Department')
plt.show()

# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSjNUGwHLsfWoxuetSvVDmTOpKK9WJJ1hRNSFXgYt8QEtZ0p6xViZd3Tn5FlsWPJjAla-BNrv7EzfE-/pub?output=csv'
df = pd.read_csv(url)

# Eliminar filas con valores faltantes
df = df.dropna()

# Codificar variables categóricas en valores numéricos
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Are_you_having_proper_sleep_everyday',
                       'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
                       'When_you_are_stressed_more']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Seleccionar características y variable objetivo
X = df[['Gender', 'Are_you_having_proper_sleep_everyday',
        'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
        'When_you_are_stressed_more']]
y = df['Department']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Crear un DataFrame para comparar los resultados
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Count the occurrences of each actual department
actual_counts = results_df['Actual'].value_counts()

# Count the occurrences of each predicted department
predicted_counts = results_df['Predicted'].value_counts()

# Create a bar plot to visualize the actual and predicted department counts
plt.figure(figsize=(12, 6))
actual_counts.plot(kind='bar', color='blue', label='Actual', alpha=0.7)
predicted_counts.plot(kind='bar', color='orange', label='Predicted', alpha=0.7)
plt.title('Actual vs. Predicted Department Counts')
plt.xlabel('Department')
plt.ylabel('Count')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Importar las librerías necesarias 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSjNUGwHLsfWoxuetSvVDmTOpKK9WJJ1hRNSFXgYt8QEtZ0p6xViZd3Tn5FlsWPJjAla-BNrv7EzfE-/pub?output=csv'
df = pd.read_csv(url)

# Eliminar filas con valores faltantes
df = df.dropna()

# Codificar variables categóricas en valores numéricos
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Are_you_having_proper_sleep_everyday',
                       'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
                       'When_you_are_stressed_more', 'Department']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Seleccionar características y variable objetivo
X = df[['Age', 'Are_you_having_proper_sleep_everyday',
        'Are_you_getting_good_food_diet_everyday', 'What_is_your_screen_time',
        'When_you_are_stressed_more']]
y = df['Gender']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Crear un DataFrame para comparar los resultados
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Contar las ocurrencias de cada combinación de valor actual y predicho
result_counts = results_df.groupby(['Actual', 'Predicted']).size().unstack(fill_value=0)

# Graficar los resultados
result_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Actual vs. Predicted Gender (Female & Male)')
plt.xlabel('Actual Gender (Female & Male)')
plt.ylabel('Count')
plt.legend(title='Predicted Gender')
plt.show()


















