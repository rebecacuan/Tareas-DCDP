import numpy as np
import pandas as pd
import datetime
import random
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rng = np.random.default_rng()

# Función para generar fechas aleatorias dentro de un rango
def generate_random_date(start_date, end_date):
    rng = np.random.default_rng()
    days_difference = ( datetime.datetime.strptime(end_date, '%d/%m/%Y').date()  - datetime.datetime.strptime(start_date, '%d/%m/%Y').date() ).days
    random_days = rng.integers(days_difference)
    return datetime.datetime.strptime(start_date, '%d/%m/%Y').date() + datetime.timedelta(days=int(random_days))

# Semilla para reproducibilidad
np.random.seed(42)

# Generar 10,000 casos simulados
n_cases = 10000

# Fechas de inicio y vencimiento de la cobertura desde 2022 a 2025
start_date_coverage = '1/1/2022'
end_date_coverage = '31/12/2025'

# Tipos de cobertura
coverage_types = ['Responsabilidad civil', 'Cobertura total', 'Cobertura de colisión', 'Cobertura amplia', 'Cobertura de robo']

# Modelos de coches y probabilidades de ocurrencia
car_models = ['Toyota Corolla', 'Honda Civic', 'Ford Focus', 'Chevrolet Cruze', 'Nissan Sentra',
              'Hyundai Elantra', 'Volkswagen Jetta', 'Kia Forte', 'Mazda 3', 'Subaru Impreza']

probabilities = [0.2, 0.15, 0.12, 0.1, 0.08, 0.1, 0.08, 0.07, 0.05, 0.05]

# Generar datos simulados
data = {
    'Número de póliza': ['P' + '0' * ( 5 - len(str(_) ) ) + str(_) for _ in range(1,n_cases+1)],
    'Fecha de inicio': [generate_random_date(start_date_coverage, end_date_coverage) for _ in range(n_cases)],
    'Tipo de cobertura': np.random.choice(coverage_types, n_cases),
    'Modelo del coche': np.random.choice(car_models, n_cases, p=probabilities),
    'Año del coche': np.random.choice(range(2010,2023), n_cases),
    'Valor asegurado': np.random.randint(10000, 50000, n_cases),
    'Deducible': np.random.choice([500, 600, 700], n_cases),
    'Estado del seguro': np.random.choice(["Al día", "Vencido"], n_cases, p=[0.75, 0.25]),
    'Gastos médicos': np.random.choice([0, 1], n_cases, p=[0.75, 0.25]),
    'Daños a terceros': np.random.choice([0, 1], n_cases, p=[0.75, 0.25])
}

data['Fecha de vencimiento'] = [generate_random_date(data['Fecha de inicio'][_].strftime('%d/%m/%Y'), end_date_coverage) for _ in range(n_cases)]

# Crear DataFrame
df = pd.DataFrame(data)


# Codificación de variables categóricas
df_encoded = pd.get_dummies(df.drop(['Número de póliza', 'Fecha de inicio', 'Fecha de vencimiento'], axis=1))

# División de datos en entrenamiento y prueba
X = df_encoded.drop('Gastos médicos', axis=1)
y = df_encoded['Gastos médicos']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.2f}")


# Interfaz de usuario
st.title('Predicción de Gastos Médicos')
st.write('Ingrese los detalles del seguro para predecir si habrá gastos médicos.')

# Formulario de entrada de datos
form = st.form(key='insurance_form')
coverage_type = form.selectbox('Tipo de cobertura', df['Tipo de cobertura'].unique())
car_model = form.selectbox('Modelo del coche', df['Modelo del coche'].unique())
car_year = form.number_input('Año del coche', min_value=2010, max_value=2022)
insured_value = form.number_input('Valor asegurado', min_value=10000, max_value=50000)
deductible = form.select_slider('Deducible', options=[500, 600, 700])
insurance_state = form.selectbox('Estado del seguro', ['Al día', 'Vencido'])
third_party_damage = form.select_slider('Daños a terceros', options=[0, 1])
submit_button = form.form_submit_button(label='Predecir')

if submit_button:
    # Preprocesamiento de datos de entrada
    user_data = {
        'Tipo de cobertura': coverage_type,
        'Modelo del coche': car_model,
        'Año del coche': car_year,
        'Valor asegurado': insured_value,
        'Deducible': deductible,
        'Estado del seguro':insurance_state,
        'Daños a terceros': third_party_damage
    }

    # Crear DataFrame
    df_user = pd.DataFrame(user_data, index=[1])
    # Codificación de variables categóricas
    df_encoded_user = pd.get_dummies(df_user)

    all_columns = list(X_train.columns)
    user_columns = list(df_encoded_user.columns)
    main_list = [item for item in all_columns if item not in user_columns]
    cols_missing = {}
    for i in main_list:
        cols_missing[i] = False
    col_missing = pd.DataFrame(cols_missing, index=[1])

    df_encoded_user_all = pd.concat([df_encoded_user, col_missing], axis=1)
    df_encoded_user_all =  df_encoded_user_all[all_columns]


    # Predicción
    y_pred = model.predict(df_encoded_user_all)
    if y_pred == [1]:
        result = True
    else:
        result = False
    st.write(f"\nEl valor predicho para los gastos medicos es el siguiente: {y_pred} ({result})")  
    

