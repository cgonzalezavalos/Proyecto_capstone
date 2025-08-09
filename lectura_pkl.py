#%%# Importar las bibliotecas necesarias
import joblib
import numpy as np
import pandas as pd

#%% Cargar el modelo desde archivo .pkl
with open('Archivos_PKL/RandomForestReg_GS.pkl', 'rb') as archivo_modelo:
    modelo = joblib.load(archivo_modelo)

Variables_predictoras = modelo.feature_names_in_
#print(Variables_predictoras)

#%%
# Lista de características en el orden esperado por el modelo
columnas = ['Alim_CuT','Alim_CuS','Alim CuI','Ag','Pb',
            'Fe','P80_Alim_Ro300','pH_Ro300','Tratamiento_Turno',
            'Sol_Cit','Aire_Celdas','Nivel_Celdas']

# Crear una lista para almacenar los valores ingresados por el usuario
valores_usuario = []

print("Por favor, ingrese los siguientes valores:")
print("(Nota: Ingrese los valores correspondientes a cada variable.)\n")

for columna in columnas:
    while True:
        try:
            valor = float(input(f"{columna}: "))
            valores_usuario.append(valor)
            break
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número válido.")

# Convertir los valores a un DataFrame con las columnas correctas
nueva_muestra = pd.DataFrame([valores_usuario], columns=columnas)


# Realizar la predicción con el modelo cargado
prediccion = modelo.predict(nueva_muestra)

# Mostrar el resultado al usuario

print(f"Predicción recuperación de cobre: {prediccion[0]:.4f}")
