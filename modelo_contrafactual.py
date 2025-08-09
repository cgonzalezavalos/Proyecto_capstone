# %%
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import math
from scipy.stats import ks_2samp
import joblib
import dice_ml
from dice_ml.utils import helpers # helper functions

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# %% [markdown]
# ## Cargar Modelos
knn_model = joblib.load('Archivos_PKL_Clasificacion/KNN_model.pkl')
modelo_Reg= joblib.load('Archivos_PKL/XGBReg_GS.pkl')

# %% # Carga el archivo Excel en un DataFrame
df = pd.read_excel('datos/df_capstone.xlsx')

# %%# Verifica las columnas del DataFrame
print("Columnas en el DataFrame:", df.columns)
df['cumplimiento'] = np.where(df['Recuperacion_Turno'] >= 86, 1, 0)
df=df.drop(columns=['Fecha','Turno'])#,'Alim CuI'])
Variables_predictoras = modelo_Reg.feature_names_in_
Variables_ajustables = ['P80_Alim_Ro300', 'Tratamiento_Turno', 'pH_Ro300','Aire_Celdas','Nivel_Celdas']
Variables_Objetivo = ['Recuperacion_Turno','cumplimiento']

# %% [markdown]
# ## separación y estandarización de datos
## Características de entrenamiento para clasificación
X_clas = df[Variables_predictoras]
X_reg = df[Variables_predictoras]
## Etiquetas / clases
y_clas = df['cumplimiento']
y_reg = df['Recuperacion_Turno']

scaler_data = StandardScaler()
X_clas = scaler_data.fit_transform(X_clas)
dataset = pd.DataFrame(X_clas, columns=Variables_predictoras)
print("Columnas en el dataset escalado:", dataset.columns)
print('*'*100)
print('*'*100)
print(dataset.info())

# Mover la columna 'cumplimiento' al primer lugar
dataset['cumplimiento'] = y_clas
columnas = ['cumplimiento'] + [col for col in dataset.columns if col != 'cumplimiento']
dataset = dataset[columnas]

# Dividir el dataset en entrenamiento y prueba
train_dataset, test_dataset = train_test_split(dataset,test_size=0.2,random_state=42, stratify=dataset['cumplimiento'])

#print(train_dataset.head())

# Verificar las proporciones de la columna 'cumplimiento'
#print("Proporciones en el conjunto de entrenamiento:")
#print(train_dataset['cumplimiento'].value_counts(normalize=True))
#print("\nProporciones en el conjunto de prueba:")
#print(test_dataset['cumplimiento'].value_counts(normalize=True))

# %% [markdown]
# ## Estadísticas de las variables predictoras

# %%
# Crear el DataFrame con medias y desviaciones estándar
statistics_df = pd.DataFrame(
    data={
        "mean": scaler_data.mean_,  # Medias de cada característica
        "std": scaler_data.scale_  # Desviaciones estándar de cada característica
    },
    index=Variables_predictoras  # Nombres de las variables predictoras como índice
)

# Mostrar el DataFrame con medias y desviaciones estándar
statistics_df.reset_index(inplace=True)
statistics_df.rename(columns={'index':'Variables'}, inplace=True)
statistics_df

# %% modelo contrafactual
# Importar las librerías necesarias
d = dice_ml.Data(dataframe=train_dataset, continuous_features=['Alim_CuT', 'Alim_CuS', 'Ag', 'Pb', 'Fe', 'P80_Alim_Ro300',
       'pH_Ro300', 'Tratamiento_Turno', 'Sol_Cit', 'Aire_Celdas','Nivel_Celdas'],
                 outcome_name='cumplimiento')

m = dice_ml.Model(model=knn_model, backend="sklearn") # se utiliza un modelo de clasificación para el modelo contrafactual
exp = dice_ml.Dice(d, m, method="random")

joblib.dump(exp, 'Archivo_PKL_Contrafactual/modelo_contrafactual.pkl')


# %%
# Verificar las variables en train_dataset
print("Columnas en train_dataset:", train_dataset.columns)
print("Número de columnas en train_dataset:", train_dataset.shape[1])

# %%
print("Valores nulos en Variables_ajustables:")
print(df[Variables_ajustables].isnull().sum())

# %%
# Revisar columnas esperadas por el escalador
print("Columnas esperadas por scaler_data:", scaler_data.feature_names_in_)

# Reordenar y seleccionar columnas para el escalador
ejemplo = df[scaler_data.feature_names_in_].iloc[:1]
print("Ejemplo reordenado para el escalador:")
print(ejemplo)

# Escalar el ejemplo
try:
    ejemplo_escalado = scaler_data.transform(ejemplo)
    print("Escalado exitoso:", ejemplo_escalado)
except Exception as e:
    print(f"Error durante el escalado: {e}")

# %%
import pandas as pd
import numpy as np

# Crear copia del DataFrame original
df_resultado = df.copy()
df_resultado['encontro_contrafactual'] = 'no'  # Nueva columna, por defecto 'no'

# Filtrar registros donde 'cumplimiento == 0'
registros = df.query('cumplimiento == 0')

# Iterar sobre cada registro
for idx, registro in registros.iterrows():
    try:
        # Preparar registro eliminando la columna objetivo
        registro_sin_objetivo = registro.drop(['cumplimiento']).to_frame().T
        registro_sin_objetivo = registro_sin_objetivo[scaler_data.feature_names_in_]

        # Escalar el registro
        registro_scaled = scaler_data.transform(registro_sin_objetivo)
        registro_scaled = pd.DataFrame(registro_scaled, columns=scaler_data.feature_names_in_)

        # Generar contrafactuales
        e = exp.generate_counterfactuals(
            query_instances=registro_scaled,
            total_CFs=50,
            desired_class=1,  # Clase objetivo deseada
            features_to_vary=Variables_ajustables,
            verbose=True
        )

        # Verificar si se encontraron contrafactuales
        if e.cf_examples_list[0].final_cfs_df.shape[0] > 0:
            # Obtener los contrafactuales desescalados
            c = e.cf_examples_list[0].final_cfs_df
            df_contrafactual = c[Variables_ajustables]
            df_contrafactual = (df_contrafactual *
                                statistics_df.set_index('Variables').loc[Variables_ajustables, 'std'].values +
                                statistics_df.set_index('Variables').loc[Variables_ajustables, 'mean'].values)

            # Crear DataFrame completo con todas las columnas esperadas
            df_contrafactual_completo = pd.DataFrame(
                np.tile(registro_sin_objetivo.values, (df_contrafactual.shape[0], 1)),
                columns=registro_sin_objetivo.columns
            )

            # Reemplazar las columnas ajustadas con valores contrafactuales
            for col in Variables_ajustables:
                df_contrafactual_completo[col] = df_contrafactual[col].values

            # Asegurar el orden de columnas esperado por el modelo
            df_contrafactual_completo = df_contrafactual_completo[scaler_data.feature_names_in_]

            # Predecir la recuperación para cada contrafactual
            df_contrafactual['recuperacion_predicha'] = modelo_Reg.predict(df_contrafactual_completo)

            # Seleccionar el contrafactual con la máxima recuperación predicha
            mejor_contrafactual = df_contrafactual.loc[df_contrafactual['recuperacion_predicha'].idxmax()]

            # Actualizar el DataFrame resultado con el mejor contrafactual
            df_resultado.loc[idx, Variables_ajustables] = mejor_contrafactual[Variables_ajustables].values
            df_resultado.loc[idx, 'encontro_contrafactual'] = 'si'
            df_resultado.loc[idx, 'recuperacion_predicha'] = mejor_contrafactual['recuperacion_predicha']
        else:
            print(f"No se encontraron contrafactuales para índice: {idx}")

    except Exception as e:
        print(f"Error al procesar índice {idx}: {e}")
        continue

# Mostrar el DataFrame resultante
print("Proceso completado. Resultados:")
print(df_resultado)


# %%
# Comparar la predicción del modelo antes y después
print("\nEvaluando el modelo Extra Trees:")

# Predecir con el DataFrame original
y_pred_antes = modelo_Reg.predict(df[Variables_predictoras])
df_resultado['prediccion_antes'] = y_pred_antes

# Predecir con el DataFrame ajustado
y_pred_despues = modelo_Reg.predict(df_resultado[Variables_predictoras])
df_resultado['prediccion_despues'] = y_pred_despues

# Calcular mejora absoluta
df_resultado['mejora_absoluta'] = df_resultado['prediccion_despues'] - df_resultado['prediccion_antes']

# Comparar resultados
mejora = (df_resultado['mejora_absoluta'] > 0).sum()
total = len(df_resultado['prediccion_antes'])
print(f"Registros con mejora en la predicción: {mejora} de {total}")
print(f"Promedio de mejora: {df_resultado['mejora_absoluta'].mean():.4f}")

# Mostrar el DataFrame resultante
print("Proceso completado. Resultados:")
print(df_resultado)


# %%
df_resultado.to_csv('Resultado_test_3.csv', index=False)
print(df_resultado.head())
print(df_resultado['encontro_contrafactual'].value_counts())

# %%
# registro_original[Variables_predictoras].update(df_contrafactual.iloc[df_fila_menor_costo.index.values[0]])
# print(f'el valor de recuperacion predicho con optimizacion es de {extra_trees_model.predict(registro_original[Variables_predictoras])}')
# print(f'el valor de la recuperacion real es de {registro_original['Recuperacion_Turno'].values}')

# # %% [markdown]
# # ## Simulación masiva (no hacer!!!)

# # %%
# df_imputed['recuperacion_contrafactual']=0
# df_imputed['ph_ro300_sugerido']=0
# df_imputed['P80_Alim_Ro300_sugerido']=0
# df_imputed['Tratamiento_turno_sugerido']=0
# df_imputed['Alim_CuT_sugerido']=0

# # %%
# df_imputed.info()

# # %%
# # Selecciona una muestra específica donde 'cumplimiento' es 0 y elimina la columna 'cumplimiento'
# # Itera sobre cada fila del DataFrame
# for i in range(df_imputed.shape[0]):
#     # Si la condición de cumplimiento es 0
#     variables_drop=['Fecha', 'RECUPERACION_PONDERADA', 'Diferencia_Porcentual', 'Recuperacion_Calculada','recuperacion_contrafactual','ph_ro300_sugerido','P80_Alim_Ro300_sugerido','Tratamiento_turno_sugerido','Alim_CuT_sugerido','Recuperacion_rl','Recuperacion_rf']
#     print(i)
#     if model_rl.predict(df_imputed.iloc[i:i+1].drop(columns=variables_drop, axis=1)).item() < 86:
#         # Selecciona la fila actual y elimina las columnas innecesarias
#         registro = df_imputed.drop(columns=variables_drop, axis=1).iloc[i:i+1]
#         #print(registro)
#         # Genera contrafactuales para esta fila
#         e = exp.generate_counterfactuals(
#             registro,
#             total_CFs=5,
#             desired_class="opposite",
#             permitted_range={
#                 'Alim CuT': [0.5, 0.9],
#                 'P80 Alim Ro300': [180, 210],
#                 'Tratamiento turno': [18000, 22000],
#                 'pH Ro300': [9, 11]
#             },verbose=False

#         )

#         # Calcula los promedios de las columnas relevantes en el contrafactual
#         c = e.cf_examples_list[0].final_cfs_df
#         promedio_ph_ro300 = c['pH Ro300'].mean()
#         promedio_Alim_CuT = c['Alim CuT'].mean()
#         promedio_P80_Alim_Ro300 = c['P80 Alim Ro300'].mean()
#         promedio_Tratamiento_turno = c['Tratamiento turno'].mean()

#         # Actualiza 'registro' con los valores promedio del contrafactual
#         registro_actualizado = registro.copy()
#         registro_actualizado.loc[:, 'pH Ro300'] = promedio_ph_ro300
#         registro_actualizado.loc[:, 'Alim CuT'] = promedio_Alim_CuT
#         registro_actualizado.loc[:, 'P80 Alim Ro300'] = promedio_P80_Alim_Ro300
#         registro_actualizado.loc[:, 'Tratamiento turno'] = promedio_Tratamiento_turno

#         # Predice la recuperación con el modelo y asigna el resultado a la fila actual de 'recuperacion_contrafactual'
#         df_imputed.loc[i, 'recuperacion_contrafactual'] = model_rl.predict(registro_actualizado).item()
#         df_imputed.loc[i,'Alim_CuT_sugerido']=promedio_Alim_CuT
#         df_imputed.loc[i,'ph_ro300_sugerido']=promedio_ph_ro300
#         df_imputed.loc[i,'P80_Alim_Ro300_sugerido']=promedio_P80_Alim_Ro300
#         df_imputed.loc[i,'Tratamiento_turno_sugerido']=promedio_Tratamiento_turno
#     else:
#         # Si 'cumplimiento' no es 0, asigna 'RECUPERACION_PONDERADA' a 'recuperacion_contrafactual'
#         df_imputed.loc[i, 'recuperacion_contrafactual'] = model_rl.predict(registro).item()
#         df_imputed.loc[i,'Alim_CuT_sugerido']=df_imputed.loc[i,'Alim CuT']
#         df_imputed.loc[i,'ph_ro300_sugerido']=df_imputed.loc[i,'pH Ro300']
#         df_imputed.loc[i,'P80_Alim_Ro300_sugerido']=df_imputed.loc[i,'P80 Alim Ro300']
#         df_imputed.loc[i,'Tratamiento_turno_sugerido']=df_imputed.loc[i,'Tratamiento turno']


# df_imputed.info()

# # %%
# df_imputed.info()

# # %%
# df_imputed.to_excel('Datos /df_imputed.xlsx', index=False)

# # %%
# df_imputed['recuperacion_contrafactual'].head()

# # %%
# sns.histplot(df_imputed['recuperacion_contrafactual'], kde=True)
# plt.xlim(60,100)

# # %%
# sns.histplot(df_imputed['RECUPERACION_PONDERADA'].head(171), kde=True)
# plt.xlim(60,100)


