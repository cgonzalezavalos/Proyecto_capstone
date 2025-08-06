# %% [markdown]
# # Modelos de Predicción

# %% [markdown]
# # Librerias

# %%
# operaciones datos y numéricas
import numpy as np
import pandas as pd
import math

# sklearn
## regresiones y clasificaciones
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
## herramientas sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# graficos
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# estadisticas
import statsmodels.api as sm

# Para guardar modelos
import joblib

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# %% [markdown]
# # Carga de Datos

# %%
#df_capstone=pd.read_excel('ArchivosExcel/dataset_regresion.xlsx')
df_capstone=pd.read_excel('datos/df_capstone.xlsx')
#df_capstone_sfs=pd.read_excel('/content/drive/MyDrive/Magister/capstone/datos/df_capstone_sfs.xlsx')
df_capstone.info()
df_capstone=df_capstone.drop(columns=['Fecha','Turno'])#,'Alim CuI'])
df_regresion = df_capstone.copy()
#df_regresion.info()

# %% [markdown]
# # Modelos de Predicción

# %% [markdown]
# ## preparación de los datos

# %% [markdown]
# ### datos para regresiones

# %%
X_reg = df_regresion.drop('Recuperacion_Turno', axis=1)
X_reg['cumple']=np.where(df_regresion['Recuperacion_Turno']>=86,1,0)
## Etiquetas / clases
y_reg = df_regresion['Recuperacion_Turno']
train_dataset_reg, test_dataset_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42,stratify=X_reg['cumple'])
## train test split
X_train_reg = train_dataset_reg.drop(columns=['cumple'], axis=1)
X_test_reg = test_dataset_reg.drop(columns=['cumple'], axis=1)

# %% [markdown]
# ### comparar distribución de recuperacion entre train y test

# %%
# comparar distribucion de recuperacion entre train y test
# sns.kdeplot(y_reg)
# sns.kdeplot(y_train_reg)
# sns.kdeplot(y_test_reg)
# plt.legend(['Recuperacion_Turno', 'Recuperacion_Turno_train', 'Recuperacion_Turno_test'])
# plt.show()

# %% [markdown]
# ## Experimento con distintos modelos supervisados de predicción

# %% [markdown]
# ### Experimento inicial sin Gridsearch

# %%
# Lista de modelos a evaluar
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Support Vector Regression': SVR(),
    'Extra Trees': ExtraTreesRegressor(),
    'XGBoost': XGBRegressor(),  # Asegúrate de importar XGBRegressor al inicio del archivo
}

# Diccionario para guardar los resultados
results = []

# Evaluación de cada modelo
for name, model in models.items():
    # Predicción cruzada para obtener predicciones y calcular métricas
    y_pred = cross_val_predict(model, X_train_reg, y_train_reg, cv=5)

    # Calcular métricas
    r2 = r2_score(y_train_reg, y_pred)
    mse = mean_squared_error(y_train_reg, y_pred)
    mae = mean_absolute_error(y_train_reg, y_pred)
    rmse = np.sqrt(mse)

    # Calcular el R^2 ajustado en el conjunto de entrenamiento
    n = X_train_reg.shape[0]  # Número de muestras
    p = X_train_reg.shape[1]  # Número de predictores
    if (n - p - 1) != 0:
        r2_adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        r2_adjusted = np.nan  # or set to None, or another value indicating undefined


    # Guardar resultados en una lista de diccionarios
    results.append({
        'Model': name,
        'R2': r2,
        'R2_Adj': r2_adjusted,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    })

# Convertir los resultados en DataFrame para mejor visualización
results_exp = pd.DataFrame(results)


# Mostrar los resultados ordenados por R2
#display(results_exp.query("R2_Adj>=0.2").sort_values(by='R2_Adj', ascending=False))
print(results_exp.sort_values(by='R2_Adj', ascending=False))

# %%
#from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
}

# Espacios de hiperparámetros para cada modelo
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 250, 500],
        'max_depth': [1, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'min_weight_fraction_leaf': [0.0, 0.05, 0.1],
        'bootstrap': [True, False]
    },
    'Extra Trees': {
        'n_estimators': [50, 100, 250, 500],
        'max_depth': [1, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'min_weight_fraction_leaf': [0.0, 0.05, 0.2, 0.4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 250, 500],
        'max_depth': [1, 10, 20, 30],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }
}
# Diccionario para guardar los mejores modelos y sus hiperparámetros
best_models = {}
print('*'*100)
print("Buscando mejores hiperparámetros...")
for name, model in models.items():
    print(f"mejores hiperparámetros para {name}...")

    # Usa GridSearchCV o RandomizedSearchCV con el espacio de hiperparámetros
    #se hace cross validation con 10 folds
    search = GridSearchCV(model, param_grids[name], cv=10, scoring='r2', n_jobs=-1)

    # Ajusta el modelo
    search.fit(X_train_reg, y_train_reg)

    # Guarda el mejor modelo y los mejores hiperparámetros
    best_models[name] = {
        'best_estimator': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_  # Cambia a positivo si es más fácil de interpretar
    }

    print(f"Mejores hiperparámetros para {name}: {search.best_params_}")
    print(f"Mejor puntaje (R2): {search.best_score_}\n")
    print('*'*100)


# %%
# Mostrar los mejores modelos y sus hiperparámetros
for name, details in best_models.items():
    print(f"{name}:")
    print(f"  Mejores hiperparámetros: {details['best_params']}")
    print(f"  Mejor puntaje (R2): {details['best_score']}\n")

print('*'*100)
print("fin busqueda de mejores hiperparametros")
print('*'*100)
print('*'*100)

# %%
modelo_RF_Reg=RandomForestRegressor(max_depth=10,min_samples_split=5, n_estimators=100, random_state=42).fit(X_train_reg, y_train_reg)
modelo_ET_REG=ExtraTreesRegressor(max_depth=10,n_estimators=100,random_state=42).fit(X_train_reg, y_train_reg)
modelo_XGB_Reg=XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1, random_state=42).fit(X_train_reg, y_train_reg)

modelo_RF_Reg_GS=best_models['Random Forest']['best_estimator']
modelo_ET_REG_GS=best_models['Extra Trees']['best_estimator']
modelo_XGB_Reg_GS=best_models['XGBoost']['best_estimator']

print('*'*100)
print("fin entrenamiento de modelos")
print('*'*100)


# %%
# Lista de modelos a evaluar
print("Evaluación de modelos de regresión:")

modelos=[modelo_RF_Reg,modelo_ET_REG,modelo_XGB_Reg,modelo_RF_Reg_GS,modelo_ET_REG_GS,modelo_XGB_Reg_GS]
modelos_nombres=['RandomForestRegressor','ExtraTreesRegressor','XGBRegressor','RandomForestRegressor_GS','ExtraTreesRegressor_GS','XGBRegressor_GS']
regresion=['Random Forest','Extra Trees','XGBoost','Random Forest GS','Extra Trees GS','XGBoost GS']

# %%

mae_score_test=[]
mape_score_test=[]
mae_score_train=[]
mape_score_train=[]
modelo_name=[]
mse_score_train=[]
mse_score_test=[]
rmse_score_train=[]
rmse_score_test=[]
r2 = []
r2_ajust = []
for modelo, nombre in zip(modelos, modelos_nombres):
    r2_value = modelo.score(X_train_reg, y_train_reg)
    r2ajust_value = 1 - (1 - r2_value)*(X_train_reg.shape[0]-1)/(X_train_reg.shape[0]-X_train_reg.shape[1]-1)
    mae_test = mean_absolute_error(y_test_reg, modelo.predict(X_test_reg))
    mape_test=mean_absolute_percentage_error(y_test_reg, modelo.predict(X_test_reg))*100
    mae_train = mean_absolute_error(y_train_reg, modelo.predict(X_train_reg))
    mape_train=mean_absolute_percentage_error(y_train_reg, modelo.predict(X_train_reg))*100
    mse_train = mean_squared_error(y_train_reg, modelo.predict(X_train_reg))
    mse_test = mean_squared_error(y_test_reg, modelo.predict(X_test_reg))
    rmse_train=np.sqrt(mse_train)
    rmse_test=np.sqrt(mse_test)

    mae_score_test.append(mae_test)
    mape_score_test.append(mape_test)
    mae_score_train.append(mae_train)
    mape_score_train.append(mape_train)
    modelo_name.append(nombre)
    r2_ajust.append(r2ajust_value)
    mse_score_train.append(mse_train)
    #rmse_score_train.append(rmse_train)
    #mse_score_test.append(mse_test)
    #rmse_score_test.append(rmse_test)
    r2.append(r2_value)
    rmse_score_train.append(rmse_train)
    mse_score_test.append(mse_test)
    rmse_score_test.append(rmse_test)


# DataFrame con las métricas de evaluación de los modelos de regresión
df_mse = pd.DataFrame({
    'modelo': modelo_name,
    'R2': r2,
    'R2 ajustado': r2_ajust,
    'mae_test': mae_score_test,
    'mape_test': mape_score_test,
    'mae_train': mae_score_train,
    'mape_train': mape_score_train,
    'mse_train': mse_score_train,
    'rmse_train': rmse_score_train,
    'mse_test': mse_score_test,
    'rmse_test': rmse_score_test
})

# If using Jupyter, you can use display(df_mse) for better formatting:
# from IPython.display import display
# display(df_mse)
# Mostrar el DataFrame de métricas
print(df_mse)

# %% [markdown]
# Guarda el modelo
import os
os.makedirs('Archivos_PKL', exist_ok=True)

# Guardar el modelo Random Forest entrenado para regresión
joblib.dump(modelo_RF_Reg, 'Archivos_PKL/RandomForestReg.pkl')
joblib.dump(modelo_RF_Reg_GS, 'Archivos_PKL/RandomForestReg_GS.pkl')

# Guardar el modelo Extra Trees entrenado para regresión
joblib.dump(modelo_ET_REG, 'Archivos_PKL/ExtraTreesReg.pkl')
joblib.dump(modelo_ET_REG_GS, 'Archivos_PKL/ExtraTreesReg_GS.pkl')

# Guardar el modelo XGBoost entrenado para regresión
joblib.dump(modelo_XGB_Reg, 'Archivos_PKL/XGBReg.pkl')
joblib.dump(modelo_XGB_Reg_GS, 'Archivos_PKL/XGBReg_GS.pkl')

print("Modelos guardados en Archivos_PKL")

# %% [markdown]
# ### Cargar modelos guardados