# %% [markdown]
# # Modelos de Clasificación

# %% [markdown]
# # Librerias

# %%
# Librerías para operaciones de datos y cálculos numéricos
import numpy as np
import pandas as pd
import math

# sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
## clasificaciones

# Para clasificación
from sklearn.ensemble import IsolationForest,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

## herramientas sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score,RandomizedSearchCV,cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score

# Importación de librerías
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score,auc,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

#from xgboost import XGBClassifier


# estadisticas
from scipy.stats import shapiro
from scipy.stats import ks_2samp
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

# Para guardar modelos
import os
import joblib

# Para la métrica personalizada
from collections import Counter

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# %% [markdown]
# # Carga de Datos

# %%
## Montamos el DRIVE para tener acceso a los datos
#drive.mount('/content/drive')
dataset_clasificacion=pd.read_excel('datos/df_capstone.xlsx')

# %%
#columns_drop=['Alim CuI','Sol_Cit','Fecha','Turno']
columns_drop=['Fecha','Turno']
dataset_clasificacion.drop(columns=columns_drop,inplace=True)
print(dataset_clasificacion.info())


# %% [markdown]
# ## Preparación de los datos

# %%
df_clasificacion=dataset_clasificacion.copy()
variables_predictoras=['Alim_CuT', 'Alim_CuS','Alim CuI' ,'Ag', 'Pb', 'Fe', 'P80_Alim_Ro300', 'pH_Ro300','Sol_Cit',
       'Tratamiento_Turno', 'Aire_Celdas', 'Nivel_Celdas']
df_clasificacion['cumplimiento']=np.where(df_clasificacion['Recuperacion_Turno']>=86,1,0)
df_clasificacion.info()

# %%
# Características de entrenamiento para clasificación
X_clas = df_clasificacion[variables_predictoras]
y_clas = df_clasificacion['cumplimiento']

# División de datos en entrenamiento y prueba
X_train_clas, X_test_clas, y_train_clas, y_test_clas = train_test_split(
    X_clas, y_clas, test_size=0.2, random_state=42, stratify=y_clas
)

# Mostrar distribución de clases
print("Distribución de clases en el conjunto completo:", Counter(y_clas))
print("Distribución de clases en el entrenamiento:", Counter(y_train_clas))
print("Distribución de clases en el test:", Counter(y_test_clas))

# Configuración de los pipelines con balanceo
pipelines = {
    'Logistic Regression': Pipeline([
        ('sampling', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('sampling', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('sampling', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    'AdaBoost': Pipeline([
        ('sampling', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', AdaBoostClassifier(random_state=42))
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('sampling', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]),
    'Decision Tree': Pipeline([
        ('sampling', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42))
    ]),
}

# Diccionario de parámetros para cada modelo
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__solver': ['liblinear', 'saga']
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, 30],
        'classifier__criterion': ['gini', 'entropy']
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    },
    'AdaBoost': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 1]
    },
    'K-Nearest Neighbors': {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance']
    },
    'Decision Tree': {
        'classifier__max_depth': [10, 20, 30],
        'classifier__criterion': ['gini', 'entropy']
    },
}

# Configuración de validación cruzada
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Diccionario para almacenar los mejores modelos
best_models = {}

# Proceso de GridSearchCV para cada modelo
for name, pipeline in pipelines.items():
    print(f"Buscando mejores hiperparámetros para {name}...")
    search = GridSearchCV(
        pipeline, param_grids[name], 
        cv=cv_strategy, scoring='f1_macro', n_jobs=-1
    )
    search.fit(X_train_clas, y_train_clas)
    
    # Guardar el mejor modelo
    best_models[name] = {
        'best_estimator': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_
    }
    
    print(f"Mejores hiperparámetros para {name}: {search.best_params_}")
    print(f"Mejor F1-Score (macro): {search.best_score_:.4f}\n")

# Evaluación de los mejores modelos en el conjunto de prueba
print("Evaluación de los mejores modelos en el conjunto de prueba:")
for name, data in best_models.items():
    print(f"\nModelo: {name}")
    best_model = data['best_estimator']
    y_pred = best_model.predict(X_test_clas)
    y_proba = best_model.predict_proba(X_test_clas)[:, 1]
    
    print("Reporte de clasificación:")
    print(classification_report(y_test_clas, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test_clas, y_proba):.4f}")


# %% Crear una lista para almacenar las métricas de cada modelo
metrics = []

# Evaluación de los mejores modelos en el conjunto de prueba
print("Evaluación de los mejores modelos en el conjunto de prueba:")
for name, data in best_models.items():
    print(f"\nModelo: {name}")
    best_model = data['best_estimator']
    
    # Predicciones
    y_pred = best_model.predict(X_test_clas)
    y_proba = best_model.predict_proba(X_test_clas)[:, 1]
    
    # Cálculo de métricas
    roc_auc = roc_auc_score(y_test_clas, y_proba)
    f1_macro = f1_score(y_test_clas, y_pred, average='macro')
    report = classification_report(y_test_clas, y_pred, output_dict=True)
    
    # Guardar métricas en una lista
    metrics.append({
        'Modelo': name,
        'F1-Score Macro': f1_macro,
        'ROC-AUC': roc_auc,
        'Precision Clase 0': report['0']['precision'],
        'Recall Clase 0': report['0']['recall'],
        'Precision Clase 1': report['1']['precision'],
        'Recall Clase 1': report['1']['recall']
    })

# Crear un DataFrame con las métricas
metrics_df = pd.DataFrame(metrics)

# Ordenar los modelos por ROC-AUC o F1-Score
metrics_df = metrics_df.sort_values(by='F1-Score Macro', ascending=False)

# Mostrar las métricas en una tabla
print("\nResumen de métricas de los modelos:")

# # Guardar las métricas en un archivo CSV si lo deseas
metrics_df.to_csv("resumen_metricas_modelos.csv", index=False)
print(metrics_df)


# Crear listas para almacenar los valores de accuracy
train_accuracy = []
test_accuracy = []
model_names = []

# Calcular la precisión en train y test para cada modelo
for name, data in best_models.items():
    best_model = data['best_estimator']

    y_train_pred = best_model.predict(X_train_clas)
    y_test_pred = best_model.predict(X_test_clas)

    train_acc = accuracy_score(y_train_clas, y_train_pred)
    test_acc = accuracy_score(y_test_clas, y_test_pred)

    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    model_names.append(name)



# %%
print("Claves disponibles en best_models:", best_models.keys())

# %% Crear la carpeta "Archivos_PKL_Clasificacion" si no existe
carpeta_modelos = "Archivos_PKL_Clasificacion"
if not os.path.exists(carpeta_modelos):
    os.makedirs(carpeta_modelos)

# Guardar los modelos usando las claves correctas
joblib.dump(best_models['Random Forest']['best_estimator'], os.path.join(carpeta_modelos, 'RandomForest_model.pkl'))
joblib.dump(best_models['Gradient Boosting']['best_estimator'], os.path.join(carpeta_modelos, 'GradientBoosting_model.pkl'))
joblib.dump(best_models['AdaBoost']['best_estimator'], os.path.join(carpeta_modelos, 'AdaBoost_model.pkl'))
joblib.dump(best_models['Logistic Regression']['best_estimator'], os.path.join(carpeta_modelos, 'LogisticRegression_model.pkl'))
joblib.dump(best_models['K-Nearest Neighbors']['best_estimator'], os.path.join(carpeta_modelos, 'KNN_model.pkl'))
joblib.dump(best_models['Decision Tree']['best_estimator'], os.path.join(carpeta_modelos, 'DecisionTree_model.pkl'))

print("Modelos guardados correctamente en la carpeta 'Archivos_PKL_Clasificacion'")