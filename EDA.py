# %% [markdown]
# # Librerias
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
## sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge, Lasso,LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
## sklearn para outliers
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import r2_score, mean_squared_error
## sklear para codificacion
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# estadisticas
from scipy.stats import ks_2samp
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# graficos
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
import plotly.io as pio
pio.templates.default = "plotly_white"


# %% [markdown]
# # Carga datos

df_original_variables=pd.read_excel('datos/df_final_v4.xlsx')
df_original_variables.drop('Missing Values',inplace=True,axis=1)
df_otras_variables=pd.read_excel('datos/df_otras_variables.xlsx')
df_otras_variables.drop('Unnamed: 0',inplace=True,axis=1)
df_raw=pd.merge(df_original_variables,df_otras_variables,on=['Fecha','Turno'],how='left')
df_raw.info()


# %%
df_raw.rename(columns={'Rec Turno CuT':'Recuperacion_Turno'},inplace=True)
df_raw.head()

# %% [markdown]
# # Datos faltantes
# eliminar los registros donde la variable **recuperacion_turno** es 0 o nula
df_clean=df_raw.query('~Recuperacion_Turno.isnull()').copy()
df_clean.info()

# %% [markdown]
# creamos una variable que cuenta la cantidad de datos faltantes por registro (**missing values**)
df_clean['Missing Values']=df_clean.isna().sum(axis=1)
df_clean.info()

# %% [markdown]
# gráfico de Missing Values por regitro

tabla=df_clean.groupby('Missing Values').size()
sns.barplot(tabla)
plt.ylabel('Cantidad de registros')
plt.xlabel('Cantidad de valores faltantes')
plt.show()

# %%
tabla_MV_turno=df_clean.groupby(['Missing Values','Turno']).size()
tabla_MV_turno.name='Cantidad de registros'
tabla_MV_turno=tabla_MV_turno.reset_index()
figsize = (12, 1.2 * len(tabla_MV_turno['Turno'].unique()))
plt.figure(figsize=figsize)
sns.barplot(tabla_MV_turno, y='Cantidad de registros', x='Missing Values', hue='Turno')
plt.show()

# %% [markdown]
# reducir solo a registros que tienen 4 o menos variables faltantes

# %%
df_clean.query("`Missing Values` < 5",inplace=True)
df_clean.info()

# %%
# Calcular la cantidad de NaN y el total de entradas por columna, excepto 'Fecha'
variables_no_numericas=['Fecha','Turno']
numeric_cols = df_clean.columns.drop(variables_no_numericas)
nan_counts = df_clean[numeric_cols].isna().sum()
total_counts = len(df_clean)

# Calcular el porcentaje de datos no NaN
non_nan_percentage = (1 - nan_counts / total_counts) * 100
nan_percentage=(nan_counts/total_counts)*100

# Crear un DataFrame con los resultados
results_df = pd.DataFrame({
    'Var': nan_counts.index,
    'count_nan': nan_counts.values,
    'Total Registros': total_counts,
    'Porcentaje completos': non_nan_percentage.values,
    'Porcentaje faltantes':nan_percentage.values
})

# Ordenar los resultados por porcentaje en orden decreciente
results_df = results_df.sort_values(by='Porcentaje completos', ascending=False)
results_df = results_df.reset_index(drop=True)
results_df

# %%
# Crear un gráfico de barras con Plotly
fig = px.bar(results_df.query("Var!='Missing Values'"), x='Var', y='Porcentaje completos',
             title='Porcentaje de Datos Disponibles por Variable',
             labels={'Var': 'Variable', 'Porcentaje completos': 'Porcentaje de Datos No NaN (%)'},
             text='Porcentaje completos')

# Añadir estilo al gráfico
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(xaxis_tickangle=-45)

# Ajustar manualmente el tamaño del gráfico
fig.update_layout(width=1000, height=800)  # Cambia estos valores según tus necesidades

# Mostrar el gráfico
fig.show()

# %% [markdown]
# # Recuperación de cobre

# %%
promedio_recuperacion=df_original_variables['Rec Turno CuT'].mean()
mediana_recuperacion=df_original_variables['Rec Turno CuT'].median()
meta_recuperacion=86
sns.histplot(df_original_variables['Rec Turno CuT'], bins=20, kde=True)
#plt.vlines(promedio_recuperacion, 0, 120, colors='r', linestyles='dashed', label='Promedio recuperación')
#plt.vlines(mediana_recuperacion, 0, 120, colors='b', linestyles='dashed', label='Mediana recuperación')
#plt.vlines(meta_recuperacion, 0, 120, colors='g', linestyles='dashed', label='Meta recuperación')
#plt.text(promedio_recuperacion, 120, f'Prom:{promedio_recuperacion:.0f}', color='r', fontsize=8, ha='right')
#plt.text(mediana_recuperacion, 120, f'Med:{mediana_recuperacion:.0f}', color='b', fontsize=8, ha='left')
#plt.text(meta_recuperacion, 120, f'Meta:{meta_recuperacion}', color='g', fontsize=8, ha='left')
#plt.legend(loc='upper left')
plt.show()

# %%
df_clean.Recuperacion_Turno.describe()

# %%
df_clean.query("Recuperacion_Turno<100 and Recuperacion_Turno>0",inplace=True)
#df_clean.query("Recuperacion_Turno>0",inplace=True)
df_clean.info()

# %%
promedio_recuperacion=df_clean['Recuperacion_Turno'].mean()
mediana_recuperacion=df_clean['Recuperacion_Turno'].median()
meta_recuperacion=86
sns.histplot(df_clean['Recuperacion_Turno'], bins=20, kde=True)
#plt.vlines(promedio_recuperacion, 0, 120, colors='r', linestyles='dashed', label='Promedio recuperación')
#plt.vlines(mediana_recuperacion, 0, 120, colors='b', linestyles='dashed', label='Mediana recuperación')
plt.vlines(meta_recuperacion, 0, 120, colors='g', linestyles='dashed', label='Meta recuperación')
#plt.text(promedio_recuperacion, 120, f'Prom:{promedio_recuperacion:.0f}', color='r', fontsize=8, ha='right')
#plt.text(mediana_recuperacion, 120, f'Med:{mediana_recuperacion:.0f}', color='b', fontsize=8, ha='left')
plt.text(meta_recuperacion, 120, f'Meta:{meta_recuperacion}', color='g', fontsize=8, ha='left')
#plt.legend(loc='upper left')
plt.ylabel('Cantidad de turnos')
plt.xlabel('Recuperación Cu')
plt.show()

# %% [markdown]
# hacer tabla de recuperacion x dia

# %%
df_clean.info()

# %%
tabla_recuperacion_dia=df_clean.groupby(['Fecha','Turno']).agg({'Recuperacion_Turno':'sum','Tratamiento turno':'sum'}).reset_index()
tabla_recuperacion_dia['Mult']=tabla_recuperacion_dia['Recuperacion_Turno']*tabla_recuperacion_dia['Tratamiento turno']
tabla_recuperacion_dia=tabla_recuperacion_dia.groupby('Fecha').agg({'Mult':'sum','Tratamiento turno':'sum'}).reset_index()
tabla_recuperacion_dia['Recuperacion']=tabla_recuperacion_dia['Mult']/tabla_recuperacion_dia['Tratamiento turno']
tabla_recuperacion_dia=tabla_recuperacion_dia[['Fecha','Recuperacion']]
tabla_recuperacion_dia.dropna(inplace=True)
tabla_recuperacion_dia['indice']=tabla_recuperacion_dia.index
tabla_recuperacion_dia

# %%
plt.figure(figsize=(12, 6))
sns.lineplot(data=tabla_recuperacion_dia,x=tabla_recuperacion_dia.index, y='Recuperacion', marker='o', color='blue',estimator=None)
plt.title('Recuperación de cobre por fecha')
plt.ylabel('Recuperación de cobre')

minima = tabla_recuperacion_dia.index.min()
maxima = tabla_recuperacion_dia.index.max()

plt.hlines(meta_recuperacion, minima, maxima, colors='r', linestyles='dashed', label='Meta recuperación')
plt.text(maxima, meta_recuperacion, f'Meta:{meta_recuperacion}', color='r', fontsize=10, ha='left')

plt.hlines(promedio_recuperacion, minima, maxima, colors='g', linestyles='dashed', label='Promedio recuperación')
plt.text(maxima, promedio_recuperacion, f'Prom:{promedio_recuperacion:.0f}', color='g', fontsize=10, ha='left')

#plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))  # Aquí defines el intervalo de 15 días
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formato de fecha año-mes-día
plt.xlabel('')
#plt.xticks(rotation=90)
plt.xlim(minima, maxima)

plt.show()


# %%
df_clean['cumplimiento']=df_clean['Recuperacion_Turno']>=meta_recuperacion
tabla_cumplimiento = df_clean.groupby(['Turno']).agg({'cumplimiento': 'sum','Recuperacion_Turno':'count'})
tabla_cumplimiento = tabla_cumplimiento.rename(columns={'Recuperacion_Turno':'cantidad_turnos'})
tabla_cumplimiento['porcentaje_cumplimiento']=tabla_cumplimiento['cumplimiento']/tabla_cumplimiento['cantidad_turnos']
tabla_cumplimiento['porcentaje_no_cumplimiento']=1-tabla_cumplimiento['porcentaje_cumplimiento']
tabla_cumplimiento = tabla_cumplimiento.reset_index()
tabla_cumplimiento

# %%
df_clean['Cumple_Meta'] = df_clean['Recuperacion_Turno'] >= meta_recuperacion
tabla_cumplimiento = df_clean.groupby(['Turno', 'Cumple_Meta'])['Recuperacion_Turno'].count().reset_index()
tabla_cumplimiento = tabla_cumplimiento.rename(columns={'Recuperacion_Turno': 'Cantidad'})
tabla_cumplimiento = tabla_cumplimiento.pivot(index='Turno', columns='Cumple_Meta', values='Cantidad').fillna(0)

# Calcula el porcentaje para cada turno
tabla_cumplimiento_pct = tabla_cumplimiento.div(tabla_cumplimiento.sum(axis=1), axis=0) * 100

# Crear el gráfico de barras apiladas con porcentajes
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['red', 'green']  # Puedes cambiar los colores si lo deseas

# barras apiladas
bottom = np.zeros(len(tabla_cumplimiento_pct))
for i, col in enumerate(tabla_cumplimiento_pct.columns):
    ax.bar(tabla_cumplimiento_pct.index, tabla_cumplimiento_pct[col], bottom=bottom, label=col, color=colors[i])
    bottom += tabla_cumplimiento_pct[col]

ax.set_title('Cumplimiento de Recuperación por Turno (Porcentaje)')
ax.set_ylabel('Porcentaje de Registros')
ax.set_xlabel('Turno')
ax.legend(title='Cumple Meta', labels=['No', 'Sí'])

# Agregar etiquetas de porcentaje
for i, rect in enumerate(ax.patches):
    height = rect.get_height()
    width = rect.get_width()
    x = rect.get_x() + width / 2
    y = rect.get_y() + height / 2

    if height > 0:  # Solo agrega etiquetas si la altura es mayor a 0
        label_text = f'{height:.1f}%'
        ax.text(x, y, label_text, ha='center', va='center', color='white', fontsize=10)

plt.show()

# %% [markdown]
# # Visualización inicial de Variables

# %%
def graficos_variables(df, variables_excluidas=None):
    # Si no se especifican variables excluidas, usa las predeterminadas
    if variables_excluidas is None:
        variables_excluidas = ['Fecha', 'Recuperacion_Turno']

    # Seleccionar columnas numéricas excluyendo las variables indicadas
    numeric_columns = df.select_dtypes(include='number').drop(columns=variables_excluidas, errors='ignore')

    # Definir el número de filas y columnas en el layout de subplots
    num_cols = 4  # Ajusta este valor según el espacio que prefieras
    num_rows = math.ceil(numeric_columns.shape[1] / num_cols)

    # Crear el tamaño de la figura
    plt.figure(figsize=(20, num_rows * 5))

    # Graficar cada columna como un histograma en un subplot diferente
    for i, col in enumerate(numeric_columns.columns):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(x=numeric_columns[col], kde=True)
        plt.title(col)
        plt.ylabel('')  # Elimina la etiqueta del eje y para ahorrar espacio

    # Ajustar los espacios entre subplots
    plt.tight_layout()
    plt.show()

# %%
graficos_variables(df_raw.drop(columns=['Turno','Recuperacion_Turno']))

# %% [markdown]
# # Missing Values

# %%
df_clean['Missing Values'].value_counts()

# %%
df_clean.info()

# %%
date_cols = df_clean.select_dtypes(include=['datetime']).columns  # Identifica columnas de fecha
cat_columns=df_clean.select_dtypes(include=['object']).columns
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns  # Identifica columnas numéricas

# Solo aplica imputación en columnas numéricas
#imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=42)
imputer = IterativeImputer(estimator=RandomForestRegressor(),skip_complete=True, random_state=42)
df_numeric_imputed = imputer.fit_transform(df_clean[numeric_cols])
df_numeric_imputed = pd.DataFrame(df_numeric_imputed, columns=numeric_cols)

# Combina los resultados con las columnas de fecha
df_capstone = pd.concat([df_clean[date_cols], df_numeric_imputed, df_clean[cat_columns]], axis=1)

df_capstone.info()

# %% [markdown]
# # Outliers

# %% [markdown]
# ## estandarizar datos

# %%
# Crear un objeto StandardScaler
estandarizar = StandardScaler()

# Ajustar el scaler y transformar los datos
scaled_df = estandarizar.fit_transform(df_capstone.drop(columns=['Fecha','Turno','Missing Values']))

# transformar a dataframe arreglo de estandarizado
scaled_df = pd.DataFrame(scaled_df, columns=df_capstone.drop(columns=['Fecha','Turno','Missing Values']).columns)
scaled_df.dropna(inplace=True)

# %%
scaled_df.info()

# %% [markdown]
# ## función mahalanobis

# %%
def Mahalanobis(x, df, cov=None): #Argumentos opcionales
    x_mu = x - df.mean(axis=0)
    if not cov:
        cov = np.cov(df.values.T)
    inv_covmat = np.linalg.inv(cov) #Pseudo inversa
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()

# %% [markdown]
# ## cluster de registros

# %%
kmeans_model = KMeans(n_clusters=6,
                      random_state=2023,
                      verbose=0)
cluster_labels = kmeans_model.fit(scaled_df)
cluster_labels=kmeans_model.predict(scaled_df)

scaled_df['cluster'] = cluster_labels
centers_kmeans = scaled_df.groupby(['cluster']).aggregate('mean').reset_index()

# %% [markdown]
# ## Mahalanobis y LOF

# %%
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
pred = lof.fit_predict(scaled_df)
scaled_df["lof"] = lof.negative_outlier_factor_
scaled_df["Mahala"] = Mahalanobis(x = scaled_df, df = scaled_df)
scaled_df['cluster'] = cluster_labels
centers_kmeans = scaled_df.groupby(['cluster']).aggregate('mean').reset_index()
centers_kmeans

# %%
scaled_df.cluster.value_counts()

# %%
sns.scatterplot(data=scaled_df, x="Mahala", y="lof", c=scaled_df['cluster'].astype(int))
plt.ylim(-5,0)
plt.title("Mahalanobis vs LOF")
#plt.vlines(50, -5, 0, color='red')
#plt.hlines(-2, 0, 500, color='red')
plt.show()

# %%
scaled_df_so = scaled_df[(scaled_df['Mahala'] < 100)]# & (scaled_df['lof'] > -1.5)]
scaled_df_so.cluster.value_counts()

# %%
sns.scatterplot(data=scaled_df_so, x="Mahala", y="lof", c=scaled_df_so['cluster'].astype(int))
plt.ylim(-2,0)
plt.title("Mahalanobis vs LOF")
plt.show()

# %%
no_std_df_so=estandarizar.inverse_transform(scaled_df_so.drop(columns=['cluster', 'lof', 'Mahala']))
no_std_df_so = pd.DataFrame(no_std_df_so, columns=df_capstone.drop(columns=['Fecha','Turno','Missing Values']).columns)
no_std_df_so['Fecha'] = df_capstone['Fecha'].reset_index(drop=True)
no_std_df_so['Turno'] = df_capstone['Turno'].reset_index(drop=True)
df_capstone=no_std_df_so.copy()
df_capstone.info()

# %% [markdown]
# # Test de Kolmogorov-Smirnov
# 
# El test de Kolmogorov-Smirnov (KS) es una prueba que compara dos distribuciones de probabilidad para evaluar si provienen de la misma distribución o si una muestra sigue una distribución específica.
# Un valor alto del p-value respalda la hipótesis nula, es decir, que no hay evidencia para decir que las distribuciones son diferentes entre si.

# %%
common_columns = df_capstone.select_dtypes(include=[np.number]).columns.intersection(df_capstone.select_dtypes(include=[np.number]).columns)

# Realizar el test de Kolmogorov-Smirnov para cada columna numérica

results = []
for column in common_columns:
    if column in df_clean and column in df_capstone:
        stat, p_value = ks_2samp(df_clean[column].dropna(), df_capstone[column])
        results.append((column, stat, p_value))

# Convertir los resultados en un DataFrame para una mejor visualización
results_df = pd.DataFrame(results, columns=['Column', 'KS Statistic', 'P-value'])

# Mostrar los resultados
results_df

# %%
graficos_variables(df_capstone)

# %% [markdown]
# # Correlación de las variables

# %% [markdown]
# ## Gráfico de correlación entre todas las variables

# %%
# Calcula la matriz de correlación
correlation_matrix = df_capstone.select_dtypes(include=[np.number]).corr()

# Configura el tamaño de la figura
plt.figure(figsize=(10, 8))

# Crea una matriz de anotaciones personalizada
annotations = correlation_matrix.applymap(lambda x: f'{x:.2f}' if abs(x) >= 0.25 else '')

# Crea un mapa de calor de la matriz de correlación
sns.heatmap(correlation_matrix, annot=annotations, fmt="", cmap="coolwarm", annot_kws={"size": 8})

# Añade etiquetas y título
plt.title("Matriz de correlación de variables")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# %% [markdown]
# ## Correlación entre variables predictoras y variable objetivo

# %%
# Extraemos las correlaciones específicas con 'RECUPERACION_PONDERADA'
target_correlation = correlation_matrix['Recuperacion_Turno'].drop('Recuperacion_Turno')  # Excluir la auto-correlación
# Ordenamos las correlaciones de mayor a menor
target_correlation_sorted = target_correlation.sort_values(ascending=False)
# Creamos el gráfico de barras para visualizar estas correlaciones ordenadas
plt.figure(figsize=(10, 6))
target_correlation_sorted.plot(kind='bar', color='skyblue')
plt.title('Correlación de las variables con variable objetivo (Recuperación)')
plt.xlabel('Columnas')
plt.ylabel('Coeficiente de Correlación')
plt.ylim(-1, 1)
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# %% [markdown]
# # Revisión de las variables

# %% [markdown]
# ## Detección de Autocolinealidad (VIF)

# %%
df_capstone.drop(columns=['Recuperacion_Turno','Fecha'])

# %%
X = df_capstone.drop(columns=['Recuperacion_Turno','Fecha','Turno']) # se elimina la variable Y:medv
X = sm.add_constant(X)

# Calcular el VIF para cada variable predictora
VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Crear un DataFrame para mostrar los resultados tabulados
vif_df = pd.DataFrame({'Variable': X.columns, 'VIF': VIF})

# Mostrar la tabla
print(vif_df.sort_values('VIF', ascending=False))

# %%
X_2 = df_capstone.drop(columns=['Recuperacion_Turno','Fecha','Turno','Alim CuI']) # se elimina la variable Y:medv
X_2 = sm.add_constant(X_2)

# Calcular el VIF para cada variable predictora
VIF_2 = [variance_inflation_factor(X_2.values, i) for i in range(X_2.shape[1])]

# Crear un DataFrame para mostrar los resultados tabulados
vif_df_2 = pd.DataFrame({'Variable': X_2.columns, 'VIF': VIF_2})

# Mostrar la tabla
print(vif_df_2.sort_values('VIF', ascending=False))

# %% [markdown]
# ## Step Forward Selecction

# %% [markdown]
# ### Lasso

# %%
#linear_model = LinearRegression()
laso=Lasso()
score='explained_variance'
features_select='auto'
#score='neg_root_mean_squared_error',
sfs_model = SequentialFeatureSelector(laso,
                                      n_features_to_select = features_select,
                                      direction='forward',
                                      scoring=score,
                                      cv=5,
                                      n_jobs=-1)

X_sfs = df_capstone.drop(columns=['Recuperacion_Turno','Turno','Fecha','Alim CuI'])
Y_sfs = df_capstone['Recuperacion_Turno']
# Perform SFFS
sfs_model.fit(X_sfs, Y_sfs)

# %%
variables_sfs=sfs_model.get_feature_names_out()
variables_sfs

# %%
X = X_sfs
y = Y_sfs

# Lista para guardar métrica y variable seleccionada
scores = []
variables_seleccionadas = []

remaining_variables = list(X.columns)
selected_variables = []

for i in range(len(remaining_variables)):
    best_score = -float("inf")
    best_variable = None

    for variable in remaining_variables:
        # grupo de variables actuales + la nueva variable
        candidate_variables = selected_variables + [variable]

        # Ajusta el modelo con variables actuales
        model = LinearRegression().fit(X[candidate_variables], y)

        # Calcula el R²
        n = X_sfs.shape[0]  # Número de muestras
        p = X_sfs.shape[1]  # Número de predictores
        r2=r2_score(y, model.predict(X[candidate_variables]))
        r2_adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        score = r2_adjusted

        # Si el modelo mejora, actualiza el mejor score y variable
        if score > best_score:
            best_score = score
            best_variable = variable

    # Añadir variable y registrar el score
    if best_variable:
        selected_variables.append(best_variable)
        remaining_variables.remove(best_variable)
        scores.append(best_score)
        variables_seleccionadas.append(best_variable)

#print("Variables Seleccionadas:", variables_seleccionadas)
#print("Scores en cada paso:", scores)

results_sfs=pd.DataFrame({'Variables Seleccionadas':variables_seleccionadas,'Scores':scores})
results_sfs


# %%
variables_seleccionadas = variables_seleccionadas
scores = scores

plt.figure(figsize=(12, 6))
sns.lineplot(x=variables_seleccionadas, y=scores, marker='o')
plt.xlabel('Variables Seleccionadas')
plt.xticks(rotation=90)
plt.ylabel('Desempeño Acumulado')
plt.title('Impacto Acumulativo de Variables Seleccionadas')
plt.show()

# %% [markdown]
# ### Random Forest

# %%
#linear_model = LinearRegression()
RandomForest=RandomForestRegressor()
score='explained_variance'
features_select='auto'
#score='neg_root_mean_squared_error',
sfs_model_2 = SequentialFeatureSelector(RandomForest,
                                      n_features_to_select = features_select,
                                      direction='forward',
                                      scoring=score,
                                      cv=5,
                                      n_jobs=-1)

X_sfs_2 = df_capstone.drop(columns=['Recuperacion_Turno','Turno','Fecha','Alim CuI'])
Y_sfs_2 = df_capstone['Recuperacion_Turno']
# Perform SFFS
sfs_model_2.fit(X_sfs_2, Y_sfs_2)

# %%
variables_sfs_2=sfs_model_2.get_feature_names_out()
variables_sfs_2

# %%
X = X_sfs_2
y = Y_sfs_2

# Lista para guardar métrica y variable seleccionada
scores = []
variables_seleccionadas = []

remaining_variables = list(X.columns)
selected_variables = []

for i in range(len(remaining_variables)):
    best_score = -float("inf")
    best_variable = None

    for variable in remaining_variables:
        # grupo de variables actuales + la nueva variable
        candidate_variables = selected_variables + [variable]

        # Ajusta el modelo con variables actuales
        model = RandomForestRegressor().fit(X[candidate_variables], y)

        # Calcula el R²
        n = X_sfs_2.shape[0]  # Número de muestras
        p = X_sfs_2.shape[1]  # Número de predictores
        r2=r2_score(y, model.predict(X[candidate_variables]))
        r2_adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        score = r2_adjusted

        # Si el modelo mejora, actualiza el mejor score y variable
        if score > best_score:
            best_score = score
            best_variable = variable

    # Añadir variable y registrar el score
    if best_variable:
        selected_variables.append(best_variable)
        remaining_variables.remove(best_variable)
        scores.append(best_score)
        variables_seleccionadas.append(best_variable)

#print("Variables Seleccionadas:", variables_seleccionadas)
#print("Scores en cada paso:", scores)

results_sfs_2=pd.DataFrame({'Variables Seleccionadas':variables_seleccionadas,'Scores':scores})
results_sfs_2


# %%
variables_seleccionadas = variables_seleccionadas
scores = scores

plt.figure(figsize=(12, 6))
sns.lineplot(x=variables_seleccionadas, y=scores, marker='o')
plt.xlabel('Variables Seleccionadas')
plt.xticks(rotation=90)
plt.ylabel('Desempeño Acumulado')
plt.title('Impacto Acumulativo de Variables Seleccionadas')
plt.show()

# %% [markdown]
# # Archivo

# %%
df_capstone.info()

# %%
df_capstone.rename({'Tratamiento turno':'Tratamiento_Turno','Ag (ppm)':'Ag','Pb (ppm)':'Pb','Fe %':'Fe','P80 Alim Ro300':'P80_Alim_Ro300','pH Ro300':'pH_Ro300','Alim CuT':'Alim_CuT','Alim CuS':'Alim_CuS','Sol Cit':'Sol_Cit'},axis=1,inplace=True)

# %%
df_capstone.info()

# %%
df_capstone.to_excel('datos/df_capstone.xlsx', index=False)
print('fin ejecución EDA.py')