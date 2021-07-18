### Importar modulos ===========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from skforecast.ForecasterAutoreg import ForecasterAutoreg
#from skforecast.ForecasterCustom import ForecasterCustom
#from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
#from skforecast.model_selection import time_series_spliter
#from skforecast.model_selection import cv_forecaster
from skforecast.model_selection import backtesting_forecaster_intervals

import warnings
warnings.filterwarnings('ignore')
# ==============================================================================


### Importar datos =============================================================
url = 'https://raw.githubusercontent.com/MilenitaR/Datos/main/MB.csv'
datos = pd.read_csv(url, sep=',')
# ==============================================================================

### Preparar datos =============================================================
datos['Fecha'] = pd.to_datetime(datos['Fecha'], format='%Y-%m-%d %H:%M:%S')
datos = datos.set_index('Fecha')
datos = datos.rename(columns={'Valor': 'Y'})
datos = datos.asfreq('T')
datos = datos['Y']
datos = datos.sort_index()
# ==============================================================================

def normalizar(datos):
  return ( ( datos-datos.min() ) / ( datos.max()-datos.min() ) )

### Datos exogenos =============================================================
periodo_ = 10

# Media
datos_exog = datos.rolling(window=periodo_, closed='right').mean() + 0.1
datos_exog = datos_exog[periodo_:]                # Elimina los primeros datos
datos = datos[periodo_:]                          # Elimina los primeros datos
# ==============================================================================

### Graficar datos =============================================================
fig, ax = plt.subplots(figsize=(9, 4))
datos.plot(ax=ax, label='Y')
datos_exog.plot(ax=ax, label='Variable exógena')
ax.legend();
plt.show()
# ==============================================================================

### Separar datos (train y test) ===============================================
#   División lineal con train_test_split()
muestra_test = 0.3              # 30% de los datos se usaran para hacer el test
 
datos_train, \
datos_test = train_test_split(datos, test_size = muestra_test, shuffle = False)

datosexog_train, \
datos_exog_test = train_test_split(datos, test_size =  muestra_test,
                                            shuffle = False)
# ==============================================================================

# Crear y entrenar forecaster auto-regresivo ===================================
forecaster_rf = ForecasterAutoreg(
                  regressor = RandomForestRegressor(random_state=123),
                  lags      = 8 )

forecaster_rf.fit(y=datos_train, exog=datos_exog_train)
# ==============================================================================

# Predicciones =================================================================
steps = len(datos_test)
predicciones = forecaster_rf.predict(steps=steps)
# Se añade el índice temporal a las predicciones
predicciones = pd.Series(data=predicciones, index=datos_test.index)
predicciones.head()
# ==============================================================================

# Gráfico
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
datos_train.plot(ax=ax, label='train')
datos_test.plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
plt.show()

# Error
# ==============================================================================
error_mse = mean_squared_error(
                y_true = datos_test,
                y_pred = predicciones
            )
print(f"Error de test (mse): {error_mse}")


# Grid search de hiperparámetros
# ==============================================================================
forecaster_rf = ForecasterAutoreg(
                    regressor = RandomForestRegressor(random_state=123),
                    lags      = 12 # Este valor será remplazado en el grid search
                 )

# Hiperparámetros del regresor
param_grid = {'n_estimators': [100, 500],
              'max_depth': [3, 5, 10]}

# Lags utilizados como predictores
#lags_grid = [10, 20]
lags_grid = [3,3]

resultados_grid = grid_search_forecaster(
                          forecaster            = forecaster_rf,
                          y                     = datos_train,
                          param_grid            = param_grid,
                          lags_grid             = lags_grid,
                          steps                 = 10,
                          method                = 'cv',
                          metric                = 'mean_squared_error',
                          initial_train_size    = int(len(datos_train)*0.5),
                          allow_incomplete_fold = False,
                          return_best           = True,
                          verbose               = False )



# Resultados Grid Search
# ==============================================================================
resultados_grid

### Modelo final

# Crear y entrenar forecaster con mejores hiperparámetros
# ==============================================================================
regressor = RandomForestRegressor(max_depth=10, n_estimators=500, random_state=123)

forecaster_rf = ForecasterAutoreg(
                    regressor = regressor,
                    lags      = 20
                )

forecaster_rf.fit(y=datos_train)
# Predicciones
# ==============================================================================
predicciones = forecaster_rf.predict(steps=steps)
# Se añade el índice a las predicciones
predicciones = pd.Series(data=predicciones, index=datos_test.index)

# Gráfico
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
datos_train.plot(ax=ax, label='train')
datos_test.plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
plt.show()


# Importancia predictores
# ==============================================================================
impotancia = forecaster_rf.get_feature_importances()
dict(zip(forecaster_rf.lags, impotancia))


### Comparacion con statsmodels

# Modelo autorregresivo lineal statsmodels
# ==============================================================================
from statsmodels.tsa.ar_model import AutoReg
lags = 15

modelo_ar = AutoReg(datos_train, lags=lags)
res = modelo_ar.fit()
predicciones_statsmodels = res.predict(start=datos_test.index[0], end=datos_test.index[-1])

# Modelo autorregresivo lineal Forecaster
# ==============================================================================
regressor = LinearRegression()
forecaster = ForecasterAutoreg(regressor=regressor, lags=lags)
forecaster.fit(y=datos_train)
predicciones_forecaster = forecaster.predict(steps=steps)

# Verificación de que las predicciones de ambos modelos son iguales
# ==============================================================================
print(np.allclose(predicciones_statsmodels.values, predicciones_forecaster))

# Verificación de que los coeficients de ambos modelos son iguales
# ==============================================================================
print(np.allclose(res.params.values[1:], forecaster.get_coef()))



# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================

# Backtest con intervalos de predicción
# ==============================================================================
muestra_test = 0.5              # 30% de los datos se usaran para hacer el test
 
datos_train, \
datos_test = train_test_split(datos, test_size = muestra_test, shuffle = False)

datos_exog_train, \
datos_exog_test = train_test_split(datos, test_size =  muestra_test,
                                            shuffle = False)


steps = 12
regressor = LinearRegression()
forecaster = ForecasterAutoreg(regressor=regressor, lags=10)

metric, predictions = backtesting_forecaster_intervals(
                            forecaster = forecaster,
                            y          = datos,
                            initial_train_size = len(datos_train),
                            steps      = steps,
                            metric     = 'mean_squared_error',
                            interval            = [1, 79],
                            n_boot              = 80,
                            in_sample_residuals = True,
                            verbose             = True
                       )

print(metric)

# Se añade índice datetime
predictions = pd.DataFrame(data=predictions, index=datos_test.index)

# Gráfico
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
#datos_train.plot(ax=ax, label='train')
datos_test.plot(ax=ax, label='test')
predictions.iloc[:, 0].plot(ax=ax, label='predictions')
ax.fill_between(predictions.index,
                predictions.iloc[:, 1],
                predictions.iloc[:, 2],
                alpha=0.5)
ax.legend();
plt.show()


# Fuente: https://www.cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn.html
