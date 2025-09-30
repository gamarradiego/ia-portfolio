---
title: "Práctica 6: Clustering y PCA - Mall Customer Segmentation"
date: 2025-09-09
---

# Práctica 6: Clustering y PCA - Mall Customer Segmentation

## Contexto
En esta práctica se analizará el **Mall Customer Segmentation**, que contiene información demográfica y de consumo de clientes de un centro comercial.  
A lo largo de la práctica se aplican técnicas de **clustering (K-Means) y PCA** para categorizar clientes.


## Objetivos
- Explorar el dataset y lograr comprender sus distintas variables.
- Lograr identificar segmentos de clientes.
- Aplicar **normalización** y comparar distintos escaladores.  
- Implementar **clustering con K-Means** y evaluar con *silhouette score*.  
- Aplicar **PCA** para reducción dimensional y visualización.  
- Comparar **feature selection (forward/backward)** vs PCA.  
- Explorar otros métodos de clustering: **DBSCAN, HDBSCAN, GMM, Spectral y Agglomerative**.  


## Actividades (con tiempos estimados)
| Actividad                   | Tiempo | Resultado esperado                         |
|-----------------------------|:------:|--------------------------------------------|
| Descarga y exploración      |  20m   | Dataset limpio y comprendido               |
| Análisis exploratorio (EDA) |  50m   | Estadísticas, gráficos, correlaciones      |
| Normalización               |  40m   | Datos escalados con distintos métodos      |
| PCA y Feature Selection     |  45m   | Reducción y selección de variables         |
| Clustering y evaluación     |  90m   | Comparación de algoritmos y métricas       |
| Documentación y reflexión   |  30m   | Portafolio actualizado  

## Desarrollo
Resumen de lo realizado, decisiones y resultados intermedios.

1. **EDA inicial**: se analizaron distribuciones, outliers, correlaciones y diferencias por género.
El dataset consiste de 5 columnas y tiene la siguiente forma:

| CustomerID | Genre  | Age | Annual Income (k$) | Spending Score (1-100) |
|------------|--------|-----|---------------------|-------------------------|
| 1        | Male   | 19  | 15                  | 39                      |
| 2        | Male | 21  | 15                  | 81                      |
| 3        | Female | 20  | 16                  | 6                      |
| 4        | Female   | 23  | 16                 | 77                      |
| 5        | Female | 31  | 17                  | 40                      |

El promedio de edad es de 38-39 años así como el ingreso anual promedia unos $60.6k. La distribución de hombres y mujeres es de 44% contra 56%. Para detectar outliers se utilizó IQR y solo se hallaron 2 (1.0%) en el caso de Annual Income.

2. **Normalización**: se compararon *MinMaxScaler, StandardScaler y RobustScaler*.  
   - MinMax y Standard ofrecieron resultados competitivos; Robust fue más tolerante a outliers.  
3. **PCA**: se determinó que con 2 componentes se retenía cerca del **60% de la varianza**, útil para visualización.  
4. **Feature Selection**: se aplicó *Forward* y *Backward Selection* con KMeans+Silhouette, comparando contra PCA.  
5. **Clustering con K-Means**: se evaluaron distintos K (2–8) usando **Elbow Method** y **Silhouette**, eligiendo un número razonable de clusters (entre 3 y 5) según contexto de negocio.  
6. **Otros algoritmos probados**:  
   - **DBSCAN/HDBSCAN**: identificación de clusters de densidad, con bajo porcentaje de ruido.  
   - **Gaussian Mixture Models (GMM)**: enfoque probabilístico, selección de componentes con AIC/BIC.  
   - **Spectral Clustering**: uso de afinidad RBF para datos no lineales.  
   - **Agglomerative Clustering**: con linkage *ward*. 


## Evidencias
- Capturas, enlaces a notebooks/repos, resultados, gráficos

## Reflexión
- Aprendí sobre la importancia que tiene escalar correctamente los datos antes de hacer clustering.
- Comparar PCA con selección de features dio perspectivas distintas: PCA es útil para visualización, mientras que selección es más interpretable.  
- Queda claro que no existe un único “mejor algoritmo” sino que la elección depende del contexto y la métrica. 

## Referencias
- Documentación oficial de [scikit-learn](https://scikit-learn.org/stable/).
- Los apuntes y presentaciones trabajadas en clase sobre clustering y PCA.
- 


---
