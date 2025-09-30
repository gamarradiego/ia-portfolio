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
- N


---

## Guía de formato y ejemplos (MkDocs Material)

Usá estos ejemplos para enriquecer tus entradas. Todos funcionan con la configuración del template.

### Admoniciones

!!! note "Nota"
    Este es un bloque informativo.

!!! tip "Sugerencia"
    Considerá alternativas y justifica decisiones.

!!! warning "Atención"
    Riesgos, limitaciones o supuestos relevantes.

### Detalles colapsables

???+ info "Ver desarrollo paso a paso"
    - Paso 1: preparar datos
    - Paso 2: entrenar modelo
    - Paso 3: evaluar métricas

### Código con resaltado y líneas numeradas

```python hl_lines="2 6" linenums="1"
def train(
    data_path: str,
    epochs: int = 10,
    learning_rate: float = 1e-3,
) -> None:
    print("Entrenando...")
    # TODO: implementar
```

### Listas de tareas (checklist)

- [ ] Preparar datos
- [x] Explorar dataset
- [ ] Entrenar baseline

### Tabla de actividades con tiempos

| Actividad           | Tiempo | Resultado esperado               |
|---------------------|:------:|----------------------------------|
| Revisión bibliográfica |  45m  | Lista de fuentes priorizadas     |
| Implementación      |  90m   | Script ejecutable/documentado    |
| Evaluación          |  60m   | Métricas y análisis de errores   |

### Imágenes con glightbox y atributos

Imagen directa (abre en lightbox):

![Diagrama del flujo](../assets/placeholder.png){ width="420" }

Click para ampliar (lightbox):

[![Vista previa](../assets/placeholder.png){ width="280" }](../assets/placeholder.png)

### Enlaces internos y relativos

Consultá también: [Acerca de mí](../acerca.md) y [Recursos](../recursos.md).

### Notas al pie y citas

Texto con una afirmación que requiere aclaración[^nota].

[^nota]: Esta es una nota al pie con detalles adicionales y referencias.

### Emojis y énfasis

Resultados destacados :rocket: :sparkles: y conceptos `clave`.
