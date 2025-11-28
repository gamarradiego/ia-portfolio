---
title: "Práctica 16: Explorando GCloud"
date: 2025-11-25
---

# Práctica 16: Explorando GCloud

## Contexto
En esta práctica se explora por primera vez el entorno de Google Cloud Platform (GCP) mediante un laboratorio guiado de Google Skills. El objetivo es familiarizarse con la consola, los proyectos, el sistema de identidades y accesos (IAM) y la habilitación de APIs.

---

## Objetivos
- Iniciar y utilizar un laboratorio temporal de Google Cloud en Google Skills Boost.
- Navegar la consola de GCP e identificar proyectos, recursos y menús principales.
- Visualizar y modificar roles y permisos en Cloud IAM.
- Habilitar APIs y revisar documentación técnica desde la consola.

---

## Actividades
| Actividad                                   | Tiempo estimado | Resultado esperado                                                     |
|----------------------------------------------|-----------------|------------------------------------------------------------------------|
| Iniciar laboratorio e ingresar a la consola  | 10 min          | Acceso correcto a GCP con credenciales temporales                     |
| Explorar proyectos y navegación general      | 10 min          | Identificación del proyecto activo y comprensión del menú de servicios |
| Revisar y modificar roles en IAM            | 10 min          | Visualización de roles y asignación exitosa de un nuevo rol           |
| Habilitar APIs y revisar documentación       | 10 min          | Dialogflow API habilitada y documentación consultada                  |
| Finalizar laboratorio                        | 5 min           | Laboratorio cerrado y recursos liberados                              |

---

## Desarrollo

### 1. Inicio del laboratorio
El laboratorio se inicia desde Google Skills Boost seleccionando **Start Lab**, lo que activa una instancia temporal sin costo.
Al iniciar, el sistema genera:

- Credenciales temporales (username y password)
- Un proyecto temporal asignado
- Un Project ID único para ese laboratorio

Estas credenciales representan una identidad temporal gestionada mediante **Cloud Identity and Access Management (IAM)** y permiten realizar acciones limitadas dentro del proyecto asignado.
Desde la interfaz del laboratorio se accede directamente a la **Google Cloud Console**, donde se inicia sesión con las credenciales proporcionadas.

### 2. Exploración de la consola y los proyectos
Una vez dentro de la consola, se identifica el proyecto activo mediante un recuadro en la esquina superior izquierda, donde se muestran:

- Nombre del proyecto
- ID del proyecto
- Número del proyecto

Desde la opción **Select a project** se visualiza la lista de proyectos disponibles. En la misma aparece el proyecto compartido **"Qwiklabs Resources"**, de solo lectura, que contiene recursos y datasets utilizados por varios laboratorios. No es necesario abrirlo; el trabajo se realiza siempre sobre el proyecto temporal del laboratorio.

En el **Navigation Menu** se exploran las principales categorías de servicios: Compute, Storage, IAM, APIs, Big Data, entre otros. Esta navegación permite identificar la estructura central de GCP.

### 3. Revisión y modificación de roles en IAM
En la tercera tarea se trabaja con permisos utilizando **IAM (Identity and Access Management)**.

- Se accede al servicio **IAM** para visualizar la lista de identidades con permiso dentro del proyecto.
- Se localiza el usuario temporal del laboratorio y se confirma que posee el rol **Editor**, el cual permite:

    - Crear, modificar y eliminar recursos
    - Pero no administrar usuarios del proyecto (solo un Owner puede hacerlo)

Luego se simula un cambio de permisos:

- Se agrega un nuevo miembro (“User 2”)
- Se le asigna el rol **Viewer**, que solo permite visualizar recursos
- Se guarda la configuración y se verifica que aparezca en el listado

Esta tarea permite comprender cómo GCP gestiona accesos de forma granular mediante permisos, roles y principios de menor privilegio.

### 4. Habilitación de APIs y servicios
En esta tarea se ingresa a **APIs & Services → Library**, donde se encuentran más de 200 APIs disponibles en Google Cloud.

Desde la barra de búsqueda se localiza y selecciona **Dialogflow API**.
Esta API permite crear aplicaciones conversacionales sin necesidad de gestionar directamente modelos de machine learning.

Los pasos realizados:

1. Abrir la página de la API.
2. Seleccionar **Enable** para habilitarla en el proyecto.
3. Verificar que quedó habilitada regresando al panel de APIs.
4. Probar la opción **Try this API**, que abre la documentación interactiva con ejemplos de uso.

Se observa que muchas APIs en GCP ofrecen métricas sobre tráfico, latencia y errores, útiles para monitoreo y optimización.

### 5. Finalización del laboratorio

Una vez completadas todas las tareas, se finaliza el entorno seleccionando **End Lab** desde Google Skills Boost.
Esto revoca las credenciales temporales y borra todos los recursos utilizados, dado que el proyecto es efímero.

---

## Reflexión
- Se comprendió cómo funciona el modelo de identidades temporales de Google Skills Boost.
- Se exploró la estructura básica de GCP y su navegación interna, lo que facilita el uso de servicios más avanzados.
- Se identificó la importancia de IAM para gestionar permisos de manera segura y controlada.

## Referencias
- [Google Skills Boost –  A Tour of Google Cloud Hands-on Labs ](https://www.skills.google/focuses/2794?catalog_rank=%7B%22rank%22%3A3%2C%22num_filters%22%3A2%2C%22has_search%22%3Atrue%7D&locale=en&parent=catalog&search_id=60924676)