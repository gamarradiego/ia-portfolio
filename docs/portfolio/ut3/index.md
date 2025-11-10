---
title: "UT3 — Inicio"
date: 2025-09-30
---

# Unidad Temática 3

## 📘 Introducción

La **Unidad Temática 3 (UT3)** aborda los fundamentos y aplicaciones prácticas del **Computer Vision**, un área clave dentro del Deep Learning enfocada en permitir que las computadoras comprendan e interpreten imágenes del mundo real.

Comenzamos explorando la **arquitectura de las redes convolucionales (CNNs)**, entendiendo su estructura jerárquica y cómo los filtros convolucionales permiten extraer características visuales como bordes, texturas y patrones complejos. A partir de esa base, implementamos desde cero una CNN simple para clasificar imágenes del conjunto **CIFAR-10**, analizando su desempeño, los problemas de *overfitting* y las estrategias de regularización que pueden aplicarse.

Luego avanzamos hacia el **Transfer Learning**, utilizando modelos preentrenados como **MobileNetV2** y ajustándolos a nuevas tareas mediante *fine-tuning*. Esto permitió observar cómo los modelos previamente entrenados en grandes datasets (como ImageNet) pueden reutilizar sus representaciones para mejorar la precisión y acelerar el entrenamiento en datasets más pequeños.

También se estudiaron las técnicas de **data augmentation**, aplicadas para aumentar la robustez del modelo y su capacidad de generalización, junto con estrategias prácticas de evaluación mediante métricas y visualizaciones de desempeño.

Finalmente, se introdujeron los conceptos de **detección de objetos** y **segmentación de imágenes**, utilizando arquitecturas modernas como **YOLO (You Only Look Once)** para detección en tiempo real y **SAM (Segment Anything Model)** para segmentación automática.

---

## 📂 Prácticas de la UT3

- [Práctica 9: CNNs y Transfer Learning con TensorFlow/Keras](01-practica9.md)  
- [Práctica 10: Data Augmentation Avanzado & Explicabilidad](02-practica10.md)
- [Práctica 11: YOLOv8 Fine-tuning & Tracking](03-practica11.md) 
- [Práctica 12: SAM Segmentation - Pretrained vs Fine-tuned](04-practica12.md) 
 
---