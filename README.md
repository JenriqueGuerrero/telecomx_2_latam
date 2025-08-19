# 📉 Telecom X – Predicción de Cancelación de Clientes (Churn Prediction)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.2+-orange?style=for-the-badge&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-0.41+-red?style=for-the-badge)

Proyecto de análisis y modelado predictivo para anticipar la cancelación de clientes en una empresa de telecomunicaciones. Utiliza técnicas de machine learning, análisis exploratorio y herramientas de interpretabilidad como SHAP, con el objetivo de entender los factores que influyen en el abandono y proponer estrategias de retención efectivas.

## 📁 Estructura del Repositorio
📦 TelecomX-Churn
├── data/ # Datos procesados y tratados
├── notebooks/ # Notebooks de análisis y modelado
├── models/ # Modelos entrenados (opcional)
├── outputs/ # Resultados y visualizaciones
├── informe_cancelacion_telecom_x.md # Informe técnico en Markdown
├── requirements.txt # Paquetes necesarios
└── README.md # Este archivo


## 🧠 Objetivo del Proyecto

Desarrollar un modelo capaz de predecir si un cliente cancelará su servicio, con base en sus características de contrato, historial de pagos y uso de servicios. Esto permite a la empresa tomar acciones proactivas y reducir su tasa de cancelación (churn rate).

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Scikit-learn**: Modelado de machine learning
- **Pandas & NumPy**: Manipulación de datos
- **Matplotlib & Seaborn**: Visualización
- **SHAP**: Interpretabilidad de modelos
- **Imbalanced-learn**: Balanceo de clases (SMOTE)
- **Jupyter Notebook**: Análisis interactivo

## 🧪 Proceso de Desarrollo

### 1. Carga y Limpieza de Datos
- Eliminación de ID único
- Codificación de variables categóricas con `get_dummies`
- Balanceo de clases con **SMOTE**
- Normalización con **StandardScaler**

### 2. Análisis Exploratorio (EDA)
- Histogramas, boxplots y gráficos de correlación
- Identificación de variables influyentes
- Análisis de churn por tipo de contrato, servicios y método de pago

### 3. Selección de Características
- **SelectKBest** con `f_classif`
- K óptimo = 29 variables seleccionadas por rendimiento

### 4. Entrenamiento de Modelos
- Regresión Logística (`LogisticRegression`)
- Random Forest (`RandomForestClassifier`)
- Gradient Boosting (`GradientBoostingClassifier`)
- K-Nearest Neighbors (`KNN`)
- Ensemble (`VotingClassifier`: LR + GB)

### 5. Optimización de Hiperparámetros
- **GridSearchCV** con validación cruzada estratificada

### 6. Evaluación
- **Métricas**: Accuracy, Precision, Recall, F1 Score
- Matrices de confusión
- Curvas ROC y AUC
- Interpretación con **SHAP** y coeficientes logísticos

## 🎯 Insights y Recomendaciones

Basado en el análisis SHAP y la interpretación del modelo:
