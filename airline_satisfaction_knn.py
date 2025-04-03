import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Cargar datos
df = pd.read_csv("/Users/clarapinto/Desktop/MineriaDatos/data/test.csv")

target = "satisfaction"
selected_cols = [
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
    target
]
df = df[selected_cols]

# Separar por tipo de variable
categorical = df.select_dtypes(include=["object", "category"]).columns.to_list()
numerical = [col for col in df.columns if col not in categorical and col != target]  # Excluir target

# Graficar distribución de variables numéricas
for col in numerical:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribución de {col}")
    plt.show(block=False)

# Graficar relación con la variable a predecir 
for col in numerical:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[target], y=df[col])
    plt.title(f"{col} vs {target}")
    plt.show()

# Preprocesador
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
])

# Modelo con Pipeline
knn_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# Separar datos
X = df.drop(columns=[target])
y = df[target].map({"neutral or dissatisfied": 0, "satisfied": 1})  # Ajusta si los valores son distintos

# Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(knn_pipeline, X, y, cv=cv, scoring='roc_auc')

print(f"ROC-AUC mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
