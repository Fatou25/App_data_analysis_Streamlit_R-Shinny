import streamlit as st
import pandas as pd
import numpy as np
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data Analysis App", layout="wide")
st.title("Application interactive de Data Analysis")

# ---- Import de données ----
st.sidebar.header("Import des données")
uploaded_file = st.sidebar.file_uploader("CSV ou Excel", type=["csv", "xlsx"])
sep = st.sidebar.text_input("Séparateur CSV (si CSV)", value=",")
sheet = st.sidebar.text_input("Nom de la feuille (si Excel)", value="")

@st.cache_data
def load_data(file, sep, sheet):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, sep=sep)
    else:
        return pd.read_excel(file, sheet_name=sheet if sheet else 0)

df = load_data(uploaded_file, sep, sheet)

if df is None:
    st.info("Charge un CSV/Excel pour commencer.")
    st.stop()

st.subheader("Aperçu")
st.dataframe(df.head())

# ---- Profil de variables ----
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]

st.write(f"Variables numériques: {len(numeric_cols)} | Catégorielles: {len(cat_cols)}")

# Stats descriptives
with st.expander("Statistiques descriptives"):
    st.markdown("**Numériques**")
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T)
    else:
        st.write("Aucune variable numérique.")
    st.markdown("**Catégorielles**")
    if cat_cols:
        st.dataframe(df[cat_cols].astype("category").describe().T)
    else:
        st.write("Aucune variable catégorielle.")

# ---- Visualisations ----
st.subheader("Visualisations")

tab1, tab2, tab3 = st.tabs(["Univariées", "Bivariées", "Multivariées"])

with tab1:
    col = st.selectbox("Variable pour histogramme / barplot", options=df.columns)
    if col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), ax=ax, kde=True)
        ax.set_title(f"Histogramme de {col}")
        st.pyplot(fig)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col].dropna(), ax=ax)
        ax.set_title(f"Boxplot de {col}")
        st.pyplot(fig)
    else:
        counts = df[col].astype(str).value_counts().head(50)
        fig, ax = plt.subplots()
        sns.barplot(x=counts.values, y=counts.index, ax=ax)
        ax.set_title(f"Barplot de {col} (Top 50)")
        st.pyplot(fig)

with tab2:
    x = st.selectbox("X", options=df.columns, key="x_bi")
    y = st.selectbox("Y", options=df.columns, key="y_bi")
    hue = st.selectbox("Couleur (optionnel)", options=["(aucune)"] + df.columns.tolist(), key="hue_bi")
    hue = None if hue == "(aucune)" else hue
    fig, ax = plt.subplots()
    if x in numeric_cols and y in numeric_cols:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    elif x in cat_cols and y in numeric_cols:
        sns.boxplot(data=df, x=x, y=y, ax=ax)
    elif x in numeric_cols and y in cat_cols:
        sns.boxplot(data=df, x=y, y=x, ax=ax)  # pivot pour boxplot
    else:
        ct = pd.crosstab(df[x].astype(str), df[y].astype(str))
        sns.heatmap(ct, annot=False, ax=ax)
        ax.set_title("Heatmap (table de contingence)")
    st.pyplot(fig)

with tab3:
    st.markdown("**Heatmap des corrélations (numériques)**")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=False, cmap="vlag", ax=ax)
        st.pyplot(fig)
    else:
        st.write("Pas assez de variables numériques.")

# ---- Modélisation ----
st.subheader("Modèles prédictifs")

target = st.selectbox("Variable cible (y)", options=df.columns)
features = st.multiselect("Variables explicatives (X)", options=[c for c in df.columns if c != target],
                          default=[c for c in df.columns if c != target][:min(5, len(df.columns)-1)])

if not features:
    st.warning("Sélectionne au moins une variable explicative.")
    st.stop()

# Détection type de tâche: classification si cible non-numérique ou peu de valeurs distinctes
is_classification = (target not in numeric_cols) or (df[target].nunique() <= max(10, int(0.05*len(df))))

st.write(f"Tâche détectée : **{'Classification' if is_classification else 'Régression'}**")

model_choice = st.selectbox("Modèle",
                            ["Régression linéaire", "Random Forest"] if not is_classification
                            else ["Logistic Regression", "Random Forest"])

# Préparation des données
X = df[features].copy()
y = df[target].copy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Préprocesseur
num_ft = [c for c in features if c in numeric_cols]
cat_ft = [c for c in features if c in cat_cols]

if is_classification:
    num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    preproc = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_ft),
            ("cat", cat_transformer, cat_ft),
        ]
    )
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
else:
    num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    preproc = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_ft),
            ("cat", cat_transformer, cat_ft),
        ]
    )
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        model = LinearRegression()

pipe = Pipeline(steps=[("preproc", preproc), ("model", model)])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

st.markdown("**Évaluation**")
if is_classification:
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    st.write(f"Accuracy: {acc:.3f} | F1 (pondéré): {f1:.3f}")
    st.text("Classification report :")
    st.code(classification_report(y_test, pred))
else:
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    st.write(f"R²: {r2:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")
