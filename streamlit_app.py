import streamlit as st
import pandas as pd

from fraud_detector import (
    TARGET_COLUMN,
    create_visualizations,
    generate_sample_data,
    plot_confusion_matrix,
    run_eda,
    train_model,
)

st.set_page_config(page_title="Détecteur de fraude", layout="wide")
st.title("Détecteur de fraude")
st.caption("EDA, modélisation et visualisation interactives avec Streamlit")

source = st.sidebar.radio("Source de données", ["Jeu de démonstration", "Uploader un CSV"])

if source == "Uploader un CSV":
    upload = st.file_uploader("Fichier CSV", type=["csv"])
    if upload is None:
        st.info("Chargez un fichier CSV pour commencer.")
        st.stop()
    data = pd.read_csv(upload)
else:
    n_rows = st.sidebar.slider("Nombre de lignes (demo)", min_value=500, max_value=5000, step=500, value=1500)
    data = generate_sample_data(n_samples=n_rows)

if data.empty:
    st.error("Le jeu de données est vide.")
    st.stop()

target_column = st.selectbox(
    "Colonne cible (fraude)",
    options=list(data.columns),
    index=(data.columns.get_loc(TARGET_COLUMN) if TARGET_COLUMN in data.columns else 0),
)

st.subheader("Aperçu des données")
st.dataframe(data.head(20), use_container_width=True)

try:
    eda = run_eda(data, target_column=target_column)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Répartition de la cible")
    st.dataframe(eda["class_distribution"].rename("count"), use_container_width=True)
with right:
    st.subheader("Valeurs manquantes")
    st.dataframe(eda["missing_values"].rename("missing_count"), use_container_width=True)

st.subheader("Résumé statistique (numérique)")
st.dataframe(eda["numeric_summary"], use_container_width=True)

st.subheader("Visualisations")
for figure in create_visualizations(data, target_column=target_column):
    st.pyplot(figure)

if st.button("Entraîner le modèle"):
    try:
        artifacts = train_model(data, target_column=target_column)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.success("Modèle entraîné")
        metrics_df = pd.DataFrame(
            [{k: round(v, 4) if v is not None else None for k, v in artifacts.metrics.items()}]
        )
        st.subheader("Métriques")
        st.dataframe(metrics_df, use_container_width=True)
        st.pyplot(plot_confusion_matrix(artifacts.y_test, artifacts.y_pred))
