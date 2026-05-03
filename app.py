import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ---------- Configuration ----------
st.set_page_config(
    page_title="Wlab — Prédiction Cancer du Sein",
    page_icon="🎀",
    layout="wide",
)

MODELS_DIR = Path(__file__).parent / "models"
SCALERS_DIR = Path(__file__).parent / "scalers"

# ---------- Chargement du modèle ----------
@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "breast_cancer.pkl")
    scaler = joblib.load(SCALERS_DIR / "breast_cancer_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ---------- Définition des features ----------
FEATURES_GROUPS = {
    "Valeurs moyennes": [
        ("radius_mean", "Rayon moyen", 6.0, 30.0, 14.0),
        ("texture_mean", "Texture moyenne", 9.0, 40.0, 19.0),
        ("perimeter_mean", "Périmètre moyen", 40.0, 200.0, 92.0),
        ("area_mean", "Aire moyenne", 140.0, 2500.0, 655.0),
        ("smoothness_mean", "Lissage moyen", 0.05, 0.17, 0.10),
        ("compactness_mean", "Compacité moyenne", 0.02, 0.35, 0.10),
        ("concavity_mean", "Concavité moyenne", 0.0, 0.43, 0.09),
        ("concave_points_mean", "Points concaves moyens", 0.0, 0.20, 0.05),
        ("symmetry_mean", "Symétrie moyenne", 0.10, 0.30, 0.18),
        ("fractal_dimension_mean", "Dimension fractale moyenne", 0.05, 0.10, 0.06),
    ],
    "Erreurs standards": [
        ("radius_se", "Rayon (SE)", 0.10, 3.0, 0.40),
        ("texture_se", "Texture (SE)", 0.30, 5.0, 1.20),
        ("perimeter_se", "Périmètre (SE)", 0.70, 22.0, 2.85),
        ("area_se", "Aire (SE)", 6.0, 550.0, 40.0),
        ("smoothness_se", "Lissage (SE)", 0.001, 0.04, 0.007),
        ("compactness_se", "Compacité (SE)", 0.002, 0.14, 0.025),
        ("concavity_se", "Concavité (SE)", 0.0, 0.40, 0.032),
        ("concave_points_se", "Points concaves (SE)", 0.0, 0.05, 0.012),
        ("symmetry_se", "Symétrie (SE)", 0.008, 0.08, 0.020),
        ("fractal_dimension_se", "Dimension fractale (SE)", 0.0009, 0.03, 0.004),
    ],
    "Pires valeurs": [
        ("radius_worst", "Rayon max", 7.0, 40.0, 16.0),
        ("texture_worst", "Texture max", 12.0, 50.0, 25.0),
        ("perimeter_worst", "Périmètre max", 50.0, 260.0, 107.0),
        ("area_worst", "Aire max", 185.0, 4250.0, 880.0),
        ("smoothness_worst", "Lissage max", 0.07, 0.23, 0.13),
        ("compactness_worst", "Compacité max", 0.03, 1.06, 0.25),
        ("concavity_worst", "Concavité max", 0.0, 1.25, 0.27),
        ("concave_points_worst", "Points concaves max", 0.0, 0.30, 0.11),
        ("symmetry_worst", "Symétrie max", 0.15, 0.66, 0.29),
        ("fractal_dimension_worst", "Dimension fractale max", 0.055, 0.21, 0.083),
    ],
}

# Ordre canonique des 30 features (utilisé pour les prédictions)
FEATURE_ORDER = [k for group in FEATURES_GROUPS.values() for (k, *_) in group]

# ---------- Fonctions utilitaires ----------
def predict_one(values_dict):
    """Prédit pour un seul échantillon depuis un dict {feature: valeur}."""
    X = np.array([[values_dict[k] for k in FEATURE_ORDER]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None
    return prediction, proba

def predict_batch(df):
    """Prédit pour un DataFrame entier (les colonnes doivent matcher FEATURE_ORDER)."""
    X = df[FEATURE_ORDER].values
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None
    return predictions, probas

def show_result(prediction, proba):
    """Affiche le résultat d'une prédiction avec mise en forme."""
    if prediction == 0:
        st.success("✅ Prédiction : **Tumeur bénigne**")
    else:
        st.error("⚠️ Prédiction : **Tumeur maligne**")
    if proba is not None:
        col1, col2 = st.columns(2)
        col1.metric("Probabilité bénin", f"{proba[0]*100:.1f} %")
        col2.metric("Probabilité malin", f"{proba[1]*100:.1f} %")

# ---------- Header ----------
col_logo, col_title = st.columns([1, 4])
with col_logo:
    logo_path = Path(__file__).parent / "img" / "logo_wlab4_2.svg"
    if logo_path.exists():
        st.image(str(logo_path), width=120)
with col_title:
    st.title("Prédiction du Cancer du Sein")
    st.caption(
        "Modèle Random Forest · Classification binaire (bénin / malin) à partir de "
        "30 caractéristiques cytologiques mesurées sur biopsie"
    )

st.divider()

# ---------- Avertissement ----------
st.warning(
    "⚠️ Cet outil est un projet pédagogique. Il ne remplace en aucun cas "
    "un diagnostic médical professionnel."
)

# ---------- Choix du mode de saisie ----------
mode_tabs = st.tabs(["📝 Saisie manuelle", "📂 Import CSV"])

# ===================================================================
# TAB 1 — SAISIE MANUELLE
# ===================================================================
with mode_tabs[0]:
    st.subheader("Caractéristiques de la tumeur")
    st.markdown("Saisissez les valeurs mesurées sur la biopsie :")

    values = {}

    inner_tabs = st.tabs(list(FEATURES_GROUPS.keys()))
    for tab, (group_name, features) in zip(inner_tabs, FEATURES_GROUPS.items()):
        with tab:
            cols = st.columns(2)
            for i, (key, label, vmin, vmax, vdefault) in enumerate(features):
                with cols[i % 2]:
                    values[key] = st.number_input(
                        label,
                        min_value=float(vmin),
                        max_value=float(vmax),
                        value=float(vdefault),
                        step=0.001,
                        format="%.4f",
                        key=f"manual_{key}",
                    )

    st.divider()

    if st.button("🔍 Lancer la prédiction", type="primary", use_container_width=True, key="predict_manual"):
        prediction, proba = predict_one(values)
        st.subheader("Résultat")
        show_result(prediction, proba)

# ===================================================================
# TAB 2 — IMPORT CSV
# ===================================================================
with mode_tabs[1]:
    st.subheader("Import d'un fichier CSV")
    st.markdown(
        "Importez un fichier CSV contenant les 30 caractéristiques cytologiques. "
        "Une ligne par patient, l'application prédit pour chaque ligne."
    )

    # Aide : format attendu
    with st.expander("📋 Format attendu du CSV"):
        st.markdown(
            "Le fichier doit contenir **30 colonnes** correspondant aux features "
            "(la casse n'est pas importante). Les noms attendus sont :"
        )
        cols_help = st.columns(3)
        for i, (group_name, features) in enumerate(FEATURES_GROUPS.items()):
            with cols_help[i]:
                st.markdown(f"**{group_name}**")
                for key, *_ in features:
                    st.markdown(f"- `{key}`")

        # Bouton pour télécharger un template
        template_df = pd.DataFrame(
            [[v[4] for group in FEATURES_GROUPS.values() for v in group]],
            columns=[k for group in FEATURES_GROUPS.values() for (k, *_) in group],
        )
        st.download_button(
            "📥 Télécharger un modèle CSV",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="modele_cancer_sein.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Impossible de lire le fichier : {e}")
            st.stop()

        # Normaliser les noms de colonnes (lowercase) pour matcher feature_order
        df.columns = [c.strip().lower() for c in df.columns]

        # Vérifier que toutes les features attendues sont présentes
        missing = [f for f in FEATURE_ORDER if f not in df.columns]
        if missing:
            st.error(
                f"❌ Colonnes manquantes dans le CSV ({len(missing)}) :\n\n"
                + ", ".join(f"`{m}`" for m in missing)
            )
            st.stop()

        # Vérifier l'absence de valeurs manquantes
        if df[FEATURE_ORDER].isnull().any().any():
            st.error("❌ Le fichier contient des valeurs manquantes. Veuillez les remplir avant l'import.")
            st.dataframe(df[df[FEATURE_ORDER].isnull().any(axis=1)])
            st.stop()

        # Aperçu
        st.success(f"✅ Fichier chargé : **{len(df)} ligne(s)** · {len(df.columns)} colonne(s)")
        with st.expander("👁️ Aperçu des données", expanded=True):
            st.dataframe(df, use_container_width=True)

        if st.button("🔍 Lancer les prédictions", type="primary", use_container_width=True, key="predict_csv"):
            predictions, probas = predict_batch(df)

            # Construire le DataFrame de résultats
            results_df = df[FEATURE_ORDER].copy()
            results_df.insert(0, "Prédiction", ["Bénin" if p == 0 else "Malin" for p in predictions])
            if probas is not None:
                results_df.insert(1, "Proba bénin (%)", (probas[:, 0] * 100).round(1))
                results_df.insert(2, "Proba malin (%)", (probas[:, 1] * 100).round(1))

            st.subheader("Résultats")

            # Si une seule ligne, afficher comme une prédiction unique
            if len(df) == 1:
                show_result(predictions[0], probas[0] if probas is not None else None)
            else:
                # Compteur global
                n_benign = int((predictions == 0).sum())
                n_malign = int((predictions == 1).sum())
                col1, col2, col3 = st.columns(3)
                col1.metric("Total patients", len(df))
                col2.metric("Bénin", n_benign)
                col3.metric("Malin", n_malign)

            # Tableau complet (toujours affiché)
            st.dataframe(
                results_df,
                use_container_width=True,
                column_config={
                    "Prédiction": st.column_config.TextColumn(width="small"),
                    "Proba bénin (%)": st.column_config.ProgressColumn(
                        format="%.1f %%", min_value=0, max_value=100
                    ),
                    "Proba malin (%)": st.column_config.ProgressColumn(
                        format="%.1f %%", min_value=0, max_value=100
                    ),
                },
            )

            # Téléchargement des résultats
            st.download_button(
                "📥 Télécharger les résultats (CSV)",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="resultats_predictions.csv",
                mime="text/csv",
            )

# ---------- Footer ----------
st.divider()
st.caption(
    "Projet Wlab — Luana de Oliveira · "
    "[GitHub](https://github.com/LuaGeo) · "
    "[Portfolio](https://luanadeoliveira.netlify.app/)"
)