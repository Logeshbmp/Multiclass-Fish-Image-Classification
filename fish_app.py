import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Multiclass Fish Classifier", page_icon="🐟", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Best Model": "fish_best_model.h5",
    "CNN (From Scratch)": "cnn_fish_model.h5",
    "MobileNet Fine-Tuned": "MobileNet_FineTuned_fish_model.h5",
    "InceptionV3 Fine-Tuned": "InceptionV3_FineTuned_fish_model.h5",
    "VGG16 Fine-Tuned": "VGG16_FineTuned_fish_model.h5",
    "ResNet50 Fine-Tuned": "ResNet50_FineTuned_fish_model.h5",
    "EfficientNetB0 Fine-Tuned": "EfficientNetB0_FineTuned_fish_model.h5",
}


def prettify_label(raw_label: str) -> str:
    return raw_label.replace("_", " ").replace("sea_food", "seafood").title()


def load_model(model_path: Path):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path, compile=False)


def get_active_model(model_name: str, model_path: Path):
    import tensorflow as tf

    active_name = st.session_state.get("active_model_name")
    if active_name != model_name:
        # Keep only one model in memory to reduce crash risk on low-RAM systems.
        tf.keras.backend.clear_session()
        gc.collect()
        st.session_state["active_model"] = load_model(model_path)
        st.session_state["active_model_name"] = model_name
    return st.session_state["active_model"]


@st.cache_data(show_spinner=False)
def load_labels(labels_path: Path):
    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    return [str(x) for x in labels]


@st.cache_data(show_spinner=False)
def load_comparison(csv_path: Path):
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB").resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, image_array, labels):
    probs = model.predict(image_array, verbose=0)[0]
    n = min(len(labels), len(probs))
    labels = labels[:n]
    probs = probs[:n]

    top_idx = int(np.argmax(probs))
    pred_label = labels[top_idx]
    pred_conf = float(probs[top_idx])

    ranked = np.argsort(probs)[::-1]
    top_n = min(5, len(ranked))
    top_rows = [(labels[i], float(probs[i])) for i in ranked[:top_n]]
    return pred_label, pred_conf, top_rows


st.title("🐟 Multiclass Fish Image Classification")
st.caption("Upload a fish image to predict species with confidence scores.")

missing_models = [name for name, f in MODEL_FILES.items() if not (ROOT / f).exists()]
if missing_models:
    st.warning(f"Some model files are missing: {', '.join(missing_models)}")

available_models = {k: v for k, v in MODEL_FILES.items() if (ROOT / v).exists()}
if not available_models:
    st.error("No model files were found in the project folder.")
    st.stop()

labels_path = ROOT / "class_labels.json"
if not labels_path.exists():
    st.error("class_labels.json is required but was not found.")
    st.stop()

labels = load_labels(labels_path)

left, right = st.columns([1, 1])
with left:
    model_name = st.selectbox("Select model", list(available_models.keys()), index=0)
    model_size_mb = (ROOT / available_models[model_name]).stat().st_size / (1024 * 1024)
    if model_size_mb > 120:
        st.warning(f"Selected model is large ({model_size_mb:.0f} MB). First prediction can take longer.")
    uploaded_file = st.file_uploader("Upload fish image", type=["jpg", "jpeg", "png", "webp"])

with right:
    st.write("**Available classes**")
    st.write(", ".join(prettify_label(x) for x in labels))

if uploaded_file is not None:
    if uploaded_file.size and uploaded_file.size > 10 * 1024 * 1024:
        st.error("Image size is too large (>10 MB). Please upload a smaller image.")
        st.stop()

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width="stretch")

    with st.spinner("Loading model and predicting..."):
        model = get_active_model(model_name, ROOT / available_models[model_name])
        arr = preprocess_image(image)
        pred_label, pred_conf, top_rows = predict(model, arr, labels)

    st.success(f"Prediction: **{prettify_label(pred_label)}**")
    st.metric("Confidence", f"{pred_conf * 100:.2f}%")

    top_df = pd.DataFrame(top_rows, columns=["Class", "Probability"])
    top_df["Class"] = top_df["Class"].map(prettify_label)

    fig = px.bar(
        top_df,
        x="Probability",
        y="Class",
        orientation="h",
        text=top_df["Probability"].map(lambda x: f"{x * 100:.1f}%"),
        title="Top Predictions",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Model Performance Comparison")
comparison_df = load_comparison(ROOT / "model_comparison.csv")
if not comparison_df.empty and {"Model", "TestAccuracy"}.issubset(comparison_df.columns):
    comparison_df = comparison_df.sort_values("TestAccuracy", ascending=False).reset_index(drop=True)
    comparison_df["TestAccuracy"] = comparison_df["TestAccuracy"].astype(float)
    st.dataframe(comparison_df, width="stretch")

    comp_fig = px.bar(
        comparison_df,
        x="Model",
        y="TestAccuracy",
        text=comparison_df["TestAccuracy"].map(lambda x: f"{x:.4f}"),
        title="Test Accuracy by Model",
    )
    comp_fig.update_yaxes(range=[0, 1])
    st.plotly_chart(comp_fig, width="stretch")
else:
    st.info("model_comparison.csv not found or missing required columns: Model, TestAccuracy.")
