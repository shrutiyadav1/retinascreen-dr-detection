import numpy as np
import pandas as pd
import cv2
import streamlit as st
import tensorflow as tf

# ---------------- Page setup ----------------
st.set_page_config(page_title="RetinaScreen", page_icon="👁️", layout="wide")

MODEL_PATH = "effnet_finetuned_final.keras"
IMG_SIZE = 224
REFERABLE_THRESHOLD = 0.49   # chosen on the validation set

GRADES = {
    0: ("No DR", "No signs of diabetic retinopathy detected.", "#2e7d32"),
    1: ("Mild", "Very small changes (microaneurysms).", "#9e9d24"),
    2: ("Moderate", "More blood vessels are affected.", "#ef6c00"),
    3: ("Severe", "Many blood vessels are blocked.", "#d84315"),
    4: ("Proliferative", "Most advanced stage, new fragile vessels form.", "#b71c1c"),
}

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.result-card {padding: 1.2rem 1.5rem; border-radius: 14px; color: white; margin-bottom: 1rem;}
.result-card h2 {margin: 0; font-size: 2rem;}
.result-card p {margin: 0.3rem 0 0 0; opacity: 0.95;}
.alert-box {padding: 0.9rem 1.2rem; border-radius: 10px; font-weight: 600; margin-bottom: 1rem;}
.alert-red {background: #4a1414; border-left: 6px solid #e53935; color: #ffcdd2;}
.alert-green {background: #14391c; border-left: 6px solid #43a047; color: #c8e6c9;}
.small-note {font-size: 0.85rem; opacity: 0.7;}
</style>
""", unsafe_allow_html=True)

# ---------------- Model loading ----------------
@st.cache_resource
def load_everything():
    model = tf.keras.models.load_model(MODEL_PATH)
    base = model.get_layer("efficientnetb0")
    last_conv = base.get_layer("top_activation")
    conv_model = tf.keras.Model(base.input, last_conv.output)

    clf_in = tf.keras.Input(shape=last_conv.output.shape[1:])
    x = clf_in
    for layer in model.layers[-3:]:
        x = layer(x)
    clf_model = tf.keras.Model(clf_in, x)
    return model, conv_model, clf_model

model, conv_model, clf_model = load_everything()

# ---------------- Helper functions ----------------
def fundus_stats(rgb_img):
    small = cv2.resize(rgb_img, (128, 128)).astype("float32")
    h, w = small.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)

    center = small[dist < h * 0.40]      # inner circle
    outside = small[dist > h * 0.55]     # outer area (corners/edges)

    r, g, b = center[:, 0].mean(), center[:, 1].mean(), center[:, 2].mean()
    outside_brightness = outside.mean()
    return r, g, b, outside_brightness

def looks_like_fundus(rgb_img):
    r, g, b, outside = fundus_stats(rgb_img)
    red_dominant = r > g and r > 1.2 * b
    dark_outside = outside < 90
    return red_dominant and dark_outside

def preprocess(rgb_img):
    img = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE / 30), -4, 128)
    return img.astype("float32")

def gradcam(img_array, class_index):
    t = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out = conv_model(t, training=False)
        tape.watch(conv_out)
        preds = clf_model(conv_out, training=False)
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    return cam / (cam.max() + 1e-8)

def make_overlay(rgb_img, cam):
    original = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return original, np.uint8(0.6 * original + 0.4 * heat)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("👁️ RetinaScreen")
    st.caption("Diabetic Retinopathy screening aid")
    st.markdown("---")
    st.markdown("**Model:** EfficientNetB0 (transfer learning + fine-tuning)")
    st.markdown("**Dataset:** APTOS 2019")
    st.markdown("**Quadratic Weighted Kappa:** 0.862")
    st.markdown("**Referable DR sensitivity:** 85.7%")
    st.markdown("**Referable DR specificity:** 93.0%")
    st.markdown("---")
    st.warning("This is not a medical diagnosis. It is a learning project and "
               "screening aid only. Please consult an eye doctor for a real "
               "examination.")

# ---------------- Main page ----------------
st.title("Diabetic Retinopathy Detection")
st.write("Upload a retinal (fundus) photograph. The model predicts the DR grade "
         "and shows a heatmap of where it focused.")

uploaded = st.file_uploader("Upload a retina image (fundus photo only)",
                            type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Upload an image above to get started.")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if not looks_like_fundus(rgb):
    st.error("This does not look like a retinal (fundus) image. Please upload a "
             "fundus photograph, not a phone photo of an eye or face.")
    st.stop()

with st.spinner("Analysing image..."):
    processed = preprocess(rgb)
    probs = model.predict(processed[np.newaxis, ...], verbose=0)[0]
    pred = int(np.argmax(probs))
    p_referable = float(probs[2:].sum())
    cam = gradcam(processed, pred)
    original, overlay = make_overlay(rgb, cam)

name, meaning, color = GRADES[pred]

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Predicted result")
    st.markdown(
        f"""<div class="result-card" style="background:{color};">
        <h2>Grade {pred}: {name}</h2>
        <p>{meaning}</p>
        <p class="small-note" style="color:white;">Confidence: {probs[pred]*100:.1f}%</p>
        </div>""",
        unsafe_allow_html=True,
    )

    if p_referable >= REFERABLE_THRESHOLD:
        st.markdown(
            f'<div class="alert-box alert-red">⚠️ Referable DR: please consult an '
            f'eye doctor (score {p_referable*100:.1f}%)</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="alert-box alert-green">✅ Low likelihood of referable DR '
            f'(score {p_referable*100:.1f}%). Regular check-ups are still '
            f'recommended.</div>',
            unsafe_allow_html=True)

    st.subheader("Grade probabilities")
    chart_df = pd.DataFrame(
        {"Probability (%)": probs * 100},
        index=[f"{i}: {GRADES[i][0]}" for i in range(5)],
    )
    st.bar_chart(chart_df)

with right:
    st.subheader("Grad-CAM: where the model looked")
    tab1, tab2 = st.tabs(["Overlay", "Original vs Processed"])
    with tab1:
        st.image(overlay, caption="Red = high attention, blue = low attention",
                 use_container_width=True)
    with tab2:
        c1, c2 = st.columns(2)
        c1.image(original, caption="Original", use_container_width=True)
        c2.image(np.clip(processed, 0, 255).astype("uint8"),
                 caption="Ben Graham processed", use_container_width=True)
    st.caption("The heatmap is an approximation of the model's focus, "
               "not proof of disease.")

st.markdown("---")
st.caption("⚠️ Not a medical diagnosis. Screening aid only. Built as a learning project.")