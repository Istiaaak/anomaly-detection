import streamlit as st
import numpy as np
import time
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import plotly.express as px

from data.data import MVTecDataset, mvtec_classes, DEFAULT_SIZE
from utils.utils_app import tensor_to_img, load_patchcore_model
from logger import get_logger

# Initialise le logger
logger = get_logger("streamlit")

def main():
    st.set_page_config(page_title="PatchCore Live Dashboard", layout="wide")

    # Sidebar config
    cls          = st.sidebar.selectbox("Classe MVTec", mvtec_classes())
    backbone_key = st.sidebar.selectbox("Backbone", list(load_patchcore_model.__globals__['backbones'].keys()))
    f_coreset    = st.sidebar.slider("Fraction coreset", 0.0, 1.0, 0.1, 0.01)
    eps          = st.sidebar.slider("Epsilon coreset", 0.01, 1.0, 0.9, 0.01)
    k_nn         = st.sidebar.number_input("k-nearest", 1, 10, 3)
    use_cache    = st.sidebar.checkbox("Charger memory bank existante", True)

    # Initialisation une seule fois
    if 'initialized' not in st.session_state:
        model, train_scores = load_patchcore_model(
            cls, backbone_key, f_coreset, eps, k_nn, use_cache
        )
        st.session_state.model        = model
        st.session_state.train_scores = train_scores
        st.session_state.idx          = 0
        st.session_state.scores       = []
        st.session_state.running      = False
        st.session_state.initialized  = True

    model        = st.session_state.model
    train_scores = st.session_state.train_scores

    # Slider de seuil
    default_thresh = float(np.percentile(train_scores, 95))
    seuil = st.sidebar.slider(
        "Seuil d'anomalie",
        min_value=float(train_scores.min()),
        max_value=float(train_scores.max()),
        value=default_thresh
    )

    # Boutons Start/Stop
    if st.sidebar.button("Démarrer le Live"):
        st.session_state.running    = True
        st.session_state.start_time = time.time()
    if st.sidebar.button("Arrêter le Live"):
        st.session_state.running    = False

    # Placeholders
    cols            = st.columns(4)
    ph1, ph2, ph3, ph4 = [c.empty() for c in cols]
    col_img, col_map    = st.columns(2)
    ph_img              = col_img.empty()
    ph_map              = col_map.empty()
    c1, c2              = st.columns(2)
    ph_chart1, ph_chart2 = c1.empty(), c2.empty()
    ph_pie              = st.empty()

    # Boucle live
    if st.session_state.running:
        # Crée une instance du dataset test à la volée
        ds        = MVTecDataset(cls, size=DEFAULT_SIZE, vanilla=(backbone_key=="WideResNet50"))
        _, test_ds = ds.get_datasets()
        while st.session_state.idx < len(test_ds) and st.session_state.running:
            sample, mask, label = test_ds[st.session_state.idx]
            score, amap         = model.predict(sample.unsqueeze(0))
            score_val           = float(score.item())
            st.session_state.scores.append(score_val)
            st.session_state.idx += 1

            # Stats
            scores = st.session_state.scores
            mean, std = np.mean(scores), np.std(scores)
            elapsed   = time.time() - st.session_state.start_time

            # Log métier
            logger.info(
                "predicted_image",
                extra={
                    "index":     st.session_state.idx,
                    "class":     cls,
                    "score":     score_val,
                    "threshold": seuil,
                    "elapsed_s": round(elapsed, 2)
                }
            )

            # KPIs
            ph1.metric("Images traitées", f"{st.session_state.idx}/{len(test_ds)}")
            ph2.metric("Temps écoulé",    f"{elapsed:.2f}s")
            ph3.metric("Score moyen",     f"{mean:.3f}")
            ph4.metric("Seuil actif",     f"{seuil:.3f}")

            # Affiche l'image originale
            img_np = tensor_to_img(sample, backbone_key=="WideResNet50")
            ph_img.image(img_np, caption="Image originale", use_container_width=True)

            # Affiche la heatmap continue
            amap_np   = amap.squeeze().detach().cpu().numpy()
            amap_norm = (amap_np - amap_np.min())/(amap_np.max()-amap_np.min()+1e-8)
            fig = go.Figure()
            fig.add_trace(go.Image(z=img_np))
            fig.add_trace(go.Heatmap(z=amap_norm, colorscale='Jet', opacity=0.5))
            fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=350)
            ph_map.plotly_chart(fig, use_container_width=True)

            # Courbe d'évolution
            fig1 = go.Figure(data=[go.Scatter(x=list(range(len(scores))), y=scores, mode='lines+markers')])
            fig1.add_hline(y=seuil, line_color='red', annotation_text='Seuil')
            fig1.update_layout(title="Anomaly Score Live", xaxis_title='Index', yaxis_title='Score')
            ph_chart1.plotly_chart(fig1, use_container_width=True)

            # Histogramme
            fig2 = px.histogram(scores, nbins=20)
            fig2.add_vline(x=seuil, line_dash='dash', line_color='red')
            ph_chart2.plotly_chart(fig2, use_container_width=True)

            # Camembert
            normal    = sum(s < seuil for s in scores)
            anomalies = sum(s >= seuil for s in scores)
            fig3      = px.pie(values=[normal, anomalies], names=['Normal','Anomalie'])
            ph_pie.plotly_chart(fig3, use_container_width=True)

            time.sleep(0.5)

        st.session_state.running = False

if __name__ == '__main__':
    main()
