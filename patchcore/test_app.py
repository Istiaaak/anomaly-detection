
import streamlit as st
import torch
import numpy as np
import time
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go
import plotly.express as px

from data.data import MVTecDataset, mvtec_classes, DEFAULT_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD
from model.patch_core import PatchCore
from utils.utils import backbones, dataset_scale_factor

# Utility to denormalize tensor
def tensor_to_img(x: torch.Tensor, vanilla: bool):
    x = x.clone().cpu()
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if vanilla else (CLIP_MEAN, CLIP_STD)
    for c in range(x.shape[0]):
        x[c] = x[c] * std[c] + mean[c]
    return x.clip(0.0, 1.0).permute(1, 2, 0).numpy()

# Main App
def main():
    st.set_page_config(page_title="PatchCore Live Dashboard", layout="wide")

    # Sidebar config
    cls = st.sidebar.selectbox("Classe MVTec", mvtec_classes())
    backbone_key = st.sidebar.selectbox("Backbone", list(backbones.keys()))
    vanilla = (backbone_key == "WideResNet50")
    size_map = {'WideResNet50': DEFAULT_SIZE, 'ResNet50':224, 'ResNet50-4':288, 'ResNet50-16':384, 'ResNet101':224}
    size = size_map.get(backbone_key, DEFAULT_SIZE)

    # Cache & hyperparameters
    cache_root = Path("./patchcore_cache/memory_bank")
    cache_root.mkdir(parents=True, exist_ok=True)
    use_cache = st.sidebar.checkbox("Charger memory bank existante", True)
    f_coreset = st.sidebar.slider("Fraction coreset", 0.0, 1.0, 0.1, 0.01)
    eps = st.sidebar.slider("Epsilon coreset", 0.01, 1.0, 0.9, 0.01)
    k_nn = st.sidebar.number_input("k-nearest", 1, 10, 3)

    # One-time initialization
    if 'initialized' not in st.session_state:
        # Dataset loading
        ds = MVTecDataset(cls, size=size, vanilla=vanilla)
        train_ds, test_ds = ds.get_datasets()
        st.session_state.train_ds = train_ds
        st.session_state.test_ds = test_ds
        st.session_state.ds_len = len(test_ds)
        
        # Model instantiation
        model = PatchCore(f_coreset=f_coreset, eps_coreset=eps, k_nearest=k_nn,
                          vanilla=vanilla, backbone=backbones[backbone_key], image_size=size)
        cache_file = cache_root / f"{cls}_{backbone_key}_f{f_coreset:.3f}.pth"
        if use_cache and cache_file.exists():
            mb = torch.load(cache_file)
            model.memory_bank = mb if isinstance(mb, torch.Tensor) else torch.cat(mb,0)
            # init avg & resize
            model.avg = torch.nn.AvgPool2d(3,stride=1)
            batch,_ = next(iter(DataLoader(train_ds, batch_size=1)))
            _ = model.forward(batch)
            fmap_size = model.features[0].shape[-2]
            model.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        else:
            model.fit(DataLoader(train_ds, batch_size=1), scale=dataset_scale_factor[backbone_key])
            mb = model.memory_bank.cpu() if isinstance(model.memory_bank,torch.Tensor) else torch.cat(model.memory_bank,0).cpu()
            torch.save(mb, cache_file)

                # Compute calibration scores on good images
        train_dl = DataLoader(train_ds, batch_size=1)
        train_scores = []
        # Unpack only image and label for train dataset
        for x, _ in train_dl:
            s, _ = model.predict(x)
            train_scores.append(s.item())
        train_scores = np.array(train_scores)
        default_thresh = float(np.percentile(train_scores, 95))
        st.session_state.train_scores = train_scores
        st.session_state.default_thresh = default_thresh

        # Init live variables
        st.session_state.model = model
        st.session_state.idx = 0
        st.session_state.scores = []
        st.session_state.running = False
        st.session_state.initialized = True

    # Threshold slider
    train_scores = st.session_state.train_scores
    seuil = st.sidebar.slider(
        "Seuil d'anomalie", 
        min_value=float(train_scores.min()), 
        max_value=float(train_scores.max()), 
        value=st.session_state.default_thresh
    )

    # Control buttons
    if st.sidebar.button("Démarrer le Live"):
        st.session_state.running = True
        st.session_state.start_time = time.time()
    if st.sidebar.button("Arrêter le Live"):
        st.session_state.running = False

    # Prepare layout
    cols = st.columns(4)
    ph1, ph2, ph3, ph4 = [c.empty() for c in cols]
    col_img, col_map = st.columns(2)
    ph_img = col_img.empty()
    ph_map = col_map.empty()
    chart1_col, chart2_col = st.columns(2)
    ph_chart1 = chart1_col.empty()
    ph_chart2 = chart2_col.empty()
    ph_pie = st.empty()

    # Live loop
    if st.session_state.running:
        model = st.session_state.model
        test_ds = st.session_state.test_ds
        while st.session_state.idx < st.session_state.ds_len and st.session_state.running:
            sample, _, _ = test_ds[st.session_state.idx]
            score, amap = model.predict(sample.unsqueeze(0))
            st.session_state.scores.append(score.item())
            st.session_state.idx += 1

            # Stats
            scores = st.session_state.scores
            mean, std = np.mean(scores), np.std(scores)
            elapsed = time.time() - st.session_state.start_time

            # Update KPI
            ph1.metric("Images traitées", f"{st.session_state.idx}/{st.session_state.ds_len}")
            ph2.metric("Temps écoulé", f"{elapsed:.2f}s")
            ph3.metric("Score moyen", f"{mean:.3f}")
            ph4.metric("Seuil actif", f"{seuil:.3f}")

            # Show original image
            img_np = tensor_to_img(sample, vanilla)
            ph_img.image(img_np, use_container_width=True)

                        # Show continuous anomaly map overlay
            amap_np = amap.squeeze().detach().cpu().numpy()
            # Normalize map between 0 and 1
            amap_norm = (amap_np - amap_np.min()) / (amap_np.max() - amap_np.min() + 1e-8)
            fig = go.Figure()
            fig.add_trace(go.Image(z=img_np))
            fig.add_trace(go.Heatmap(z=amap_norm, colorscale='Jet', opacity=0.5, zmid=0.5, zmin=0, zmax=1))
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), title_text="Carte d'anomalie", height=350)
            ph_map.plotly_chart(fig, use_container_width=True)

            # Evolution chart
            fig1 = go.Figure(data=[go.Scatter(x=list(range(len(scores))), y=scores, mode='lines+markers')])
            fig1.add_hline(y=seuil, line_color='red', annotation_text='Seuil')
            fig1.update_layout(title="Anomaly Score Live", xaxis_title='Index', yaxis_title='Score')
            ph_chart1.plotly_chart(fig1, use_container_width=True)

            # Histogram with threshold line
            fig2 = px.histogram(scores, nbins=20)
            fig2.add_vline(x=seuil, line_dash='dash', line_color='red')
            ph_chart2.plotly_chart(fig2, use_container_width=True)

            # Distribution pie
            normal = sum(s < seuil for s in scores)
            anomalies = sum(s >= seuil for s in scores)
            fig3 = px.pie(values=[normal, anomalies], names=['Normal','Anomalie'])
            ph_pie.plotly_chart(fig3, use_container_width=True)

            time.sleep(0.5)
        st.session_state.running = False

if __name__ == '__main__':
    main()
