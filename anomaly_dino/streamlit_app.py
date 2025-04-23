import streamlit as st
import numpy as np
import cv2
import faiss
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
from matplotlib.colors import LinearSegmentedColormap
import base64

# IMPORTS de vos modules anomaly_dino
# Adaptez ces imports selon votre organisation
from src.backbones import get_model
from PFE.anomaly_dino.detection import augment_image
from PFE.anomaly_dino.utils_dino import get_dataset_info

# Configuration de la page Streamlit
st.set_page_config(page_title="Détection d'Anomalies Few-Shot", layout="wide")
st.title("Détection d'anomalies Few-Shot")

###############################################################################
# Chargement du Modèle
###############################################################################

def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    st.info("Chargement du modèle DINOv2 (cela peut prendre quelques minutes)...")
    model = get_model("dinov2_vits14", device, smaller_edge_size=448)
    # Le modèle pré-entraîné est encapsulé dans model.model, donc on passe en mode inférence sur cet objet
    model.model.eval()
    st.success("Modèle chargé.")
    return model

model = load_model()

# Pour cet exemple nous utilisons une seule classe personnalisée.
# Comme nous n'utilisons pas le dataset local, nous allons définir des paramètres par défaut.
object_name = "custom"
# Par défaut, nous activons le masquage et désactivons la rotation (vous pouvez ajuster ces paramètres)
masking_default = {object_name: True}
rotation_default = {object_name: False}
# La fonction get_dataset_info peut être utilisée pour récupérer d'autres paramètres si nécessaire,
# ici nous l'appelons juste pour illustration.
_, _, _, _ = get_dataset_info("MVTec", "informed")

###############################################################################
# Fonctions pour construire la Memory Bank et détecter l'anomalie
###############################################################################

def build_memory_bank_from_upload(uploaded_files):
    """
    Construit une memory bank à partir des images de référence uploadées par l'utilisateur.
    Chaque image est préparée, ses caractéristiques sont extraites et un masquage est appliqué
    pour ne conserver que les patches pertinents.
    Renvoie l'index FAISS et la taille de grille (grid_size) utilisée lors de l'extraction.
    """
    features_ref = []
    grid_size_ref = None

    if not uploaded_files:
        st.error("Aucune image de référence uploadée.")
        return None, None

    for file in uploaded_files:
        st.info(f"Traitement de l'image de référence : {file.name}")
        try:
            img = Image.open(file).convert("RGB")
        except Exception as e:
            st.error(f"Erreur lors de l'ouverture de l'image {file.name} : {e}")
            continue

        # Conversion de l'image pour OpenCV
        img_cv = np.array(img)
        # Si l'augmentation par rotation est activée, vous pouvez la gérer ici
        if rotation_default[object_name]:
            # Par exemple, générer des rotations; ici nous utilisons l'image originale pour simplifier
            imgs_versions = [img_cv]  # Vous pouvez appeler augment_image(img_cv) si besoin
        else:
            imgs_versions = [img_cv]

        for img_version in imgs_versions:
            st.write("Préparation de l'image...")
            image_tensor, grid_size = model.prepare_image(img_version)
            st.write("Extraction des caractéristiques...")
            feats = model.extract_features(image_tensor)
            st.write("Application du masquage...")
            mask = model.compute_background_mask_from_image(img_version, threshold=10, masking_type=masking_default[object_name])
            features_ref.append(feats[mask])
            grid_size_ref = grid_size  # On suppose que toutes les images ont la même configuration

    if features_ref:
        st.write("Création de l'index FAISS...")
        features_ref = np.concatenate(features_ref, axis=0)
        index = faiss.IndexFlatL2(features_ref.shape[1])
        faiss.normalize_L2(features_ref)
        index.add(features_ref)
        st.success("Memory bank construite.")
        return index, grid_size_ref
    else:
        st.error("Erreur : aucune caractéristique extraite.")
        return None, None

def detect_anomaly(test_image, memory_index):
    """
    Applique la détection d'anomalie sur l'image de test en utilisant la memory bank construite.
    Renvoie le score d'anomalie, une carte d'anomalie et l'image test sous forme d'array.
    """
    # Conversion de l'image PIL en tableau numpy
    image_cv = np.array(test_image)
    st.write("Préparation de l'image test...")
    image_tensor, grid_size_test = model.prepare_image(image_cv)
    st.write("Extraction des caractéristiques de l'image test...")
    feats_test = model.extract_features(image_tensor)
    faiss.normalize_L2(feats_test)
    
    # Recherche des plus proches voisins (k=1)
    distances, _ = memory_index.search(feats_test, k=1)
    distances = distances / 2  # Transformation pour obtenir l'équivalent de 1 - similarité cosinus
    
    # Application du masquage sur l'image test
    st.write("Application du masquage sur l'image test...")
    mask_test = model.compute_background_mask_from_image(image_cv, threshold=10, masking_type=masking_default[object_name])
    distances[~mask_test] = 0.0
    
    # Calcul du score d'anomalie : moyenne des 1% des distances les plus élevées
    all_dist = distances.flatten()
    sorted_dist = np.sort(all_dist)[::-1]
    top_count = max(1, int(len(sorted_dist) * 0.01))
    anomaly_score = float(np.mean(sorted_dist[:top_count]))
    
    # Remodelage pour générer la carte d'anomalie
    distance_map = distances.reshape(grid_size_test)
    anomaly_map = cv2.resize(distance_map, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_LINEAR)
    return anomaly_score, anomaly_map, image_cv

###############################################################################
# Interface de l'application avec Streamlit
###############################################################################

# Nous utilisons deux onglets : l'un pour construire la memory bank et l'autre pour l'inférence
tab_memory, tab_infer = st.tabs(["Memory Bank", "Inférence"])

with tab_memory:
    st.header("Construction de la memory bank")
    st.write("Uploader vos images de référence (normales) afin de construire votre memory bank.")
    uploaded_refs = st.file_uploader("Images de Référence", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="refs")
    
    if st.button("Construire la Memory Bank"):
        if uploaded_refs is not None and len(uploaded_refs) > 0:
            with st.spinner("Construction de la memory bank en cours..."):
                memory_index, grid_size_ref = build_memory_bank_from_upload(uploaded_refs)
                if memory_index is not None:
                    st.session_state["memory_index"] = memory_index
                    st.session_state["grid_size_ref"] = grid_size_ref
                else:
                    st.error("La memory bank n'a pas pu être construite.")
        else:
            st.error("Veuillez uploader au moins une image de référence.")

with tab_infer:
    st.header("Inférence sur une Image de test")
    st.write("Uploader une image de test pour effectuer la détection d'anomalie.")
    uploaded_test = st.file_uploader("Image de test", type=["png", "jpg", "jpeg"], key="test")
    
    if uploaded_test is not None:
        try:
            test_img = Image.open(uploaded_test).convert("RGB")
            st.image(test_img, caption="Image de test", use_column_width=True, width=200)
        except Exception as e:
            st.error(f"Erreur lors de l'ouverture de l'image de test : {e}")
        
        if "memory_index" in st.session_state:
            with st.spinner("Détection d'anomalies en cours..."):
                score, anomaly_map, input_img = detect_anomaly(test_img, st.session_state["memory_index"])
            st.write(f"Score d'anomalie : {score:.3f}")
            # Création d'une colormap personnalisée pour visualiser la carte d'anomalie
            neon_violet = (0.5, 0.1, 0.5, 0.4)
            neon_yellow = (0.8, 1.0, 0.02, 0.7)
            colors = [(1.0, 1.0, 1.0, 0.0), neon_violet, neon_yellow]
            cmap = LinearSegmentedColormap.from_list("AnomalyMap", colors, N=256)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(input_img)
            ax.imshow(anomaly_map, cmap=cmap, alpha=0.7)
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("Veuillez d'abord construire la memory bank en uploadant vos images de référence.")

