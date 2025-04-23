# necessary imports
import matplotlib.pyplot as plt
import cv2
import faiss
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from PFE.anomaly_dino.detection import augment_image
from PFE.anomaly_dino.utils_dino import resize_mask_img, get_dataset_info
from src.backbones import get_model
import os
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io
import gradio as gr
# -------------------------------
# Configuration et chargement du modèle
# -------------------------------

# Répertoire du dataset MVTEC AD
data_root = "./data"
mvtec_path = os.path.join(data_root, "mvtec ad")
object_name = "hazelnut"  # Exemple : "hazelnut". Vous pouvez modifier selon l’objet étudié.

# Récupération des infos du dataset (liste d'objets, type d'anomalie, paramètres masking/rotation)
objects, object_anomalies, masking_default, rotation_default = get_dataset_info("MVTec", "informed")

# Chargement du modèle DINOv2 via votre wrapper
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = get_model("dinov2_vits14", device, smaller_edge_size=448)
# Comme le wrapper ne possède pas directement eval(), on appelle eval() sur le modèle encapsulé :
model.model.eval()

################################################################################
# Construction de la memory bank few-shot à partir d'une ou quelques images "good"
################################################################################
def build_memory_bank(model, object_name="hazelnut", ref_files=["000.png"], mvtec_path=mvtec_path):
    features_ref = []
    ref_folder = os.path.join(mvtec_path, object_name, "train", "good")
    for fname in ref_files:
        ref_path = os.path.join(ref_folder, fname)
        # Charger l'image de référence en RGB
        image_ref = cv2.cvtColor(cv2.imread(ref_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        # Appliquer une augmentation par rotation si activée pour cet objet
        if rotation_default[object_name]:
            imgs = augment_image(image_ref)
        else:
            imgs = [image_ref]
        for img in imgs:
            # Préparation de l'image et extraction des features
            tensor_ref, grid_size_ref = model.prepare_image(img)
            feats = model.extract_features(tensor_ref)
            # Masquage : on utilise ici compute_background_mask_from_image inspiré du notebook
            mask = model.compute_background_mask_from_image(img, threshold=10, masking_type=masking_default[object_name])
            # Garder uniquement les features des patches retenus
            features_ref.append(feats[mask])
    features_ref = np.concatenate(features_ref, axis=0)
    index = faiss.IndexFlatL2(features_ref.shape[1])
    faiss.normalize_L2(features_ref)
    index.add(features_ref)
    return index, grid_size_ref

# Pour few-shot, nous utilisons une image de référence (peut être étendue à plusieurs)
ref_files = [f"{i:03}.png" for i in range(100)]

memory_index, grid_size_ref = build_memory_bank(model, object_name=object_name, ref_files=ref_files)

print("Memory bank construite (few-shot) avec succès.")

################################################################################
# Fonction d'inférence d'anomalies inspirée du notebook
################################################################################
def detect_anomaly(image, memory_index, model, grid_size_ref, object_name="hazelnut", k=1):
    """
    Pour une image test (format numpy issu d'une image PIL), prépare l'image,
    extrait les features, recherche kNN dans la memory bank et calcule le score d'anomalie.
    """
    # Convertir l'image PIL en array (si ce n'est déjà fait)
    image_cv = np.array(image)
    
    # Extraction des features de l'image test
    tensor_test, grid_size_test = model.prepare_image(image_cv)
    feats_test = model.extract_features(tensor_test)
    faiss.normalize_L2(feats_test)
    
    # Recherche kNN : pour chaque patch, trouver le plus proche dans la bank
    distances, _ = memory_index.search(feats_test, k=k)
    distances = distances / 2  # Correspond à transformer la distance L2 normalisée en équivalent "1 - cosinus"
    
    # Optionnel : masque sur l'image test
    mask_test = model.compute_background_mask_from_image(image_cv, threshold=10, masking_type=masking_default[object_name])
    distances[~mask_test] = 0.0
    
    # Calcul du score d'anomalie : moyenne des 1 % des distances les plus élevées
    all_dist = distances.flatten()
    sorted_dist = np.sort(all_dist)[::-1]
    top_count = max(1, int(len(sorted_dist) * 0.01))
    anomaly_score = np.mean(sorted_dist[:top_count])
    
    # Remodeler la carte d'anomalie et la redimensionner à la taille de l'image test
    distance_map = distances.reshape(grid_size_test)
    anomaly_map = cv2.resize(distance_map, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_LINEAR)
    return anomaly_score, anomaly_map, image_cv




################################################################################
# Fonction d'inférence pour Gradio
################################################################################
def inference_gradio(image):
    """
    Cette fonction reçoit une image uploadée par l'utilisateur via Gradio,
    détecte les anomalies en se basant sur la memory bank few-shot, et renvoie
    le score d'anomalie ainsi qu'une image de visualisation avec la carte d'anomalie.
    """
    score, anomaly_map, input_image = detect_anomaly(image, memory_index, model, grid_size_ref, object_name=object_name, k=1)
    
    # Création d'une colormap personnalisée pour visualiser la carte d'anomalie
    neon_violet = (0.5, 0.1, 0.5, 0.4)
    neon_yellow = (0.8, 1.0, 0.02, 0.7)
    colors = [(1.0, 1.0, 1.0, 0.0), neon_violet, neon_yellow]
    cmap = LinearSegmentedColormap.from_list("AnomalyMap", colors, N=256)
    
    # Utilisation de matplotlib pour superposer la carte d'anomalie sur l'image test
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(input_image)
    ax.imshow(anomaly_map, cmap=cmap, alpha=0.7)
    ax.axis("off")
    plt.title(f"Score d'anomalie : {score:.3f}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    result_img = Image.open(buf)
    
    return f"Score d'anomalie : {score:.3f}", result_img

################################################################################
# Définition de l'interface Gradio
################################################################################
iface = gr.Interface(
    fn=inference_gradio,
    inputs=gr.Image(type="pil", label="Image de test (MVTEC AD)"),
    outputs=[
        gr.Textbox(label="Score d'anomalie"),
        gr.Image(label="Carte d'anomalie")
    ],
    title="Détection d’anomalies Few-Shot sur MVTEC AD",
    description="Uploader une image de test pour obtenir un score d'anomalie et une carte d'anomalie via une approche few-shot basée sur DINOv2."
)


if __name__ == "__main__":
    iface.launch(share=True)