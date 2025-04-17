import gradio as gr
import numpy as np
import cv2
import faiss
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
from matplotlib.colors import LinearSegmentedColormap
import base64

# Importez vos modules personnalisés (à adapter selon votre structure de projet)
from src.backbones import get_model
from detection import augment_image
from  import get_dataset_info

# Charger le modèle (ici, on utilise l'approche few-shot pour une classe "custom")
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Chargement du modèle DINOv2...")
    model = get_model("dinov2_vits14", device, smaller_edge_size=448)
    model.model.eval()  # Appliquez l'évaluation sur le modèle encapsulé
    print("Modèle chargé.")
    return model

model = load_model()

# Pour simplifier, nous définissons une seule classe "custom" avec des paramètres de masquage/rotation par défaut
object_name = "custom"
masking_default = {object_name: True}
rotation_default = {object_name: False}
# Vous pouvez éventuellement appeler get_dataset_info si vous en avez besoin
_, _, _, _ = get_dataset_info("MVTec", "informed")

# Fonction pour construire la memory bank à partir d'images de référence uploadées
def build_memory_bank(uploaded_refs):
    features_ref = []
    grid_size_ref = None
    if not uploaded_refs:
        return None, None
    for file in uploaded_refs:
        try:
            img = Image.open(file).convert("RGB")
        except Exception as e:
            print(f"Erreur avec l'image {file.name}: {e}")
            continue
        img_cv = np.array(img)
        # Si vous souhaitez activer la rotation, insérez augment_image ici
        imgs_versions = [img_cv]  # Pour simplifier, utilisez simplement l'image originale
        for img_version in imgs_versions:
            image_tensor, grid_size = model.prepare_image(img_version)
            feats = model.extract_features(image_tensor)
            mask = model.compute_background_mask_from_image(img_version, threshold=10, masking_type=masking_default[object_name])
            features_ref.append(feats[mask])
            grid_size_ref = grid_size
    if features_ref:
        features_ref = np.concatenate(features_ref, axis=0)
        index = faiss.IndexFlatL2(features_ref.shape[1])
        faiss.normalize_L2(features_ref)
        index.add(features_ref)
        return index, grid_size_ref
    else:
        return None, None

# Fonction pour détecter l'anomalie sur l'image test à partir de la memory bank
def detect_anomaly(test_img, memory_index):
    # Convertir l'image de test en tableau numpy
    image_cv = np.array(test_img)
    image_tensor, grid_size_test = model.prepare_image(image_cv)
    feats_test = model.extract_features(image_tensor)
    faiss.normalize_L2(feats_test)
    
    # Recherche kNN (k=1) pour chaque patch
    distances, _ = memory_index.search(feats_test, k=1)
    distances = distances / 2  # Normalisation pour obtenir un équivalent de (1 - similarité cosinus)
    
    # Masquage sur l'image de test
    mask_test = model.compute_background_mask_from_image(image_cv, threshold=10, masking_type=masking_default[object_name])
    distances[~mask_test] = 0.0
    
    # Calcul du score d'anomalie : moyenne des 1% des distances les plus élevées
    all_dist = distances.flatten()
    sorted_dist = np.sort(all_dist)[::-1]
    top_count = max(1, int(len(sorted_dist) * 0.01))
    anomaly_score = float(np.mean(sorted_dist[:top_count]))
    
    # Remodeler la carte d'anomalie pour correspondre à l'image de test
    distance_map = distances.reshape(grid_size_test)
    anomaly_map = cv2.resize(distance_map, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_LINEAR)
    return anomaly_score, anomaly_map, image_cv

# Interface d'inférence Gradio
def inference(uploaded_refs, test_img):
    # Construire la memory bank à partir des images de référence uploadées
    if not uploaded_refs:
        return "Veuillez uploader au moins une image de référence.", None
    memory_index, grid_size_ref = build_memory_bank(uploaded_refs)
    if memory_index is None:
        return "Erreur lors de la construction de la memory bank.", None
    
    # Détecter l'anomalie sur l'image de test
    anomaly_score, anomaly_map, input_image = detect_anomaly(test_img, memory_index)
    
    # Création d'une colormap personnalisée pour la carte d'anomalie
    neon_violet = (0.5, 0.1, 0.5, 0.4)
    neon_yellow = (0.8, 1.0, 0.02, 0.7)
    colors = [(1.0, 1.0, 1.0, 0.0), neon_violet, neon_yellow]
    cmap = LinearSegmentedColormap.from_list("AnomalyMap", colors, N=256)
    
    # Affichage de l'image de test avec la carte d'anomalie superposée
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(input_image)
    ax.imshow(anomaly_map, cmap=cmap, alpha=0.7)
    ax.axis("off")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    result_img = Image.open(buf)
    
    return f"Score d'anomalie : {anomaly_score:.3f}", result_img

# Définir les entrées et sorties dans Gradio
iface = gr.Interface(
    fn=inference,
    inputs=[
        gr.inputs.File(label="Images de Référence (Few-shot)", file_count="multiple"),
        gr.inputs.Image(type="pil", label="Image de Test")
    ],
    outputs=[
        gr.outputs.Textbox(label="Score d'anomalie"),
        gr.outputs.Image(label="Carte d'anomalie")
    ],
    title="Détection d'Anomalies Few-Shot avec vos Images",
    description="Uploader vos images de référence pour construire la memory bank, puis une image de test pour détecter une anomalie.",
    allow_flagging="never",   # désactiver la fonctionnalité de flag si non nécessaire
    share=True                # cette option permet de générer une URL publique pour la démo
)

iface.launch(share=True)
