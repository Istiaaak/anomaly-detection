import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import faiss
import cv2
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from detection import augment_image
from utils_dino import get_dataset_info
from src.backbones import get_model
import os

# -------------------------------
# Configuration et chargement du modèle
# -------------------------------

# Répertoire du dataset MVTEC AD
data_root = "./data"
mvtec_path = os.path.join(data_root, "mvtec ad")
object_names = ["hazelnut", "metal_nut", "cable"]  # Exemple de classes

# Récupération des infos du dataset (liste d'objets, type d'anomalie, paramètres masking/rotation)
objects, object_anomalies, masking_default, rotation_default = get_dataset_info("MVTec", "informed")

# Chargement du modèle DINOv2 via votre wrapper
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = get_model("dinov2_vits14", device, smaller_edge_size=448)
model.model.eval()

# -------------------------------
# Construction de la memory bank pour plusieurs classes
# -------------------------------

def build_memory_bank(model, object_name="hazelnut", ref_files=["000.png"], mvtec_path=mvtec_path):
    features_ref = []
    ref_folder = os.path.join(mvtec_path, object_name, "train", "good")
    for fname in ref_files:
        ref_path = os.path.join(ref_folder, fname)
        image_ref = cv2.cvtColor(cv2.imread(ref_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rotation_default[object_name]:
            imgs = augment_image(image_ref)
        else:
            imgs = [image_ref]
        for img in imgs:
            tensor_ref, grid_size_ref = model.prepare_image(img)
            feats = model.extract_features(tensor_ref)
            mask = model.compute_background_mask_from_image(img, threshold=10, masking_type=masking_default[object_name])
            features_ref.append(feats[mask])
    features_ref = np.concatenate(features_ref, axis=0)
    index = faiss.IndexFlatL2(features_ref.shape[1])
    faiss.normalize_L2(features_ref)
    index.add(features_ref)
    return index, grid_size_ref

def build_memory_bank_multiple_classes(model, mvtec_path=mvtec_path, object_names=["hazelnut", "metal_nut"]):
    memory_indices = {}
    grid_sizes = {}
    for object_name in object_names:
        ref_files = [f"{i:03}.png" for i in range(100)]
        memory_index, grid_size_ref = build_memory_bank(model, object_name=object_name, ref_files=ref_files)
        memory_indices[object_name] = memory_index
        grid_sizes[object_name] = grid_size_ref
    return memory_indices, grid_sizes

# Charger la memory bank pour plusieurs classes
memory_indices, grid_sizes = build_memory_bank_multiple_classes(model)

# -------------------------------
# Variables Globales pour les Scores et Métriques
# -------------------------------

anomaly_scores = []  # Liste pour stocker les scores d'anomalie au fur et à mesure
true_labels = []  # Labels réels (normal ou anomalie)
predicted_labels = []  # Labels prédits (normal ou anomalie)

# -------------------------------
# Fonction d'inférence pour les anomalies
# -------------------------------

def detect_anomaly(image, memory_indices, model, grid_sizes, object_name, k=1):
    image_cv = np.array(image)
    tensor_test, grid_size_test = model.prepare_image(image_cv)
    feats_test = model.extract_features(tensor_test)
    faiss.normalize_L2(feats_test)
    memory_index = memory_indices[object_name]
    distances, _ = memory_index.search(feats_test, k=k)
    distances = distances / 2  # L2 -> "1 - cosinus"
    
    mask_test = model.compute_background_mask_from_image(image_cv, threshold=10, masking_type=masking_default[object_name])
    distances[~mask_test] = 0.0

    all_dist = distances.flatten()
    sorted_dist = np.sort(all_dist)[::-1]
    top_count = max(1, int(len(sorted_dist) * 0.01))
    anomaly_score = np.mean(sorted_dist[:top_count])

    distance_map = distances.reshape(grid_size_test)
    anomaly_map = cv2.resize(distance_map, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_LINEAR)
    return anomaly_score, anomaly_map, image_cv

# -------------------------------
# Calcul des métriques (Précision, Rappel, F1)
# -------------------------------

def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return precision, recall, f1

# -------------------------------
# Histogramme des scores d'anomalie
# -------------------------------

def plot_anomaly_histogram(scores):
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=50, color='green', alpha=0.7)
    plt.title("Distribution des scores d'anomalie")
    plt.xlabel("Score d'anomalie")
    plt.ylabel("Fréquence")
    return plt

# -------------------------------
# Fonction d'inférence pour Gradio
# -------------------------------

def inference_gradio(image, object_name):
    # Effectuer la détection d'anomalie pour la classe sélectionnée
    score, anomaly_map, input_image = detect_anomaly(image, memory_indices, model, grid_sizes, object_name=object_name, k=1)

    # Ajouter le score d'anomalie à la liste
    anomaly_scores.append(score)
    true_labels.append(1)  # Simuler un label réel (1 pour anomalie)
    predicted_labels.append(1 if score > 0.5 else 0)

    # Calcul des métriques
    precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)

    # Tracer l'histogramme des scores d'anomalie
    hist_plot = plot_anomaly_histogram(anomaly_scores)

    # Visualiser l'image de sortie avec les résultats
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(input_image)
    ax.axis("off")
    ax.set_title(f"Score d'anomalie : {score:.3f}")

    # Convertir l'image en format pour Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    result_img = Image.open(buf)

    # Retourner les résultats
    return f"Score d'anomalie : {score:.3f}", result_img, f"Précision : {precision:.3f}", f"Rappel : {recall:.3f}", f"F1-score : {f1:.3f}", hist_plot

# -------------------------------
# Interface Gradio
# -------------------------------

iface = gr.Interface(
    fn=inference_gradio,
    inputs=[
        gr.Image(type="pil", label="Image de test (MVTEC AD)"),
        gr.Dropdown(choices=object_names, label="Sélectionnez l'objet")
    ],
    outputs=[
        gr.Textbox(label="Score d'anomalie"),
        gr.Image(label="Carte d'anomalie"),
        gr.Textbox(label="Précision"),
        gr.Textbox(label="Rappel"),
        gr.Textbox(label="F1-score"),
        gr.Plot(label="Histogramme des scores d'anomalie")
    ],
    title="Détection d’anomalies Few-Shot sur MVTEC AD",
    description="Uploader une image de test pour obtenir un score d'anomalie, des métriques et une carte d'anomalie via une approche few-shot basée sur DINOv2."
)

if __name__ == "__main__":
    iface.launch(share=True)
