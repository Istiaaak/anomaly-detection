#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline complet pour :
- Construire une memory bank à partir du dataset MVTEC AD (toutes les classes) avec stockage des features par classe
- Calculer pour chaque classe un centre représentatif (moyenne des features)
- Détecter les anomalies et classifier une image test via la proximité dans l'espace des features
- Visualiser le clustering via t-SNE et K-Means
- Mettre en place plusieurs onglets dans une interface Gradio :
    1. Détection & Classification (pour une image test)
    2. Visualisation du Clustering
    3. Dashboard Utilisateur (évaluation avec ground truth, si disponible)
    4. Dashboard Temps Réel (calcul des métriques sur un ensemble d’images non annotées)
    
Ce code fournit notamment un dashboard en temps réel qui, en simulant une fenêtre glissante sur des images uploadées, affiche la distribution et la tendance des scores d'anomalie.
"""

import os
import cv2
import numpy as np
import torch
import faiss
import matplotlib.pyplot as plt
from PIL import Image
import io
import gradio as gr

# Imports des modules personnalisés (assurez-vous qu'ils sont dans votre PYTHONPATH)
from detection import augment_image
from utils_dino import resize_mask_img, get_dataset_info
from src.backbones import get_model

# -----------------------------------------------------------------------------
# Construction de la memory bank avec stockage des features par classe
# -----------------------------------------------------------------------------
def build_combined_memory_bank(model, objects, mvtec_path, ref_files_per_class):
    """
    Parcourt les classes du dataset MVTEC et extrait les features des images "good" du training.
    
    Retourne :
      - index FAISS (sur l'ensemble des features)
      - grid_sizes : dictionnaire {classe: dernière taille de grille traitée}
      - final_features : tableau numpy global des features
      - features_by_class : dictionnaire {classe: features_array} pour la classification
    """
    combined_features = []
    grid_sizes = {}
    features_by_class = {}
    for obj in objects:
        print(f"Construction de la memory bank pour la classe : {obj}")
        ref_folder = os.path.join(mvtec_path, obj, "train", "good")
        features_list = []
        for fname in ref_files_per_class:
            ref_path = os.path.join(ref_folder, fname)
            if not os.path.exists(ref_path):
                continue
            # Charger l'image en RGB
            image_ref = cv2.cvtColor(cv2.imread(ref_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # Appliquer une augmentation par rotation si activée pour cet objet
            imgs = augment_image(image_ref) if rotation_default[obj] else [image_ref]
            for img in imgs:
                tensor_ref, grid_size_ref = model.prepare_image(img)
                feats = model.extract_features(tensor_ref)
                mask = model.compute_background_mask_from_image(img, threshold=10, masking_type=masking_default[obj])
                features_list.append(feats[mask])
        if features_list:
            features_obj = np.concatenate(features_list, axis=0)
            combined_features.append(features_obj)
            grid_sizes[obj] = grid_size_ref
            features_by_class[obj] = features_obj
    if combined_features:
        final_features = np.concatenate(combined_features, axis=0)
        faiss.normalize_L2(final_features)
        index = faiss.IndexFlatL2(final_features.shape[1])
        index.add(final_features)
        return index, grid_sizes, final_features, features_by_class
    else:
        raise ValueError("Aucune feature extraite pour construire la memory bank.")

# -----------------------------------------------------------------------------
# Calcul des centres par classe (vecteur moyen pour chaque classe)
# -----------------------------------------------------------------------------
def compute_class_centers(features_by_class):
    """
    Calcule et renvoie un dictionnaire {classe: centre} où centre est la moyenne normalisée des features.
    """
    class_centers = {}
    for cls, feats in features_by_class.items():
        center = np.mean(feats, axis=0)
        norm = np.linalg.norm(center)
        if norm != 0:
            center = center / norm
        class_centers[cls] = center
    return class_centers

# -----------------------------------------------------------------------------
# Détection d'anomalies (version multiclasses)
# -----------------------------------------------------------------------------
def detect_anomaly_multiclass(image, memory_index, model, object_name=None, k=1):
    """
    Extrait les features de l'image test, effectue une recherche kNN dans l'index FAISS,
    et calcule un score d'anomalie.
    
    Retourne : score d'anomalie, carte d'anomalie redimensionnée, image test (numpy array)
    """
    image_cv = np.array(image)
    tensor_test, grid_size_test = model.prepare_image(image_cv)
    feats_test = model.extract_features(tensor_test)
    faiss.normalize_L2(feats_test)
    
    distances, _ = memory_index.search(feats_test, k=k)
    distances = distances / 2  # Transformation de la distance L2 en équivalent "1 - cosinus"
    
    if object_name is not None:
        mask_test = model.compute_background_mask_from_image(image_cv, threshold=10, masking_type=masking_default[object_name])
    else:
        mask_test = np.ones(feats_test.shape[0], dtype=bool)
    distances[~mask_test] = 0.0
    
    all_dist = distances.flatten()
    sorted_dist = np.sort(all_dist)[::-1]
    top_count = max(1, int(len(sorted_dist) * 0.01))
    anomaly_score = np.mean(sorted_dist[:top_count])
    
    distance_map = distances.reshape(grid_size_test)
    anomaly_map = cv2.resize(distance_map, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_LINEAR)
    return anomaly_score, anomaly_map, image_cv

# -----------------------------------------------------------------------------
# Classification par proximité : comparaison avec les centres de classes
# -----------------------------------------------------------------------------
def classify_image(image, model, class_centers):
    """
    Extrait les features de l'image test, calcule leur moyenne et la compare aux centres de chaque classe.
    Renvoie la classe prédite et la distance associée.
    """
    image_cv = np.array(image)
    tensor_test, _ = model.prepare_image(image_cv)
    feats_test = model.extract_features(tensor_test)
    test_repr = np.mean(feats_test, axis=0)
    norm = np.linalg.norm(test_repr)
    if norm != 0:
        test_repr = test_repr / norm
    
    best_cls = None
    best_score = float("inf")
    for cls, center in class_centers.items():
        score = np.linalg.norm(test_repr - center)
        if score < best_score:
            best_score = score
            best_cls = cls
    return best_cls, best_score

# -----------------------------------------------------------------------------
# Fonction d'inférence combinée pour Gradio (détection + classification)
# -----------------------------------------------------------------------------
def inference_complete(image, memory_index, model, class_centers):
    """
    Pour une image test, détecte les anomalies ET prédit la classe via la proximité dans l'espace des features.
    Affiche le score d'anomalie, la carte d'anomalie et la classe prédite.
    """
    anomaly_score, anomaly_map, input_image = detect_anomaly_multiclass(image, memory_index, model, object_name=None, k=1)
    pred_class, class_distance = classify_image(image, model, class_centers)
    
    cmap = plt.cm.viridis  
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(input_image)
    ax.imshow(anomaly_map, cmap=cmap, alpha=0.7)
    ax.axis("off")
    ax.set_title(f"Score d'anomalie : {anomaly_score:.3f}\nClasse prédite : {pred_class} (distance: {class_distance:.3f})")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    result_img = Image.open(buf)
    
    return f"Score d'anomalie : {anomaly_score:.3f}\nClasse prédite : {pred_class}", result_img

# -----------------------------------------------------------------------------
# Visualisation du clustering via t-SNE et K-Means
# -----------------------------------------------------------------------------
def visualize_clustering(final_features, n_clusters=10):
    """
    Applique K-Means sur l'ensemble des features, réduit la dimension via t-SNE et retourne une image
    affichant les clusters.
    """
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(final_features)
    print(f"Clustering terminé. Nombre de clusters : {n_clusters}")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    features_2d = tsne.fit_transform(final_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='tab10', s=1)
    ax.set_title("Visualisation t-SNE des clusters de la memory bank")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.colorbar(scatter, ticks=range(n_clusters), label="Cluster ID")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    cluster_img = Image.open(buf)
    return cluster_img

# -----------------------------------------------------------------------------
# Dashboard d'évaluation sur les images test fournies par l'utilisateur (avec ground truth)
# -----------------------------------------------------------------------------
def evaluate_user_dashboard(test_images, ground_truth_labels):
    """
    Prend une liste d'images test et une chaîne d'étiquettes (séparées par des virgules),
    puis effectue la classification pour chaque image afin de calculer et afficher des métriques.
    
    Les étiquettes doivent être dans le même ordre que les images.
    Retourne un rapport textuel et la matrice de confusion sous forme d'image.
    """
    if not test_images:
        return "Aucune image test fournie.", None
    
    ground_truth_list = [label.strip() for label in ground_truth_labels.split(",")]
    if len(ground_truth_list) != len(test_images):
        return "Le nombre d'étiquettes ne correspond pas au nombre d'images.", None
    
    predictions = []
    for image in test_images:
        pred_class, _ = classify_image(image, model, class_centers)
        predictions.append(pred_class)
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    report = classification_report(ground_truth_list, predictions, output_dict=True)
    acc = accuracy_score(ground_truth_list, predictions)
    cm = confusion_matrix(ground_truth_list, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Véritables")
    plt.colorbar(cax)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    cm_img = Image.open(buf)
    
    report_str = f"Accuracy: {acc:.3f}\nRapport de classification:\n"
    for label, metrics in report.items():
        report_str += f"{label}: {metrics}\n"
    
    return report_str, cm_img

# -----------------------------------------------------------------------------
# Dashboard en temps réel : visualisation dynamique des métriques sans ground truth
# -----------------------------------------------------------------------------
def dashboard_realtime(test_images):
    """
    Prend une liste d'images test (simulant une fenêtre glissante) et calcule :
      - Un histogramme des scores d'anomalie
      - Une tendance (courbe) des scores
      - Quelques statistiques : moyenne, écart-type, minimum, maximum
    Retourne un résumé textuel et une image du dashboard.
    """
    if not test_images:
        return "Aucune image uploadée.", None
    
    anomaly_scores = []
    for image in test_images:
        score, _, _ = detect_anomaly_multiclass(image, memory_index, model, object_name=None, k=1)
        anomaly_scores.append(score)
    
    mean_score = np.mean(anomaly_scores)
    std_score = np.std(anomaly_scores)
    min_score = np.min(anomaly_scores)
    max_score = np.max(anomaly_scores)
    
    # Création d'un graphique avec histogramme et tendance
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(anomaly_scores, bins=20, color='blue', alpha=0.7)
    axs[0].set_title("Histogramme des Scores d'Anomalie")
    axs[0].set_xlabel("Score d'anomalie")
    axs[0].set_ylabel("Fréquence")
    
    axs[1].plot(anomaly_scores, marker='o', linestyle='-', color='green')
    axs[1].set_title("Tendance des Scores d'Anomalie")
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("Score d'anomalie")
    
    fig.suptitle(f"Moyenne: {mean_score:.3f}, Écart-type: {std_score:.3f}, Min: {min_score:.3f}, Max: {max_score:.3f}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    dashboard_img = Image.open(buf)
    
    summary_text = (f"Moyenne: {mean_score:.3f}\n"
                    f"Écart-type: {std_score:.3f}\n"
                    f"Minimum: {min_score:.3f}\n"
                    f"Maximum: {max_score:.3f}")
    return summary_text, dashboard_img

# -----------------------------------------------------------------------------
# Partie principale : chargement ou création de la memory bank et lancement de Gradio
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration du dataset MVTEC AD
    data_root = "./data"
    mvtec_path = os.path.join(data_root, "mvtec ad")
    
    # Récupération des informations du dataset
    objects, object_anomalies, masking_default, rotation_default = get_dataset_info("MVTec", "informed")
    
    # Chargement du modèle DINOv2 via le wrapper (mode évaluation)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = get_model("dinov2_vits14", device, smaller_edge_size=448)
    model.model.eval()
    
    # Vérification de l'existence d'une memory bank sauvegardée
    memory_bank_file = "memory_bank.pth"
    if os.path.exists(memory_bank_file):
        print("Memory bank existante détectée, chargement...")
        memory_bank_dict = torch.load(memory_bank_file)
        final_features = (
            memory_bank_dict["final_features"].numpy()
            if isinstance(memory_bank_dict["final_features"], torch.Tensor)
            else memory_bank_dict["final_features"]
        )
        grid_sizes = memory_bank_dict["grid_sizes"]
        objects = memory_bank_dict["objects"]
        features_by_class = memory_bank_dict["features_by_class"]
        faiss.normalize_L2(final_features)
        memory_index = faiss.IndexFlatL2(final_features.shape[1])
        memory_index.add(final_features)
        print("Memory bank chargée et index FAISS reconstruit.")
    else:
        print("Aucune memory bank existante détectée. Construction en cours...")
        ref_files_per_class = [f"{i:03}.png" for i in range(100)]
        memory_index, grid_sizes, final_features, features_by_class = build_combined_memory_bank(
            model, objects, mvtec_path, ref_files_per_class
        )
        memory_bank_dict = {
            "final_features": torch.tensor(final_features),
            "grid_sizes": grid_sizes,
            "objects": objects,
            "features_by_class": features_by_class,
        }
        torch.save(memory_bank_dict, memory_bank_file)
        print(f"Memory bank construite et sauvegardée dans '{memory_bank_file}'.")
    
    # Calcul des centres par classe
    class_centers = compute_class_centers(features_by_class)
    
    # Modification de dashboard_realtime pour ouvrir les fichiers images à partir du chemin
    def dashboard_realtime(file_paths):
        """
        Prend une liste de chemins de fichiers image (upload via gr.Files avec type="filepath")
        et calcule :
          - Un histogramme des scores d'anomalie
          - Une tendance des scores
          - Quelques statistiques : moyenne, écart-type, min, max
        Retourne un résumé textuel et une image du dashboard.
        """
        if not file_paths:
            return "Aucune image uploadée.", None
        
        anomaly_scores = []
        # Pour chaque chemin de fichier, on ouvre l'image
        for file_path in file_paths:
            image = Image.open(file_path).convert("RGB")
            score, _, _ = detect_anomaly_multiclass(image, memory_index, model, object_name=None, k=1)
            anomaly_scores.append(score)
        
        mean_score = np.mean(anomaly_scores)
        std_score = np.std(anomaly_scores)
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].hist(anomaly_scores, bins=20, color='blue', alpha=0.7)
        axs[0].set_title("Histogramme des Scores d'Anomalie")
        axs[0].set_xlabel("Score d'anomalie")
        axs[0].set_ylabel("Fréquence")
        
        axs[1].plot(anomaly_scores, marker='o', linestyle='-', color='green')
        axs[1].set_title("Tendance des Scores d'Anomalie")
        axs[1].set_xlabel("Index")
        axs[1].set_ylabel("Score d'anomalie")
        
        fig.suptitle(f"Moyenne: {mean_score:.3f}, Écart-type: {std_score:.3f}, Min: {min_score:.3f}, Max: {max_score:.3f}")
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        dashboard_img = Image.open(buf)
        
        summary_text = (f"Moyenne: {mean_score:.3f}\n"
                        f"Écart-type: {std_score:.3f}\n"
                        f"Minimum: {min_score:.3f}\n"
                        f"Maximum: {max_score:.3f}")
        return summary_text, dashboard_img
    
    # Définition des interfaces Gradio
    
    # 1. Détection & Classification (image test)
    iface_anomaly = gr.Interface(
        fn=lambda img: inference_complete(img, memory_index, model, class_centers),
        inputs=gr.Image(type="pil", label="Image de test (MVTEC AD)"),
        outputs=[gr.Textbox(label="Informations de détection"), gr.Image(label="Visualisation")],
        title="Détection d'anomalies et Classification",
        description="Uploader une image pour obtenir un score d'anomalie, une carte d'anomalie et la classe prédite."
    )
    
    # 2. Visualisation du Clustering
    iface_clustering = gr.Interface(
        fn=lambda: visualize_clustering(final_features, n_clusters=10),
        inputs=[],
        outputs=gr.Image(label="Visualisation des clusters"),
        title="Visualisation du Clustering",
        description="Affichage t-SNE des clusters obtenus par K-Means sur la memory bank."
    )
    
    # 3. Dashboard Temps Réel (sans ground truth)
    iface_realtime_dashboard = gr.Interface(
        fn=dashboard_realtime,
        inputs=gr.Files(type="filepath", label="Images test (multi-upload)"),
        outputs=[gr.Textbox(label="Résumé des métriques"), gr.Image(label="Dashboard")],
        title="Dashboard en Temps Réel",
        description="Uploader plusieurs images pour visualiser en temps réel la distribution des scores d'anomalie."
    )
    
    # Regroupement des interfaces dans un tableau de bord à onglets
    demo = gr.TabbedInterface(
        [iface_anomaly, iface_clustering, iface_realtime_dashboard],
        tab_names=["Détection & Classification", "Visualisation Clustering", "Dashboard Temps Réel"]
    )
    
    demo.launch(share=True)
