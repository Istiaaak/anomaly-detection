from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import mlflow

from data.data import MVTecDataset
from utils.utils_app import load_patchcore_model  # utilitaire pour charger modèle et scores

default_args = {
    'owner': 'you',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='patchcore_testset_pipeline',
    default_args=default_args,
    description='Pipeline PatchCore Anomaly Detection',
    schedule_interval='@daily',
    start_date=datetime(2025, 4, 22),
    catchup=False,
    tags=['patchcore', 'testset']
) as dag:

    def train_model(**ctx):
        # Démarre un run MLflow
        mlflow.set_experiment('PatchCore_Experiment')
        run = mlflow.start_run()
        ctx['ti'].xcom_push('mlflow_run_id', run.info.run_id)

        # Paramètres
        cls = 'bottle'
        backbone_key = 'WideResNet50'
        f_coreset = 0.1
        eps       = 0.9
        k_nn      = 3
        use_cache = False

        # Log des paramètres
        mlflow.log_param('class', cls)
        mlflow.log_param('backbone', backbone_key)
        mlflow.log_param('f_coreset', f_coreset)
        mlflow.log_param('eps_coreset', eps)
        mlflow.log_param('k_nearest', k_nn)

        # Chargement et fit du modèle, calibration seuil
        model, train_scores = load_patchcore_model(
            cls, backbone_key, f_coreset, eps, k_nn, use_cache
        )
        threshold = float(np.percentile(train_scores, 95))
        mlflow.log_metric('train_threshold_95pct', threshold)

        # Stockage en XCom
        ctx['ti'].xcom_push('model', model)
        ctx['ti'].xcom_push('threshold', threshold)

        mlflow.end_run()

    def inference_test(**ctx):
        # Reprend le run MLflow
        run_id = ctx['ti'].xcom_pull('mlflow_run_id')
        mlflow.start_run(run_id=run_id)

        # Récupère modèle et seuil
        model    = ctx['ti'].xcom_pull('model')
        threshold= ctx['ti'].xcom_pull('threshold')
        cls      = 'bottle'
        size     = model.image_size
        vanilla  = model.vanilla

        # Chargement dataset de test
        ds = MVTecDataset(cls, size=size, vanilla=vanilla)
        _, test_ds = ds.get_datasets()
        test_dl    = DataLoader(test_ds, batch_size=1)

        image_scores, image_labels = [], []
        pixel_preds,  pixel_labels = [], []

        for img, mask, label in test_dl:
            s, amap = model.predict(img)
            image_scores.append(s.item())
            image_labels.append(int(label.item()))

            amap_np = amap.squeeze().cpu().numpy()
            amap_norm = (amap_np - amap_np.min()) / (amap_np.max() - amap_np.min() + 1e-8)
            pixel_preds.extend(amap_norm.flatten().tolist())
            pixel_labels.extend(mask.squeeze().cpu().numpy().flatten().tolist())

        ctx['ti'].xcom_push('image_scores',    image_scores)
        ctx['ti'].xcom_push('image_labels',    image_labels)
        ctx['ti'].xcom_push('pixel_preds',     pixel_preds)
        ctx['ti'].xcom_push('pixel_labels',    pixel_labels)

        mlflow.end_run()

    def compute_metrics(**ctx):
        # Reprend le run MLflow
        run_id = ctx['ti'].xcom_pull('mlflow_run_id')
        mlflow.start_run(run_id=run_id)

        # Récupère scores et labels
        scores    = np.array(ctx['ti'].xcom_pull('image_scores'))
        labels    = np.array(ctx['ti'].xcom_pull('image_labels'))
        pix_preds = np.array(ctx['ti'].xcom_pull('pixel_preds'))
        pix_labs  = np.array(ctx['ti'].xcom_pull('pixel_labels'))

        # Calcul ROC-AUC
        img_roc = roc_auc_score(labels, scores)
        pix_roc = roc_auc_score(pix_labs, pix_preds)

        # Log métriques
        mlflow.log_metric('image_roc_auc', img_roc)
        mlflow.log_metric('pixel_roc_auc', pix_roc)

        ctx['ti'].xcom_push('img_roc', img_roc)
        ctx['ti'].xcom_push('pix_roc', pix_roc)

        mlflow.end_run()

    def notify(**ctx):
        img_roc = ctx['ti'].xcom_pull('img_roc')
        pix_roc = ctx['ti'].xcom_pull('pix_roc')
        print(f"Pipeline terminée. Image ROC-AUC: {img_roc:.4f}, Pixel ROC-AUC: {pix_roc:.4f}")
        # Ici, on peut envoyer une alerte email/Slack, etc.

    # Définition des tâches
    t1 = PythonOperator(task_id='train_model',     python_callable=train_model,     provide_context=True)
    t2 = PythonOperator(task_id='inference_test',  python_callable=inference_test,  provide_context=True)
    t3 = PythonOperator(task_id='compute_metrics', python_callable=compute_metrics, provide_context=True)
    t4 = PythonOperator(task_id='notify',          python_callable=notify,          provide_context=True)

    # Orchestration
    t1 >> t2 >> t3 >> t4
