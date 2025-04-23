from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
import io
import time
import numpy as np
from PIL import Image

from utils.utils_app import tensor_to_img, load_patchcore_model
from logger import get_logger

# Initialise le logger
logger = get_logger("api")

app = FastAPI(title="PatchCore Anomaly Detection API")

# Chargement modèle et seuil à l'startup
model, train_scores = load_patchcore_model(
    cls="bottle",
    backbone_key="WideResNet50",
    f_coreset=0.1,
    eps=0.9,
    k_nn=3,
    use_cache=True
)
default_thresh = float(np.percentile(train_scores, 95))

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    threshold: float = None
):
    start_time = time.time()

    # Lecture et préparation de l'image
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img = img.resize((model.image_size, model.image_size))
    # Transform identique à MVTecDataset
    transform = load_patchcore_model.__globals__['MVTecDataset'](
        "bottle", size=model.image_size, vanilla=model.vanilla
    ).get_datasets()[0].transform
    tensor = transform(img).unsqueeze(0)

    # Prédiction
    score, amap = model.predict(tensor)
    score_val   = float(score.item())

    # Seuil utilisé
    thr = threshold if threshold is not None else default_thresh

    # Génère le masque binaire
    amap_np   = amap.squeeze().detach().cpu().numpy()
    amap_norm = (amap_np - amap_np.min())/(amap_np.max()-amap_np.min()+1e-8)
    mask_bin  = (amap_norm >= thr).astype(int).tolist()

    duration = time.time() - start_time

    # Log métier
    logger.info(
        "api_predict",
        extra={
            "filename":    file.filename,
            "score":       score_val,
            "threshold":   thr,
            "duration_ms": int(duration*1000)
        }
    )

    return JSONResponse({
        "score":     score_val,
        "threshold": thr,
        "mask":      mask_bin
    })
