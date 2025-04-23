import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data.data import MVTecDataset, DEFAULT_SIZE
from model.patch_core import PatchCore
from utils.utils import backbones, dataset_scale_factor

def tensor_to_img(x: torch.Tensor, vanilla: bool) -> np.ndarray:
    x = x.clone().cpu()
    mean, std = (torch.tensor([.485, .456, .406]), torch.tensor([.229, .224, .225])) if vanilla else \
                (torch.tensor([.481, .457, .408]), torch.tensor([.268, .261, .275]))
    for c in range(x.shape[0]):
        x[c] = x[c] * std[c] + mean[c]
    return x.clip(0.0, 1.0).permute(1, 2, 0).numpy()

def load_patchcore_model(
    cls: str,
    backbone_key: str,
    f_coreset: float,
    eps: float,
    k_nn: int,
    use_cache: bool
):
    """
    Charge ou construit le modèle PatchCore
    et retourne (model, train_scores)
    """
    size = DEFAULT_SIZE
    vanilla = (backbone_key == 'WideResNet50')
    ds = MVTecDataset(cls, size=size, vanilla=vanilla)
    train_ds, _ = ds.get_datasets()

    # Instanciation
    model = PatchCore(
        f_coreset=f_coreset,
        eps_coreset=eps,
        k_nearest=k_nn,
        vanilla=vanilla,
        backbone=backbones[backbone_key],
        image_size=size
    )

    cache_file = Path("./patchcore_cache/memory_bank") / f"{cls}_{backbone_key}_f{f_coreset:.3f}.pth"
    if use_cache and cache_file.exists():
        mb = torch.load(cache_file)
        model.memory_bank = mb if isinstance(mb, torch.Tensor) else torch.cat(mb, 0)
        # init avg & resize
        model.avg = torch.nn.AvgPool2d(3, stride=1)
        batch, _ = next(iter(DataLoader(train_ds, batch_size=1)))
        _ = model.forward(batch)
        fmap_size = model.features[0].shape[-2]
        model.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
    else:
        model.fit(DataLoader(train_ds, batch_size=1), scale=dataset_scale_factor[backbone_key])
        mb = model.memory_bank.cpu() if isinstance(model.memory_bank, torch.Tensor) else torch.cat(model.memory_bank, 0).cpu()
        torch.save(mb, cache_file)

    # Calibration du seuil sur les "good"
    train_dl = DataLoader(train_ds, batch_size=1)
    train_scores = []
    for x, _ in train_dl:
        s, _ = model.predict(x)
        train_scores.append(s.item())
    train_scores = np.array(train_scores)

    return model, train_scores
