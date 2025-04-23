# run.py

import os
import torch
from pathlib import Path

from data.data import MVTecDataset, mvtec_classes, DEFAULT_SIZE
from model.patch_core import PatchCore
from utils.utils import backbones, dataset_scale_factor

ALL_CLASSES = mvtec_classes()

def run_model(
        classes: list = ALL_CLASSES,
        backbone: str = 'WideResNet50',
        cache_root: str = './patchcore_cache/memory_bank'
) -> None:

    # Paramètres
    f_coreset = 0.1
    vanilla = backbone == "WideResNet50"

    # Résolution image selon backbone
    if vanilla:
        size = DEFAULT_SIZE
    elif backbone == 'ResNet50':
        size = 224
    elif backbone == 'ResNet50-4':
        size = 288
    elif backbone == 'ResNet50-16':
        size = 384
    elif backbone == 'ResNet101':
        size = 224
    else:
        raise ValueError("Backbone invalide")

    # Prépare le dossier de cache
    cache_dir = Path(cache_root).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Running PatchCore (backbone={backbone}, size={size})")
    for cls in classes:
        print(f"\n▶ Classe = {cls}")
        # DataLoader
        train_dl, test_dl = MVTecDataset(cls, size=size, vanilla=vanilla).get_dataloaders()

        # Instancie le modèle
        patch_core = PatchCore(
            f_coreset=f_coreset,
            vanilla=vanilla,
            backbone=backbones[backbone],
            image_size=size
        )

        # FIT / construction de memory_bank
        print("   • Training (fit) …")
        patch_core.fit(train_dl, scale=dataset_scale_factor[backbone])

        # Sauvegarde de la memory_bank générée
        cache_file = cache_dir / f"{cls}_{backbone}_f{f_coreset:.3f}.pth"
        print(f"   • Saving memory_bank to {cache_file}")
        # On s'assure de sauver un Tensor CPU
        mb = patch_core.memory_bank.cpu() if isinstance(patch_core.memory_bank, torch.Tensor) else torch.cat(patch_core.memory_bank, dim=0).cpu()
        torch.save(mb, cache_file)

        # (Optionnel) Évaluation
        print("   • Testing (evaluate) …")
        image_rocauc, pixel_rocauc = patch_core.evaluate(test_dl)
        print(f"     → Image-level ROC AUC = {image_rocauc:.4f}")
        print(f"     → Pixel-level ROC AUC = {pixel_rocauc:.4f}")

    print("\n✅ Done.")

if __name__ == "__main__":
    # Exemple : ne traiter que la classe 'transistor'
    run_model(backbone='WideResNet50')
