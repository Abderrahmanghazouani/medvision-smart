"""
MedVision Smart - Réglage des hyperparamètres du CNN encodeur-décodeur simple
(baseline sans skip connections, notebook 02).

4 entraînements : perte {BCE+Dice, Dice} x learning rate {1e-3, 1e-4}.
Tout est évalué sur la VALIDATION (le test est réservé au notebook 04).

Lancement (depuis le dossier medvision-smart, environnement activé) :
    python reglage_cnn_simple.py

Fichiers produits :
    modeles/cnn_simple_<config>.pth          poids du meilleur modèle de chaque config
    outputs/historique_cnn_<config>.csv      courbes d'apprentissage (1 ligne par époque)
    outputs/resultats_cnn_val.csv            métriques image par image sur la validation
    outputs/resume_reglage_cnn.csv           tableau récapitulatif des 4 configs
"""
import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import charger_split, lire_image, lire_masque, metriques, redimensionner

# ------------------------------------------------------------------ paramètres
SEED = 42
TAILLE = 256
BATCH = 8
NB_EPOQUES = 40
PATIENCE = 10
CONFIGS = [  # (nom de la perte, learning rate)
    ("bce_dice", 1e-3),
    ("bce_dice", 1e-4),
    ("dice", 1e-3),
    ("dice", 1e-4),
]
OUT_DIR, MODELES_DIR = "outputs", "modeles"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELES_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixer_graine(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ données
def charger_partition(df):
    """Charge toutes les images d'une partition en mémoire (rapide : ~800 images 256x256)."""
    imgs, masques = [], []
    for _, ligne in df.iterrows():
        img, m = redimensionner(lire_image(ligne["image_path"]), lire_masque(ligne["mask_paths"]), TAILLE)
        imgs.append(img.astype(np.float32) / 255.0)
        masques.append(m.astype(np.float32))
    x = torch.from_numpy(np.stack(imgs))[:, None]      # (N, 1, H, W)
    y = torch.from_numpy(np.stack(masques))[:, None]
    return x, y


def miroir_aleatoire(x, y):
    """Augmentation : miroir gauche-droite sur ~50 % des images du lot."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x[flip] = x[flip].flip(-1)
    y[flip] = y[flip].flip(-1)
    return x, y


# ------------------------------------------------------------------ modèle
def bloc_conv(entree, sortie):
    return nn.Sequential(
        nn.Conv2d(entree, sortie, 3, padding=1), nn.BatchNorm2d(sortie), nn.ReLU(inplace=True),
        nn.Conv2d(sortie, sortie, 3, padding=1), nn.BatchNorm2d(sortie), nn.ReLU(inplace=True),
    )


class CNNSimple(nn.Module):
    """Encodeur-décodeur SANS skip connections (baseline face à U-Net)."""

    def __init__(self, f=16):
        super().__init__()
        self.enc1, self.enc2 = bloc_conv(1, f), bloc_conv(f, 2 * f)
        self.enc3, self.enc4 = bloc_conv(2 * f, 4 * f), bloc_conv(4 * f, 8 * f)
        self.pool = nn.MaxPool2d(2)
        self.centre = bloc_conv(8 * f, 16 * f)
        self.up4, self.dec4 = nn.ConvTranspose2d(16 * f, 8 * f, 2, stride=2), bloc_conv(8 * f, 8 * f)
        self.up3, self.dec3 = nn.ConvTranspose2d(8 * f, 4 * f, 2, stride=2), bloc_conv(4 * f, 4 * f)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * f, 2 * f, 2, stride=2), bloc_conv(2 * f, 2 * f)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * f, f, 2, stride=2), bloc_conv(f, f)
        self.sortie = nn.Conv2d(f, 1, 1)

    def forward(self, x):
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))
        x = self.centre(x)
        x = self.dec4(self.up4(x))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return self.sortie(x)


# ------------------------------------------------------------------ pertes
_bce = nn.BCEWithLogitsLoss()


def perte_dice(logits, cible, eps=1.0):
    proba = torch.sigmoid(logits)
    inter = (proba * cible).sum(dim=(1, 2, 3))
    total = proba.sum(dim=(1, 2, 3)) + cible.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (total + eps)).mean()


PERTES = {
    "bce_dice": lambda logits, cible: _bce(logits, cible) + perte_dice(logits, cible),
    "dice": perte_dice,
}


# ------------------------------------------------------------------ évaluation
@torch.no_grad()
def predire(modele, x):
    modele.eval()
    preds = []
    for i in range(0, len(x), BATCH):
        preds.append((torch.sigmoid(modele(x[i:i + BATCH].to(DEVICE))) > 0.5).cpu().numpy()[:, 0])
    return np.concatenate(preds)


def dice_moyen(modele, x, y):
    preds = predire(modele, x)
    return float(np.mean([metriques(p, v)["dice"] for p, v in zip(preds, y.numpy()[:, 0])]))


@torch.no_grad()
def evaluer_image_par_image(modele, x, y, df, nom):
    """Métriques + temps d'inférence pour chaque image (1 image à la fois)."""
    modele.eval()
    lignes = []
    for k in range(len(x)):
        xi = x[k:k + 1].to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        debut = time.perf_counter()
        pred = (torch.sigmoid(modele(xi)) > 0.5)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        duree_ms = (time.perf_counter() - debut) * 1000
        pred = pred.cpu().numpy()[0, 0]
        lignes.append({"modele": nom, "id": df.iloc[k]["id"], "classe": df.iloc[k]["classe"],
                       **metriques(pred, y[k, 0].numpy()), "temps_ms": duree_ms})
    return pd.DataFrame(lignes)


# ------------------------------------------------------------------ entraînement
def entrainer(nom_perte, lr, x_train, y_train, x_val, y_val):
    fixer_graine()
    modele = CNNSimple().to(DEVICE)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=lr)
    perte_fn = PERTES[nom_perte]
    generateur = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH, shuffle=True, generator=generateur)

    historique, meilleur_dice, meilleurs_poids, meilleure_epoque, sans_progres = [], -1.0, None, 0, 0
    debut_total = time.perf_counter()

    for epoque in range(1, NB_EPOQUES + 1):
        debut = time.perf_counter()
        modele.train()
        pertes = []
        for xb, yb in loader:
            xb, yb = miroir_aleatoire(xb.to(DEVICE), yb.to(DEVICE))
            optimiseur.zero_grad()
            perte = perte_fn(modele(xb), yb)
            perte.backward()
            optimiseur.step()
            pertes.append(perte.item())

        modele.eval()
        with torch.no_grad():
            pertes_val = [perte_fn(modele(x_val[i:i + BATCH].to(DEVICE)), y_val[i:i + BATCH].to(DEVICE)).item()
                          for i in range(0, len(x_val), BATCH)]
        dice_val = dice_moyen(modele, x_val, y_val)

        historique.append({"epoque": epoque, "perte_train": float(np.mean(pertes)),
                           "perte_val": float(np.mean(pertes_val)), "dice_val": dice_val})
        print(f"  Époque {epoque:2d} | perte train {np.mean(pertes):.3f} | perte val {np.mean(pertes_val):.3f} "
              f"| Dice val {dice_val:.3f} | {time.perf_counter() - debut:.0f} s")

        if dice_val > meilleur_dice:
            meilleur_dice, meilleure_epoque, sans_progres = dice_val, epoque, 0
            meilleurs_poids = copy.deepcopy(modele.state_dict())
        else:
            sans_progres += 1
            if sans_progres >= PATIENCE:
                print(f"  Arrêt anticipé : pas d'amélioration depuis {PATIENCE} époques.")
                break

    modele.load_state_dict(meilleurs_poids)
    return modele, pd.DataFrame(historique), meilleure_epoque, time.perf_counter() - debut_total


# ------------------------------------------------------------------ programme principal
def main():
    print("Appareil utilisé :", DEVICE)
    train_df, val_df = charger_split("train"), charger_split("val")
    print(f"Chargement des images : train {len(train_df)} | val {len(val_df)}")
    x_train, y_train = charger_partition(train_df)
    x_val, y_val = charger_partition(val_df)

    resultats, resume = [], []
    for nom_perte, lr in CONFIGS:
        nom = f"{nom_perte}_lr{lr:g}"
        print(f"\n=== Configuration : perte = {nom_perte} | learning rate = {lr:g} ===")
        modele, hist, meilleure_epoque, duree = entrainer(nom_perte, lr, x_train, y_train, x_val, y_val)

        torch.save(modele.state_dict(), f"{MODELES_DIR}/cnn_simple_{nom}.pth")
        hist.to_csv(f"{OUT_DIR}/historique_cnn_{nom}.csv", index=False)

        res = evaluer_image_par_image(modele, x_val, y_val, val_df, f"CNN simple ({nom})")
        resultats.append(res)
        tumeurs = res[res["classe"] != "normal"]
        resume.append({
            "config": nom, "perte": nom_perte, "lr": lr,
            "meilleure_epoque": meilleure_epoque, "nb_epoques": len(hist),
            "temps_entrainement_min": round(duree / 60, 1),
            "dice": res["dice"].mean(), "iou": res["iou"].mean(), "map": res["map"].mean(),
            "rappel": res["rappel"].mean(), "precision": res["precision"].mean(),
            "dice_tumeurs": tumeurs["dice"].mean(),
            "dice_benign": res.loc[res["classe"] == "benign", "dice"].mean(),
            "dice_malignant": res.loc[res["classe"] == "malignant", "dice"].mean(),
            "dice_normal": res.loc[res["classe"] == "normal", "dice"].mean(),
            "temps_inference_ms": res["temps_ms"].mean(),
        })
        print(f"  -> Dice val {res['dice'].mean():.3f} | meilleure époque {meilleure_epoque} | {duree / 60:.1f} min")

    pd.concat(resultats).to_csv(f"{OUT_DIR}/resultats_cnn_val.csv", index=False)
    resume = pd.DataFrame(resume).round(4).sort_values("dice", ascending=False)
    resume.to_csv(f"{OUT_DIR}/resume_reglage_cnn.csv", index=False)

    print("\n=== Récapitulatif (validation) ===")
    print(resume[["config", "dice", "iou", "map", "rappel", "precision", "dice_malignant",
                  "meilleure_epoque", "temps_entrainement_min"]].to_string(index=False))
    print(f"\nMeilleure configuration : {resume.iloc[0]['config']}")


if __name__ == "__main__":
    main()