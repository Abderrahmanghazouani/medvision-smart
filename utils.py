# Fonctions partagées du projet MedVision Smart.
import cv2
import numpy as np
import pandas as pd

SEED = 42
SPLITS_CSV = "splits.csv"


def lire_image(chemin):
    # Lit une échographie en niveaux de gris (tableau 2D, valeurs 0-255).
    return cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)


def lire_masque(mask_paths):
    # Lit et fusionne (OU logique) tous les masques d'une image. Renvoie un tableau 0/1.
    masque = None
    for chemin in mask_paths.split(";"):
        m = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE) > 0
        masque = m if masque is None else (masque | m)
    return masque.astype(np.uint8)


def charger_split(split=None, chemin=SPLITS_CSV):
    # Charge splits.csv ; si split = "train", "val" ou "test", ne garde que cette partition.
    df = pd.read_csv(chemin)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


# ---------------------------------------------------------------
# Ajouté par Hiba (notebook 02) : prétraitement et métriques communs
# ---------------------------------------------------------------

def redimensionner(img, masque, taille=256):
    # Image : INTER_AREA | Masque : INTER_NEAREST pour rester en 0/1
    img = cv2.resize(img, (taille, taille), interpolation=cv2.INTER_AREA)
    masque = cv2.resize(masque, (taille, taille), interpolation=cv2.INTER_NEAREST)
    return img, masque


def metriques(pred, verite):
    """Compare un masque prédit et le masque réel (tableaux 0/1 de même taille).
    Image sans tumeur : Dice = IoU = 1 si rien n'est prédit, 0 sinon ; rappel = NaN."""
    pred = pred.astype(bool)
    verite = verite.astype(bool)
    vp = np.logical_and(pred, verite).sum()
    fp = np.logical_and(pred, ~verite).sum()
    fn = np.logical_and(~pred, verite).sum()

    if verite.sum() == 0:
        ok = float(pred.sum() == 0)
        return {"dice": ok, "iou": ok, "rappel": np.nan, "precision": ok}

    return {
        "dice":      2 * vp / (2 * vp + fp + fn),
        "iou":       vp / (vp + fp + fn),
        "rappel":    vp / (vp + fn),
        "precision": vp / (vp + fp) if (vp + fp) > 0 else 0.0,
    }
