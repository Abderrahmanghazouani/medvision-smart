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
