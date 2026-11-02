# Baseline d'Otsu - MedVision Smart (notebook 02, Hiba)
# Utilisation : img = image en niveaux de gris uint8, déjà redimensionnée
#               avec utils.redimensionner(img, masque, 256)
import cv2
import numpy as np

# Meilleur réglage trouvé sur la validation (27 combinaisons, Dice val = 0,201)
MEILLEUR_REGLAGE = {"flou": 3, "nettoyage": 9, "zone": "sans_bord"}


def otsu_parametre(img, flou=3, nettoyage=9, zone="sans_bord"):
    """Segmentation par seuillage d'Otsu.
    flou      : taille du noyau du flou gaussien (3, 5 ou 9)
    nettoyage : taille de l'ouverture morphologique, 0 = aucune (0, 5 ou 9)
    zone      : "toutes"      -> garde toutes les zones sombres
                "plus_grande" -> garde la plus grande zone
                "sans_bord"   -> garde la plus grande zone qui ne touche pas le bord
                                 (rectangle englobant collé à un des 4 bords)
    Renvoie un masque 0/1 (uint8) de la même taille que l'image."""
    # 1. Flou gaussien
    img_f = cv2.GaussianBlur(img, (flou, flou), 0)
    # 2. Seuil d'Otsu inversé : les zones sombres deviennent 1
    _, binaire = cv2.threshold(img_f, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 3. Nettoyage : ouverture avec un noyau elliptique
    if nettoyage > 0:
        noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nettoyage, nettoyage))
        binaire = cv2.morphologyEx(binaire, cv2.MORPH_OPEN, noyau)
    if zone == "toutes":
        return binaire
    # 4. Choix d'une seule zone (connexité 8, par défaut dans OpenCV)
    nb, etiquettes, stats, _ = cv2.connectedComponentsWithStats(binaire)
    h, w = binaire.shape
    meilleure, aire_max = None, 0
    for k in range(1, nb):
        x, y, larg, haut, aire = stats[k]
        touche_bord = x == 0 or y == 0 or x + larg >= w or y + haut >= h
        if zone == "sans_bord" and touche_bord:
            continue
        if aire > aire_max:
            meilleure, aire_max = k, aire
    if meilleure is None:          # aucune zone valable -> masque vide
        return np.zeros_like(binaire)
    return (etiquettes == meilleure).astype(np.uint8)