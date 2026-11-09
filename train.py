"""
MedVision Smart - Entraînement d'UN modèle, sur UN dataset, avec UNE graine.

Exemple :
    python train.py --dataset busi --modele unet --graine 0

Ce que le script fait (identique pour tous les modèles) :
    1. lit le découpage officiel (splits.csv pour BUSI, splits_kvasir.csv pour Kvasir-SEG) ;
       le TEST n'est jamais chargé ici ;
    2. redimensionne en 256 x 256 (image : INTER_AREA, masque : INTER_NEAREST), masque binaire (> 127),
       image en 3 canaux normalisée avec la moyenne et l'écart-type d'ImageNet ;
    3. augmentation sur l'entraînement uniquement : miroir horizontal avec une probabilité de 0,5 ;
    4. Adam, perte BCE + Dice, arrêt anticipé sur le Dice de validation ;
    5. enregistre à chaque époque : perte d'entraînement, perte de validation, Dice et IoU de validation ;
    6. garde les poids de la meilleure époque et un résumé (temps, mémoire GPU maximale, paramètres, couches).

Sorties : runs/<dataset>/<modele>/graine<g>/
    historique.csv   une ligne par époque
    resume.json      configuration + meilleurs scores + coûts
    best.pth         poids de la meilleure époque (à mettre dans la Release GitHub, pas dans git)
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import MODELES, LR_DEFAUT, creer_modele, complexite

SPLITS = {"busi": "splits.csv", "kvasir": "splits_kvasir.csv"}
MOYENNE = np.array([0.485, 0.456, 0.406], dtype=np.float32)
ECART = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SEUIL_MASQUE = 127


# ------------------------------------------------------------------ données
def lire_paire(ligne, taille):
    """Image RGB uint8 (taille, taille, 3) et masque binaire uint8 (taille, taille)."""
    img = cv2.cvtColor(cv2.imread(ligne["image_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    masque = None
    for chemin in str(ligne["mask_paths"]).split(";"):       # BUSI : plusieurs masques possibles -> OU logique
        m = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE) > SEUIL_MASQUE
        masque = m if masque is None else (masque | m)
    img = cv2.resize(img, (taille, taille), interpolation=cv2.INTER_AREA)
    masque = cv2.resize(masque.astype(np.uint8), (taille, taille), interpolation=cv2.INTER_NEAREST)
    return img, masque


def charger_partition(dataset, split, taille, limite=None):
    df = pd.read_csv(SPLITS[dataset])
    assert split in ("train", "val"), "le test n'est jamais chargé pendant l'entraînement"
    df = df[df["split"] == split].reset_index(drop=True)
    if limite:
        df = df.head(limite)
    paires = [lire_paire(l, taille) for _, l in df.iterrows()]
    X = np.stack([p[0] for p in paires])          # uint8, (N, H, W, 3)
    Y = np.stack([p[1] for p in paires])          # uint8, (N, H, W)
    return X, Y


def vers_tenseur(X, Y, device):
    """uint8 -> float normalisé ImageNet, au format (N, 3, H, W) et (N, 1, H, W)."""
    x = (X.astype(np.float32) / 255.0 - MOYENNE) / ECART
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
    y = torch.from_numpy(Y.astype(np.float32))[:, None].to(device)
    return x, y


def miroir_aleatoire(x, y):
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x[flip] = x[flip].flip(-1)
    y[flip] = y[flip].flip(-1)
    return x, y


# ------------------------------------------------------------------ pertes et métriques
_bce = nn.BCEWithLogitsLoss()


def perte_dice(logits, cible, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * cible).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + cible.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


PERTES = {"bce_dice": lambda l, c: _bce(l, c) + perte_dice(l, c), "dice": perte_dice}


def dice_iou(pred, verite):
    """Dice et IoU d'une image (tableaux booléens). Masque réel vide : 1 si rien n'est prédit, 0 sinon."""
    vp = np.logical_and(pred, verite).sum()
    fp = np.logical_and(pred, ~verite).sum()
    fn = np.logical_and(~pred, verite).sum()
    if verite.sum() == 0:
        ok = float(pred.sum() == 0)
        return ok, ok
    return 2 * vp / (2 * vp + fp + fn), vp / (vp + fp + fn)


# ------------------------------------------------------------------ entraînement
def fixer_graine(graine):
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def entrainer(dataset, modele, graine, lr=None, perte="bce_dice", epoques=60, patience=10, batch=8,
              taille=256, sortie="runs", limite=None, pretrained=True, donnees=None, verbeux=True):
    lr = lr if lr is not None else LR_DEFAUT[modele]
    dossier = os.path.join(sortie, dataset, modele, f"graine{graine}")
    os.makedirs(dossier, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixer_graine(graine)

    if donnees is None:
        donnees = {s: charger_partition(dataset, s, taille, limite) for s in ("train", "val")}
    Xtr, Ytr = donnees["train"]
    Xva, Yva = donnees["val"]
    x_val, y_val = vers_tenseur(Xva, Yva, device)

    net = creer_modele(modele, pretrained=pretrained).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    f_perte = PERTES[perte]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    historique, meilleur, meilleure_ep, sans_progres = [], -1.0, 0, 0
    debut = time.time()
    for ep in range(1, epoques + 1):
        t0 = time.time()
        net.train()
        ordre = np.random.permutation(len(Xtr))
        pertes = []
        for k in range(0, len(ordre) - batch + 1, batch):
            idx = ordre[k:k + batch]
            x, y = miroir_aleatoire(*vers_tenseur(Xtr[idx], Ytr[idx], device))
            opt.zero_grad()
            l = f_perte(net(x), y)
            l.backward()
            opt.step()
            pertes.append(l.item())

        net.eval()
        pv, dices, ious = [], [], []
        with torch.no_grad():
            for k in range(0, len(x_val), batch):
                logits = net(x_val[k:k + batch])
                pv.append(f_perte(logits, y_val[k:k + batch]).item() * len(logits))
                preds = (torch.sigmoid(logits) > 0.5)[:, 0].cpu().numpy()
                for p, v in zip(preds, Yva[k:k + batch].astype(bool)):
                    d, i = dice_iou(p, v)
                    dices.append(d)
                    ious.append(i)
        ligne = {"epoque": ep, "perte_train": float(np.mean(pertes)), "perte_val": float(sum(pv) / len(x_val)),
                 "dice_val": float(np.mean(dices)), "iou_val": float(np.mean(ious)),
                 "duree_s": round(time.time() - t0, 2)}
        historique.append(ligne)
        if verbeux:
            print(f"[{dataset}|{modele}|g{graine}] ép {ep:3d} | perte train {ligne['perte_train']:.3f} "
                  f"| perte val {ligne['perte_val']:.3f} | Dice val {ligne['dice_val']:.3f} "
                  f"| IoU val {ligne['iou_val']:.3f} | {ligne['duree_s']:.0f} s", flush=True)

        if ligne["dice_val"] > meilleur:
            meilleur, meilleure_ep, sans_progres = ligne["dice_val"], ep, 0
            torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, os.path.join(dossier, "best.pth"))
        else:
            sans_progres += 1
            if sans_progres >= patience:
                break

    duree = time.time() - debut
    pd.DataFrame(historique).to_csv(os.path.join(dossier, "historique.csv"), index=False)
    best = historique[meilleure_ep - 1]
    import segmentation_models_pytorch as smp
    resume = {
        "dataset": dataset, "modele": modele, "graine": graine, "lr": lr, "perte": perte, "batch": batch,
        "taille": taille, "optimiseur": "Adam", "epoques_max": epoques, "patience": patience,
        "augmentation": "miroir horizontal p=0.5 (train uniquement)",
        "normalisation": "pixels/255 puis moyenne/écart-type ImageNet",
        "encodeur_pretraine": bool(pretrained) and modele != "cnn_simple",
        "meilleure_epoque": meilleure_ep, "epoques_effectuees": len(historique),
        "dice_val": best["dice_val"], "iou_val": best["iou_val"], "perte_val": best["perte_val"],
        "temps_entrainement_s": round(duree, 1),
        "memoire_gpu_max_mo": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
        **complexite(net),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "versions": {"torch": torch.__version__, "smp": smp.__version__},
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
    }
    with open(os.path.join(dossier, "resume.json"), "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)
    del net, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return resume


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entraîne un modèle de segmentation (MedVision Smart).")
    ap.add_argument("--dataset", choices=list(SPLITS), required=True)
    ap.add_argument("--modele", choices=MODELES, required=True)
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None, help="par défaut : LR_DEFAUT du modèle")
    ap.add_argument("--perte", choices=list(PERTES), default="bce_dice")
    ap.add_argument("--epoques", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--taille", type=int, default=256)
    ap.add_argument("--sortie", default="runs")
    ap.add_argument("--limite", type=int, default=None, help="n'utiliser que N images (test rapide)")
    ap.add_argument("--sans-pretrain", action="store_true", help="poids aléatoires (test rapide hors ligne)")
    a = ap.parse_args()
    r = entrainer(a.dataset, a.modele, a.graine, a.lr, a.perte, a.epoques, a.patience, a.batch, a.taille,
                  a.sortie, a.limite, not a.sans_pretrain)
    print(json.dumps(r, indent=2, ensure_ascii=False))
