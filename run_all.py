"""
MedVision Smart - Lance toutes les expériences d'une traite (reprend là où il s'est arrêté).

Toutes les expériences finales (6 réseaux x 5 graines x 2 datasets = 60 entraînements) :
    python run_all.py

Étape 1 - réglage du learning rate des nouveaux modèles (graine 0, sur la VALIDATION) :
    python run_all.py --modeles attention_unet deeplabv3plus segformer --graines 0 --lrs 1e-3 1e-4 --sortie runs_reglage
    -> écrit lr_choisis.json (meilleur learning rate par dataset et par modèle)

Étape 2 - expériences finales (relit lr_choisis.json ; sinon LR_DEFAUT de models.py) :
    python run_all.py

Un entraînement déjà terminé (resume.json présent) est sauté : on peut arrêter et relancer sans rien perdre.
Les données d'un dataset ne sont lues qu'une fois pour tous ses entraînements.
"""
import argparse
import glob
import json
import os
import time

import pandas as pd

from models import MODELES
from train import SPLITS, charger_partition, entrainer

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=list(SPLITS))
ap.add_argument("--modeles", nargs="+", default=MODELES)
ap.add_argument("--graines", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--lrs", nargs="+", type=float, default=[None], help="plusieurs valeurs = réglage")
ap.add_argument("--epoques", type=int, default=60)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--sortie", default="runs")
ap.add_argument("--limite", type=int, default=None)
ap.add_argument("--sans-pretrain", action="store_true")
a = ap.parse_args()

reglage = len(a.lrs) > 1
FICHIER_LR = "lr_choisis.json"   # écrit par le réglage, relu par les expériences finales
lr_choisis = json.load(open(FICHIER_LR, encoding="utf-8")) if os.path.exists(FICHIER_LR) else {}
total = len(a.datasets) * len(a.modeles) * len(a.graines) * len(a.lrs)
n, debut = 0, time.time()
for dataset in a.datasets:
    print(f"\n=== Chargement de {dataset} ===", flush=True)
    donnees = {s: charger_partition(dataset, s, 256, a.limite) for s in ("train", "val")}
    for modele in a.modeles:
        for lr in a.lrs:
            if lr is None:   # expérience finale : learning rate retenu au réglage, sinon celui de LR_DEFAUT
                lr = lr_choisis.get(dataset, {}).get(modele)
            sortie = os.path.join(a.sortie, f"lr{lr:g}") if reglage else a.sortie
            for graine in a.graines:
                n += 1
                fichier = os.path.join(sortie, dataset, modele, f"graine{graine}", "resume.json")
                if os.path.exists(fichier):
                    print(f"[{n}/{total}] déjà fait : {fichier}")
                    continue
                print(f"\n[{n}/{total}] {dataset} | {modele} | graine {graine} | lr {lr} "
                      f"| écoulé {(time.time() - debut) / 60:.0f} min", flush=True)
                entrainer(dataset, modele, graine, lr=lr, epoques=a.epoques, patience=a.patience,
                          sortie=sortie, limite=a.limite, pretrained=not a.sans_pretrain, donnees=donnees)

# Tableau récapitulatif de tout ce qui est terminé
resumes = [json.load(open(f, encoding="utf-8")) for f in glob.glob(os.path.join(a.sortie, "**", "resume.json"), recursive=True)]
if resumes:
    t = pd.DataFrame(resumes)
    cols = ["dice_val", "iou_val", "temps_entrainement_s", "memoire_gpu_max_mo", "meilleure_epoque"]
    g = t.groupby(["dataset", "modele", "lr"])[cols].agg(["mean", "std"]).round(3)
    nom = os.path.join(a.sortie, "recapitulatif_validation.csv")
    g.to_csv(nom)
    print("\n=== Validation : moyenne et écart-type sur les graines ===")
    print(g.to_string())
    print(f"\nEnregistré : {nom}")

    # Réglage : on retient, pour chaque dataset et chaque modèle, le learning rate au meilleur Dice de validation
    if reglage:
        choix = {}
        meilleurs = t.groupby(["dataset", "modele", "lr"])["dice_val"].mean().reset_index()
        for (ds, mod), g2 in meilleurs.groupby(["dataset", "modele"]):
            choix.setdefault(ds, {})[mod] = float(g2.loc[g2["dice_val"].idxmax(), "lr"])
        anciens = json.load(open(FICHIER_LR, encoding="utf-8")) if os.path.exists(FICHIER_LR) else {}
        for ds, d in choix.items():
            anciens.setdefault(ds, {}).update(d)
        json.dump(anciens, open(FICHIER_LR, "w", encoding="utf-8"), indent=2)
        print(f"\nLearning rates retenus (enregistrés dans {FICHIER_LR}) :", json.dumps(choix, indent=2))
