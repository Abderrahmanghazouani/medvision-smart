# MedVision Smart — Segmentation biomédicale intelligente

Comparaison rigoureuse d'algorithmes de segmentation de lésions sur deux jeux de données médicaux, en vue de leur intégration dans une plateforme en microservices avec validation humaine des masques.

**Sujet 1.5 — [SAI-3]** · Module Deep Learning et Microservices · 5IIR, 2026–2027
**Équipe :** Abderrahman Ghazouani · Hibatallah Hamrani · Mohammed Anas Latifi

---

## Sommaire

1. [Objectifs](#1-objectifs)
2. [Jeux de données](#2-jeux-de-données)
3. [Méthodologie](#3-méthodologie)
4. [Architectures comparées](#4-architectures-comparées)
5. [Résultats](#5-résultats)
6. [Installation](#6-installation)
7. [Utilisation](#7-utilisation)
8. [Structure du dépôt](#8-structure-du-dépôt)
9. [Limites](#9-limites)

---

## 1. Objectifs

- **Segmenter automatiquement** des lésions : pour chaque image, produire un masque binaire (1 = lésion, 0 = fond).
- **Comparer objectivement** 7 algorithmes, d'une baseline sans apprentissage à des architectures récentes, dans un pipeline unique et reproductible.
- **Garantir une évaluation sans fuite de données** entre entraînement, validation et test.
- **Vérifier la généralisation** de la démarche sur un second dataset d'une autre modalité.
- **Choisir de façon argumentée** le modèle à déployer selon la précision, la complexité et le temps d'inférence.

## 2. Jeux de données

| | **BUSI** | **Kvasir-SEG** |
|---|---|---|
| Examen | Échographie mammaire | Coloscopie |
| Lésions | Tumeurs du sein | Polypes |
| Images | 780 (niveaux de gris) | 1 000 (couleur) |
| Catégories | *benign* 437 · *malignant* 210 · *normal* 133 | une seule ; analyse par taille (petit / moyen / grand) |
| Source | [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) | [Simula](https://datasets.simula.no/kvasir-seg/) |
| Après nettoyage | **762** images | **1 000** images |
| Découpage train / val / test | 537 / 114 / 111 | 701 / 149 / 150 |

Les datasets ne sont pas versionnés : les placer dans `data/Dataset_BUSI_with_GT/` et `data/Kvasir-SEG/`.

### Nettoyage de BUSI : de 780 à 762 images

| Étape | Problème | Traitement | Images |
|---|---|---|---|
| Masques multiples | 17 images à 2 ou 3 masques | Fusion par OU logique | 780 |
| Doublons exacts (MD5) | 1 image présente dans *benign* et *malignant* | 2 copies retirées | 778 |
| Quasi-doublons (pHash + corrélation) | 137 groupes, 310 images | Regroupés pour le découpage | 778 |
| Groupes contradictoires | 7 groupes étiquetés dans 2 classes | 16 images retirées | **762** |

**Critère d'exclusion :** une image est retirée uniquement si la même échographie (copie exacte ou quasi-doublon) apparaît sous **deux étiquettes différentes**, car on ne peut pas savoir laquelle est correcte.

## 3. Méthodologie

### 3.1 Découpage sans fuite

Les deux datasets contiennent des **quasi-doublons** : la même image redimensionnée, compressée, ou capturée plusieurs fois au cours d'un même examen. Ils sont détectés par **hash perceptuel (pHash)** et **corrélation des pixels**, regroupés par union-find, puis le découpage 70 / 15 / 15 est fait **par groupe** : toutes les copies d'une image tombent dans la même partition. Trois contrôles automatiques le vérifient (aucune image, aucune empreinte MD5 et aucun groupe partagés entre partitions).

- **BUSI** : un découpage image par image aurait laissé 68 groupes sur 137 à cheval entre partitions.
- **Kvasir-SEG** : les seuils de BUSI créaient un faux groupe de 110 images, à cause du cadre commun à toutes les images. Les empreintes sont donc calculées sur le **centre** de l'image, avec des seuils recalibrés (distance ≤ 10, corrélation ≥ 0,90), vérifiés à l'œil.

### 3.2 Prétraitement

| Opération | Détail |
|---|---|
| Redimensionnement | 256 × 256 ; image en `INTER_AREA`, masque en `INTER_NEAREST` (reste binaire) |
| Masque | Binarisation `> 127` (nécessaire pour les masques JPEG de Kvasir-SEG) |
| Canaux | 3 canaux pour tous les modèles (BUSI, en niveaux de gris, est dupliqué sur 3 canaux) |
| Normalisation | Pixels divisés par 255, puis moyenne et écart-type d'ImageNet |
| Augmentation | Miroir horizontal, probabilité 0,5, **entraînement uniquement** |

### 3.3 Entraînement

| Paramètre | Valeur |
|---|---|
| Optimiseur | Adam |
| Perte | BCE + Dice |
| Batch | 8 |
| Époques | 60 au maximum, arrêt anticipé (patience 10) sur le Dice de validation |
| Learning rate | 10⁻³ (CNN simple), 10⁻⁴ (encodeurs pré-entraînés) ; nouveaux modèles réglés sur la validation |
| Graines | 0, 1, 2, 3, 4 : le découpage est fixe, la graine change l'initialisation, l'augmentation et l'ordre des images |

### 3.4 Évaluation

- **Le test n'est utilisé qu'une fois**, après le choix de toutes les configurations sur la validation.
- **Métriques par image, à 256 × 256** : Dice, IoU, précision, rappel, F1, mAP (seuils d'IoU de 0,50 à 0,95). Pour une image sans lésion : 1 si rien n'est prédit, 0 sinon.
- Résultats en **moyenne ± écart-type sur 5 graines**, intervalles de confiance à 95 % par bootstrap, tests statistiques appariés entre modèles.
- Matrices de confusion et courbes ROC au niveau des pixels (et des images pour BUSI).
- Coûts : temps d'entraînement, temps d'inférence par image, mémoire GPU maximale, paramètres, couches.

## 4. Architectures comparées

| Modèle | Idée principale | Encodeur |
|---|---|---|
| Seuillage d'Otsu | Baseline sans apprentissage : garde les pixels sombres | — |
| CNN encodeur-décodeur | Réduit puis agrandit l'image, **sans** skip connections | entraîné de zéro |
| U-Net | **Skip connections** entre encodeur et décodeur | ResNet34, ImageNet |
| U-Net++ | Skip connections **denses et imbriquées** | ResNet34, ImageNet |
| Attention U-Net | **Portes d'attention** qui filtrent les skip connections | ResNet34, ImageNet |
| DeepLabV3+ | Convolutions **dilatées** à plusieurs échelles (ASPP) | ResNet34, ImageNet |
| SegFormer | Encodeur **Transformer** hiérarchique, décodeur léger | MiT-B2, ImageNet |

Le nombre de couches et de paramètres de chaque réseau s'obtient avec `python models.py`.

## 5. Résultats

> Résultats finaux en cours de calcul (6 réseaux × 5 graines × 2 datasets). Cette section sera complétée avec les tableaux en moyenne ± écart-type.

**Étape 1 (BUSI, une graine, test de 111 images) :**

| Modèle | Dice | Dice sur les lésions | Rappel | Tumeurs non détectées | ms / image |
|---|---|---|---|---|---|
| Otsu | 0,129 | 0,154 | 0,195 | 6 / 93 | 1,4 |
| CNN simple | 0,732 | 0,691 | 0,697 | 1 / 93 | 4,8 |
| **U-Net** | 0,729 | **0,741** | **0,752** | 2 / 93 | 11,5 |
| U-Net++ | **0,752** | 0,736 | 0,740 | 10 / 93 | 14,8 |

## 6. Installation

```bash
git clone https://github.com/Abderrahmanghazouani/medvision-smart.git
cd medvision-smart
python -m venv .venv
# Windows : .venv\Scripts\activate   |   Linux / macOS : source .venv/bin/activate

# 1) PyTorch adapté à la machine (voir https://pytorch.org), par exemple avec CUDA 12.1 :
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2) le reste des dépendances :
pip install -r requirements.txt
```

Poids des modèles entraînés : [Releases GitHub](https://github.com/Abderrahmanghazouani/medvision-smart/releases).

## 7. Utilisation

```bash
# Préparation des données (une seule fois, résultats déjà fournis : splits.csv, splits_kvasir.csv)
jupyter notebook 01_exploration.ipynb           # BUSI
jupyter notebook 01_exploration_kvasir.ipynb    # Kvasir-SEG

# Complexité des 6 réseaux
python models.py

# Un entraînement
python train.py --dataset busi --modele unet --graine 0

# Réglage du learning rate des nouveaux modèles, puis toutes les expériences
python run_all.py --modeles attention_unet deeplabv3plus segformer --graines 0 --lrs 1e-3 1e-4 --sortie runs_reglage
python run_all.py

# Test rapide sans GPU ni téléchargement (16 images, 2 époques)
python train.py --dataset kvasir --modele unet --limite 16 --epoques 2 --sans-pretrain
```

Chaque entraînement écrit dans `runs/<dataset>/<modele>/graine<g>/` : `historique.csv` (pertes, Dice et IoU par époque), `resume.json` (configuration, scores, temps, mémoire GPU, paramètres) et `best.pth` (poids).

## 8. Structure du dépôt

```
medvision-smart/
├── 01_exploration.ipynb          BUSI : exploration, nettoyage, découpage sans fuite
├── 01_exploration_kvasir.ipynb   Kvasir-SEG : idem
├── 02_baselines.ipynb            Otsu et CNN simple
├── 03_unet_unetpp.ipynb          U-Net et U-Net++ (étape 1)
├── 04_evaluation.ipynb           évaluation finale sur le test, analyse d'erreurs
├── models.py                     les 6 réseaux + complexité
├── train.py                      entraînement d'un modèle
├── run_all.py                    toutes les expériences (reprise automatique)
├── utils.py                      lecture des données, métriques communes
├── splits.csv / splits_kvasir.csv  découpages officiels (ne jamais les refaire)
├── requirements.txt
└── outputs/                      figures et tableaux
```

## 9. Limites

- **Pas d'identifiant patient** dans les deux datasets : le découpage par groupe de quasi-doublons en est la meilleure approximation.
- **Petits jeux de test** (111 et 150 images) : intervalles de confiance larges.
- **Qualité des annotations** : 7 groupes d'images de BUSI étaient étiquetés de façon contradictoire.
- **Seuil de décision fixé à 0,5**, non réglé.
- **Usage pédagogique** : ces modèles ne sont pas des dispositifs médicaux.
