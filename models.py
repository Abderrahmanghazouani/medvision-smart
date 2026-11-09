"""
MedVision Smart - Les 6 réseaux de segmentation comparés (Otsu est à part : otsu.py).

Tous les modèles reçoivent une image (N, 3, 256, 256) normalisée ImageNet
et renvoient des logits (N, 1, 256, 256) : sigmoid(logits) > 0,5 = pixel de lésion.

    cnn_simple      CNN encodeur-décodeur SANS skip connections (même architecture que le notebook 02)
    unet            U-Net, encodeur ResNet34 pré-entraîné ImageNet
    unetpp          U-Net++, même encodeur
    attention_unet  Attention U-Net (Oktay et al., 2018) : portes d'attention sur les skip connections, même encodeur
    deeplabv3plus   DeepLabV3+, même encodeur (convolutions dilatées ASPP)
    segformer       SegFormer, encodeur Mix Transformer MiT-B2 pré-entraîné ImageNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

MODELES = ["cnn_simple", "unet", "unetpp", "attention_unet", "deeplabv3plus", "segformer"]

NOMS = {
    "cnn_simple": "CNN simple", "unet": "U-Net", "unetpp": "U-Net++",
    "attention_unet": "Attention U-Net", "deeplabv3plus": "DeepLabV3+", "segformer": "SegFormer",
}

# Learning rate par défaut de chaque modèle (choisi sur la validation : notebooks 02, 03 et run_all --lrs)
LR_DEFAUT = {
    "cnn_simple": 1e-3,       # entraîné de zéro : préfère des pas plus grands
    "unet": 1e-4,             # encodeur pré-entraîné : petits pas
    "unetpp": 1e-4,
    "attention_unet": 1e-4,
    "deeplabv3plus": 1e-4,
    "segformer": 1e-4,
}


# ---------------------------------------------------------------- CNN simple
def bloc_conv(entree, sortie):
    """Deux convolutions 3x3 + BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(entree, sortie, 3, padding=1), nn.BatchNorm2d(sortie), nn.ReLU(inplace=True),
        nn.Conv2d(sortie, sortie, 3, padding=1), nn.BatchNorm2d(sortie), nn.ReLU(inplace=True),
    )


class CNNSimple(nn.Module):
    """Encodeur-décodeur SANS skip connections : 4 niveaux de réduction, un goulot, 4 niveaux d'agrandissement."""

    def __init__(self, in_channels=3, f=16):
        super().__init__()
        self.enc1, self.enc2 = bloc_conv(in_channels, f), bloc_conv(f, 2 * f)
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


# ---------------------------------------------------------------- Attention U-Net
class PorteAttention(nn.Module):
    """Porte d'attention (Oktay et al., 2018) : le décodeur (g) indique quelles zones du skip (x) garder."""

    def __init__(self, ch_g, ch_x, ch_inter):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(ch_g, ch_inter, 1), nn.BatchNorm2d(ch_inter))
        self.wx = nn.Sequential(nn.Conv2d(ch_x, ch_inter, 1), nn.BatchNorm2d(ch_inter))
        self.psi = nn.Sequential(nn.Conv2d(ch_inter, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        alpha = self.psi(F.relu(self.wg(g) + self.wx(x)))   # carte d'attention entre 0 et 1
        return x * alpha


class BlocDecodeurAttention(nn.Module):
    def __init__(self, ch_in, ch_skip, ch_out):
        super().__init__()
        self.porte = PorteAttention(ch_in, ch_skip, max(ch_skip // 2, 8)) if ch_skip else None
        self.conv = bloc_conv(ch_in + ch_skip, ch_out)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, self.porte(x, skip)], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """U-Net à portes d'attention, avec le même encodeur ResNet34 et les mêmes canaux de décodeur que smp.Unet."""

    def __init__(self, encoder_weights="imagenet", in_channels=3, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        self.encoder = smp.encoders.get_encoder("resnet34", in_channels=in_channels, depth=5, weights=encoder_weights)
        ch = list(self.encoder.out_channels)          # (3, 64, 64, 128, 256, 512)
        skips = ch[1:-1][::-1] + [0]                  # (256, 128, 64, 64, 0)
        entrees = [ch[-1]] + list(decoder_channels[:-1])
        self.blocs = nn.ModuleList(
            [BlocDecodeurAttention(i, s, o) for i, s, o in zip(entrees, skips, decoder_channels)])
        self.tete = nn.Conv2d(decoder_channels[-1], 1, 3, padding=1)

    def forward(self, x):
        f = self.encoder(x)
        skips = f[1:-1][::-1] + [None]
        y = f[-1]
        for bloc, s in zip(self.blocs, skips):
            y = bloc(y, s)
        return self.tete(y)


# ---------------------------------------------------------------- fabrique
def creer_modele(nom, pretrained=True):
    """Construit le réseau `nom`. pretrained=False : poids aléatoires (tests rapides, sans téléchargement)."""
    w = "imagenet" if pretrained else None
    if nom == "cnn_simple":
        return CNNSimple(in_channels=3)
    if nom == "unet":
        return smp.Unet("resnet34", encoder_weights=w, in_channels=3, classes=1)
    if nom == "unetpp":
        return smp.UnetPlusPlus("resnet34", encoder_weights=w, in_channels=3, classes=1)
    if nom == "attention_unet":
        return AttentionUNet(encoder_weights=w, in_channels=3)
    if nom == "deeplabv3plus":
        return smp.DeepLabV3Plus("resnet34", encoder_weights=w, in_channels=3, classes=1)
    if nom == "segformer":
        return smp.Segformer("mit_b2", encoder_weights=w, in_channels=3, classes=1)
    raise ValueError(f"Modèle inconnu : {nom}. Choix : {MODELES}")


def complexite(modele):
    """Nombre de paramètres et de couches à poids (convolutions, convolutions transposées, linéaires)."""
    n_params = sum(p.numel() for p in modele.parameters())
    n_couches = sum(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) for m in modele.modules())
    return {"n_parametres": int(n_params), "n_couches": int(n_couches)}


if __name__ == "__main__":
    # python models.py  -> tableau de complexité des 6 réseaux (pour le rapport)
    import pandas as pd
    lignes = []
    for nom in MODELES:
        m = creer_modele(nom, pretrained=False).eval()
        with torch.no_grad():
            sortie = m(torch.zeros(1, 3, 256, 256))
        assert sortie.shape == (1, 1, 256, 256), (nom, sortie.shape)
        lignes.append({"modele": NOMS[nom], **complexite(m)})
    t = pd.DataFrame(lignes)
    t["parametres_M"] = (t["n_parametres"] / 1e6).round(2)
    print(t[["modele", "n_couches", "parametres_M"]].to_string(index=False))
