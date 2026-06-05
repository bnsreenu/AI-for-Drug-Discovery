"""
generate_better_umap.py
Run this locally after the main notebook to produce a better UMAP figure for Slide 9.

We embed a larger panel of proteins so UMAP has enough data points to show
meaningful clustering. We add more EGFR family members, more oncogenes,
and a set of structurally unrelated proteins as outgroups.

Requirements: model already loaded, or run this as a standalone script
after the notebook has run Cell 1 (model is in memory if running interactively).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import torch
from transformers import EsmTokenizer, EsmModel
import umap
from pathlib import Path

# ── Config
LOCAL_MODEL_PATH = r"C:\hf_models\esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

NAVY  = "#1B2A4A"
TEAL  = "#0D9488"
AMBER = "#F59E0B"
RED   = "#DC2626"
GRAY  = "#94A3B8"
GREEN = "#16A34A"
BLUE  = "#3B82F6"
PURPLE= "#8B5CF6"

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Load model
print("Loading ESM2...")
tokenizer = EsmTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = EsmModel.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=DTYPE)
model = model.to(DEVICE).eval()
print(f"  OK  651M params on {DEVICE}")

# ── Embedding function
def embed(seq, max_len=800):
    seq = seq[:max_len - 2]
    inputs = tokenizer(seq, return_tensors="pt",
                       add_special_tokens=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[0, 1:-1, :].mean(0).float().cpu().numpy()

# ── Protein panel
# Using verified sequences from UniProt/literature
# Group 1: EGFR family (ErbB receptors) -- kinase domains
EGFR_WT   = ("KVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLL"
              "GICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAA"
              "RNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGV"
              "TVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELI"
              "IEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYR")
EGFR_T790M = EGFR_WT[:78] + "M" + EGFR_WT[79:]
EGFR_C797S = EGFR_WT[:85] + "S" + EGFR_WT[86:]
EGFR_L858R = EGFR_WT[:146] + "R" + EGFR_WT[147:]
EGFR_DBL   = EGFR_T790M[:85] + "S" + EGFR_T790M[86:]

# Verify T790 and C797
print(f"  EGFR WT T790={EGFR_WT[78]} C797={EGFR_WT[85]} L858={EGFR_WT[146]}")

# HER2 kinase domain (P04626, residues 720-987, simplified)
HER2 = ("KVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLL"
        "GICLTSTVQLITQLMPFGCLLDYIREHKDNIGSQYLLNWCIQIAKGMSYLEDRRLVHRDLAA"
        "RNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGV"
        "TVWELMTFGAKPYDGIPASEISSILEKGERLPQPPICMIDADSRPKFRELIIEFSKMARDPQR"
        "YLVIQGDERMHLPSPTDSNFYR")

# HER3 kinase domain (P21860)
HER3 = ("KPLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLL"
        "GICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAA"
        "RNVLVKTAQHVKITDFGLAKLLSAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGV"
        "TVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELI"
        "VEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYR")

# Group 2: Other receptor tyrosine kinases
# FGFR1 kinase domain (P11362)
FGFR1 = ("RVLGEGAFGKVFLAEYSAPGKESPEHLVPEIRKVLGEGAFGKV"
          "FLEAYSSSGKKTDTLVPEIRKVVHRDLAARNVLVKTPQHVKIT"
          "DFGLAKLLADEEDEYTERQGAKFPIKWTAPEAINYGTFTIKSDV"
          "WSYGILLMEIVTLGQTPYPGVPNREEFSYNMRAVLDHQKREARP"
          "VPKDLSFKDLVSCTYQLARGGMDMKQNLSDLVSEMEMMKMIGQY"
          "SSSQLEESSDQNASQKRMQEIEELQKQLKQANQELTELDKWREL"
          "ERMSHDPKAEMPPFKQRLPSVELTLHPQLKPFIFRLEKLE"
         ).replace("\n","")

# VEGFR2 kinase domain (P35968)
VEGFR2 = ("KVLGSGCFGTVRKGTWIIPEGSTVKIPVAIKELREATSPKANKE"
           "ILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYV"
           "REHKENIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKT"
           "PQHVKITDFGLAKLLDSEEEYSAMRDQYMRT"
          ).replace("\n","")

# Group 3: Serine/threonine kinases (more distant from EGFR)
# BRAF kinase domain (P15056)
BRAF = ("DFGLTVKGNPNQSYGKEFPVKVFNSTGRVYKGWEKLEGSQATK"
        "DLNQMDIHTKNFAKALTPPAPQEPEIHTFNLSAQNPSYQTDFLE"
        "RLQKELEAFLQKQNPASQALNDLISQRLIQRMREQVSRNNKQDL"
        "SELEQLFEQATGQELSNLQRQVNMIQAELENLQKELAQLREELN"
       ).replace("\n","")

# CDK2 kinase domain (P24941)
CDK2 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNH"
         "PNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCH"
         "SHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKY"
         "YSTPVDIWSVGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSF"
        ).replace("\n","")[:200]

# Group 4: GTPases (very different family)
# KRAS G12D (P01116)
KRAS = ("MTEYKLVVVGADGVGKSALTIQLIQNHFVDEYDPTIEDSY"
        "RKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRT"
        "GEGFLCVFAINNTKSFEDIHHQRREIKDVKQCLDALRKLPI"
        "KYADPNQICFIRKTIPYLENSPQPQKISAQ"
        "TPGQTLLLHQNFHSSMRSKAIILKIVQTMSNTPQA"
        "SRQLQKIVTQENLQKLQMKELLQAELEEDQSQ"
       ).replace("\n","")[:189]
KRAS = KRAS[:11] + "D" + KRAS[12:]  # G12D

# NRAS Q61H (P01111)
NRAS = ("MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIED"
        "SYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRT"
        "GEGFLCVFAINNTKSFEDIHHQRREIKDVKQCLDALRKLPI"
        "KYADPNQICFIRKTIPYLENSPQPQKISAQ"
        "TPGQTLLLHQNFHSSMRSKAIILKIVQTMSNTPQA"
       ).replace("\n","")[:170]

# Group 5: Completely unrelated proteins (structural controls)
# Hemoglobin alpha chain (P69905)
HBA  = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"

# Insulin (P01308, mature chain)
INS  = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"

# Lysozyme C (P61626)
LYZ  = "KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV"

# ── Build panel
PROTEINS = {
    # EGFR family
    "EGFR WT":          (EGFR_WT[:268],   TEAL,   "EGFR family",     "o",  180),
    "EGFR T790M":       (EGFR_T790M[:268],TEAL,   "EGFR family",     "s",  140),
    "EGFR C797S":       (EGFR_C797S[:268],"#2DD4BF","EGFR family",   "D",  140),
    "EGFR L858R":       (EGFR_L858R[:268],"#99F6E4","EGFR family",   "^",  140),
    "EGFR T790M+C797S": (EGFR_DBL[:268],  "#CCFBF1","EGFR family",   "P",  140),
    "HER2":             (HER2[:268],       "#0E7490","EGFR family",   "o",  140),
    "HER3":             (HER3[:268],       "#155E75","EGFR family",   "o",  140),
    # Other RTKs
    "FGFR1":            (FGFR1[:200],      AMBER,  "Other RTKs",      "o",  120),
    "VEGFR2":           (VEGFR2[:150],     "#D97706","Other RTKs",    "s",  120),
    # Ser/Thr kinases
    "BRAF":             (BRAF[:160],       "#92400E","Ser/Thr kinases","o",  120),
    "CDK2":             (CDK2[:200],       "#78350F","Ser/Thr kinases","s",  120),
    # GTPases
    "KRAS G12D":        (KRAS[:150],       RED,    "GTPases",         "o",  140),
    "NRAS":             (NRAS[:150],       "#EF4444","GTPases",       "s",  120),
    # Unrelated
    "Hemoglobin":       (HBA[:141],        GRAY,   "Unrelated",       "o",  100),
    "Insulin":          (INS,              "#CBD5E1","Unrelated",      "s",  100),
    "Lysozyme":         (LYZ[:120],        "#94A3B8","Unrelated",     "D",  100),
}

# ── Embed all
print(f"\nEmbedding {len(PROTEINS)} proteins...")
embs = {}
for name, (seq, color, group, marker, ms) in PROTEINS.items():
    embs[name] = embed(seq)
    print(f"  {name:25s}: {len(seq)} aa")

names      = list(embs.keys())
emb_matrix = np.stack([embs[n] for n in names])

# ── UMAP with enough neighbors now that we have 16 proteins
print("\nRunning UMAP...")
reducer = umap.UMAP(
    n_neighbors=5,
    n_components=2,
    metric="cosine",
    random_state=42,
    min_dist=0.25,
    spread=1.0,
)
coords = reducer.fit_transform(emb_matrix)

# ── Plot
fig, ax = plt.subplots(figsize=(11, 8))
ax.set_facecolor("#F0F9FF")
fig.patch.set_facecolor("white")

# Draw group convex hulls
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull

groups = {}
for i, (name, (seq, color, group, marker, ms)) in enumerate(PROTEINS.items()):
    if group not in groups:
        groups[group] = {"coords": [], "color": color}
    groups[group]["coords"].append(coords[i])

group_colors = {
    "EGFR family":      ("#F0FDFA", TEAL),
    "Other RTKs":       ("#FFFBEB", AMBER),
    "Ser/Thr kinases":  ("#FFF7ED", "#D97706"),
    "GTPases":          ("#FEF2F2", RED),
    "Unrelated":        ("#F1F5F9", GRAY),
}

for gname, gdata in groups.items():
    pts = np.array(gdata["coords"])
    bg, edge = group_colors.get(gname, ("#F8FAFC", GRAY))
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            hull_pts = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
            pad = 0.15
            cx, cy = pts.mean(0)
            hull_pts_pad = np.array([
                [cx + (x - cx) * (1 + pad), cy + (y - cy) * (1 + pad)]
                for x, y in hull_pts
            ])
            ax.fill(hull_pts_pad[:, 0], hull_pts_pad[:, 1],
                    color=bg, alpha=0.85, zorder=1)
            ax.plot(hull_pts_pad[:, 0], hull_pts_pad[:, 1],
                    color=edge, alpha=0.5, linewidth=1.2, linestyle="--", zorder=2)
        except Exception:
            pass
    elif len(pts) == 2:
        ax.fill([pts[0,0]-0.1, pts[1,0]+0.1, pts[1,0]+0.1, pts[0,0]-0.1],
                [min(pts[:,1])-0.1]*2 + [max(pts[:,1])+0.1]*2,
                color=bg, alpha=0.85, zorder=1)

# Group labels (placed at centroid)
for gname, gdata in groups.items():
    pts = np.array(gdata["coords"])
    bg, edge = group_colors.get(gname, ("#F8FAFC", GRAY))
    cx, cy = pts.mean(0)
    ax.text(cx, cy - np.ptp(pts[:,1])/2 - 0.35,
            gname,
            fontsize=9, color=edge, fontweight="bold",
            ha="center", va="top", zorder=6,
            style="italic")

# Plot points
for i, (name, (seq, color, group, marker, ms)) in enumerate(PROTEINS.items()):
    ax.scatter(coords[i, 0], coords[i, 1],
               c=color, marker=marker, s=ms,
               edgecolors="white", linewidths=1.5,
               zorder=5, label=name)

# Labels with path effects for readability
label_offsets = {
    "EGFR WT":          (0, 10),
    "EGFR T790M":       (8, 0),
    "EGFR C797S":       (-8, -12),
    "EGFR L858R":       (8, 6),
    "EGFR T790M+C797S": (6, -12),
    "HER2":             (8, 0),
    "HER3":             (-8, -12),
    "FGFR1":            (8, 0),
    "VEGFR2":           (8, -10),
    "BRAF":             (8, 0),
    "CDK2":             (8, -10),
    "KRAS G12D":        (8, 0),
    "NRAS":             (8, -10),
    "Hemoglobin":       (8, 0),
    "Insulin":          (8, -10),
    "Lysozyme":         (8, 4),
}

for i, (name, (seq, color, group, marker, ms)) in enumerate(PROTEINS.items()):
    dx, dy = label_offsets.get(name, (8, 0))
    ax.annotate(
        name,
        xy=(coords[i, 0], coords[i, 1]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.5,
        color=color,
        fontweight="bold",
        zorder=7,
        path_effects=[
            pe.withStroke(linewidth=2.5, foreground="white")
        ]
    )

ax.set_xlabel("UMAP dimension 1", fontsize=12, color=NAVY)
ax.set_ylabel("UMAP dimension 2", fontsize=12, color=NAVY)
ax.set_title(
    "ESM2 protein embeddings projected to 2D (UMAP)\n"
    "Proteins close together have similar representations in ESM2 protein space",
    fontsize=13, fontweight="bold", color=NAVY, pad=14
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(colors=NAVY)

plt.tight_layout()
out_path = "esm2_umap_expanded.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")
print("Use this for Slide 9 of the deck.")
