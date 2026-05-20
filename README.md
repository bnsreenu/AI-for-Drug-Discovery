# AI for Drug Discovery

This repository contains the Python code, notebooks, and supporting material for the **AI for Drug Discovery** YouTube series by Sreenivas Bhattiprolu.

The goal of this series is to explain how modern AI and machine learning methods are actually used across the drug discovery pipeline — with a strong focus on practical workflows, scientific reasoning, and realistic data analysis rather than toy examples.

The tutorials are designed for:
- Scientists entering AI/ML
- Data scientists interested in drug discovery
- Students learning cheminformatics and computational biology
- Researchers working with biological imaging or screening data
- Anyone interested in practical AI applications in pharmaceutical research

---

# YouTube Playlist

📺 Video Playlist:  
https://www.youtube.com/playlist?list=PLZsOBAyNTZwbjl4od2Q4FcwN7205NVxT9

---

# Topics Covered

## 1. Drug Discovery Pipeline Overview
Conceptual introduction to:
- Target identification
- Hit discovery
- Lead optimization
- Preclinical development
- Clinical trials
- Where AI fits into each stage

---

## 2. Dose-Response Analysis and IC50 Fitting
Topics covered:
- 4-parameter logistic (4PL) fitting
- IC50 estimation
- Confidence intervals
- Curve quality assessment
- Handling noisy and imperfect biological data
- LLM-assisted curve interpretation

---

## 3. Molecular Representations
Topics covered:
- SMILES representations
- Molecular fingerprints
- Morgan fingerprints
- MACCS keys
- Tanimoto similarity
- Physicochemical descriptors
- Lipinski Rule of Five
- Preparing QSAR-ready datasets from ChEMBL

---

## 4. QSAR Modeling
Topics covered:
- Regression and classification workflows
- Scaffold-aware train/test splitting
- Stratified Murcko scaffold splits
- XGBoost models
- Neural networks
- Early stopping
- Model evaluation
- SHAP interpretability

---

## 5. Virtual Screening
Topics covered:
- Building screening libraries
- Drug-likeness filtering
- Scoring compounds with trained QSAR models
- Diversity selection
- MaxMin selection
- Chemical space visualization using UMAP

---

## 6. High-Content Analysis / Cell Painting
Topics covered:
- What Cell Painting assays are and what each imaging channel captures
- Extracting morphological features from microscopy images
- Understanding CellProfiler-derived features
- Working with the LINCS Cell Painting dataset from the Broad Institute
- Merging profiles with mechanism-of-action (MoA) annotations
- PCA and UMAP visualization of phenotypic clustering
- Hierarchical clustering heatmaps of MoA consensus profiles

---

# Repository Structure

This repository contains:
- Python scripts
- Jupyter notebooks
- Data preparation workflows
- Visualization utilities
- Model training pipelines
- Example datasets
- Supporting material for YouTube tutorials

The code is intended to be educational and heavily commented where possible.

---

# Data Sources

Examples throughout the series may use public datasets and resources including:
- ChEMBL
- LINCS Cell Painting datasets
- Public microscopy datasets
- Open biological imaging repositories

---

# Disclaimer

This repository is intended for educational and research purposes only.

The workflows shown here are simplified educational implementations designed to explain concepts clearly. Real-world pharmaceutical pipelines involve significantly larger datasets, extensive validation, domain expertise, and regulatory considerations.

---

# License

This repository is released under the MIT License unless otherwise specified.
