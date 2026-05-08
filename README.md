# Physics-Informed ML Framework for Hardness Prediction in Multi-Principal Element Alloys

## Overview

This repository contains the codebase developed for studying hardness prediction in multi-principal element alloys (MPEAs) using a combination of physics-informed descriptors, machine learning, explainability analysis, and finite element validation.

The workflow begins with literature-derived dataset preparation and descriptor construction, followed by baseline and optimized machine learning models for Vickers hardness prediction. Additional analyses include SHAP-based feature interpretation, inverse alloy screening, FEM-assisted validation, mesh convergence studies, and statistical relationship analysis between descriptors.

The overall objective of this work is to develop a reproducible and physically meaningful computational workflow for accelerated alloy design and hardness evaluation.

---

## Repository Structure

src/
├── step1_dataset_preprocessing.py
├── step1_1_dataset_validation.py
├── step2_baseline_ml_models.py
├── step3_descriptor_engineering.py
├── step4_descriptor_ml_models.py
├── step4_1_eda_visualization.py
├── step4_2_model_consistency.py
├── step5_inverse_alloy_design.py
├── step6_fem_material_generation.py
├── step6_1_fem_validation.py
├── step6_2_mesh_convergence.py
├── step7_creep_analysis.py
├── step8_shap_analysis.py
└── step9_feature_relationship_analysis.py

---

## Workflow Summary

### Dataset Preparation

Experimental MPEA datasets collected from literature sources were cleaned, normalized, validated, and consolidated into a unified dataset for modeling.

### Descriptor Engineering

A set of physics-informed descriptors related to atomic size, thermodynamic behavior, elastic response, and electronic characteristics was generated for hardness prediction.

### Machine Learning Modeling

Multiple regression models were evaluated using cross-validation and hyperparameter optimization to compare predictive performance.

### Explainability Analysis

SHAP analysis was performed to investigate feature importance, descriptor interactions, and model interpretability.

### Inverse Alloy Design

The trained framework was used to explore candidate alloy compositions and identify systems with potentially improved hardness response.

### FEM Validation

Finite element simulations based on Vickers indentation methodology were used to compare predicted and simulated hardness behavior.

---

## Main Libraries

* numpy
* pandas
* scikit-learn
* xgboost
* shap
* matplotlib
* seaborn
* scipy
* statsmodels

---

## Installation

pip install -r requirements.txt

---

## License

This project is released under the MIT License.
