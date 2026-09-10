# ESKD Model Post-Deployment Evaluation

This repository contains the evaluation, statistical analysis, fairness assessment, and figure-generation components used to assess the post-deployment performance of a machine learning model for predicting progression to End-Stage Kidney Disease (ESKD). The repository is intended to support transparency and reproducibility of the evaluation framework while protecting patient privacy and complying with institutional data governance requirements. 

---

## Repository Scope

This repository focuses on:

- Model performance evaluation
- Error analysis
- Fairness and bias assessment
- Bootstrap-based subgroup analyses
- Statistical reporting
- Figure generation for publication
- Post-deployment monitoring analyses

The repository does **not** contain machine learning model training code, clinical deployment infrastructure, or raw electronic health record (EHR) data. 

---

## Why Data Preprocessing Code Is Not Released

The original ESKD model was developed using longitudinal EHR data from a large academic healthcare system. Feature construction required access to protected clinical information, including laboratory measurements, diagnoses, procedures, medications, demographics, encounter history, and other health-related records. 

Although we support open scientific research whenever possible, the preprocessing and cohort-construction components are **not released** for several reasons:

### 1. Protected Health Information (PHI)

The preprocessing workflow operates directly on patient-level EHR data. Releasing these pipelines could unintentionally expose information about institutional data structures, identifiers, coding systems, or operational workflows that may increase re-identification risk.

### 2. Institution-Specific Data Sources

The original preprocessing pipeline relies on local:

- Clinical data warehouses
- EHR databases
- ICD and procedure mappings
- Medication hierarchies
- Operational reporting systems
- Institutional data models

These resources are highly organization-specific and would not be directly usable outside the originating health system. 

### 3. Regulatory and Ethical Requirements

The underlying data are governed by:

- HIPAA
- Institutional Review Board (IRB) requirements
- Data use agreements
- Institutional privacy and security policies

Public release of code tightly coupled to restricted clinical datasets may conflict with these obligations.

### 4. Production Workflow Protection

The original system includes clinical operations components, internal monitoring tools, dashboards, notification systems, and deployment infrastructure. These components are intentionally excluded from public release. 

---

## What Is Included

### Performance Evaluation

The repository contains code used to evaluate model performance after deployment, including:

- AUROC
- Precision-Recall AUC
- Calibration assessment
- Confusion matrix generation
- Threshold-based performance metrics
- Recall, precision, specificity, accuracy, and F1 score calculations

These analyses correspond to the post-deployment evaluation framework described in the associated publications. 

---

### Fairness and Equity Assessment

The repository includes code supporting fairness analyses across demographic and socioeconomic groups, including:

- Race
- Ethnicity
- Sex
- Age groups
- Social Vulnerability Index (SVI)

Supported analyses include:

- Subgroup performance evaluation
- Demographic parity assessment
- Effect-size estimation
- Statistical comparison of subgroup metrics
- Bootstrap-based disparity analysis

The framework was designed to evaluate whether model performance remains stable across patient populations after deployment. 

---

### Bootstrap Analysis

Included workflows support:

- Stratified bootstrap resampling
- Confidence interval estimation
- Metric ratio calculations
- Metric difference calculations
- Performance stability assessment

These analyses are used to characterize uncertainty and subgroup performance variation. 

---

### Figure Generation

The repository contains scripts used to create publication-quality figures, including:

- ROC curves
- Precision-Recall curves
- Calibration curves
- Fairness visualizations
- Lead-time analyses
- Confusion matrix summaries
- Subgroup comparison plots

Generated figures may differ slightly from published versions because of software-version differences, plotting libraries, and formatting adjustments made during manuscript preparation. 

---

## Reproducing the Analysis

Researchers may reproduce the evaluation pipeline using their own appropriately governed datasets.

At minimum, the evaluation framework expects access to:

- Model predictions
- Ground-truth outcomes
- Evaluation timestamps
- Demographic variables (where permitted)
- Relevant subgroup attributes

The evaluation code is designed to operate on derived analytical datasets rather than raw EHR data. Users are responsible for ensuring compliance with all applicable privacy, ethical, institutional, and regulatory requirements.

---

Any clinical deployment should undergo independent validation, privacy review, regulatory assessment, and institutional approval before use. 

---
