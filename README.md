# Spectral Color Formulation Prediction Using Deep Learning
End-to-End System for 6-Base Color Formulation, Spectral Reconstruction, and ΔE2000 Evaluation

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Spectral Color Mixing Theory](#spectral-color-mixing-theory)
- [Pipeline Overview](#pipeline-overview)
- [Base Spectrum Estimation (NNLS)](#base-spectrum-estimation-nnls)
- [6-Base Selection](#6-base-selection)
- [Ground-Truth 6-Weight Generation](#ground-truth-6-weight-generation)
- [Neural Network Architecture](#neural-network-architecture)
- [Training Strategy](#training-strategy)
- [Spectral Reconstruction](#spectral-reconstruction)
- [Reflectance → CIELAB Conversion](#reflectance--cielab-conversion)
- [ΔE2000 Computation](#dE2000-computation)
- [K-Fold Cross-Validation](#k-fold-cross-validation)
- [Project Structure](#project-structure)
- [Installation & Dependencies](#installation--dependencies)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

## Project Overview
This project builds a physically consistent, end-to-end deep learning system that predicts 6 colorant mixture weights (from an original set of 17 base paints) that best reproduce a given reflectance spectrum (R400–R700).

The full pipeline:

- Reflectance → Neural Network → 6-Base Formulation → Spectral Reconstruction → Lab → ΔE2000

It is built to be research-grade and optimized for color matching, ink formulation, coating design, and industrial color reproduction workflows.
**NOTE :- 
- The dataset is confidential and not included in this repository.
- The cleaned_notebook_v3.ipynb contains the complete pipeline with all steps.
-  **Similar algorithm is used by the second largest paint producer in the world for accurate color reproduction
. They have achieved 91 percent accuracy with their model, this model touches 80-85 percent**

## Problem Statement
Given:
- A reflectance curve 
𝑅
(
𝜆
)
R(λ), where 
𝜆
=
400
…
700
 nm
λ=400…700 nm

- Target LAB values

- Ground-truth 17-base mixture weights from experiments

We want to:

- Determine 17 base spectral curves (unknown)

- Reduce them to a smaller 6-base working set

- Predict 6-base mixture weights using deep learning

- Reconstruct reflectance from predicted weights

- Convert reflectance → Lab

- Compute color difference ΔE2000

with the goal of achieving:

- ΔE2000 < 1.0 for most samples (Perceptually perfect match)


## Dataset Description
The dataset contains:
| Component                               | Description                                      |
| --------------------------------------- | ------------------------------------------------ |
| **Reflectance R400–R700 (31 features)** | Reflectance at 10 nm intervals                   |
| **Lab values (L*, a*, b*)**             | Target perceptual color under D65                |
| **Base weights B1–B17**                 | Ground-truth mixing ratios used in manufacturing |

| R400 | R410 | ... | R700 | L_D65 | a_D65 | b_D65 | B1 | B2 | ... | B17 |
| ---- | ---- | --- | ---- | ----- | ----- | ----- | -- | -- | --- | --- |


**All missing values are filled with zero, and all reflectances are in 
[
0
,
1
]
[0,1].**


## Spectral Color Mixing Theory

Colorants mix additively in the reflectance domain:

```
R(λ)≈i=1∑17​wi​⋅Bi​(λ)
```
Where:

- 𝑅
(
𝜆
)
R(λ) = measured reflectance

- 𝑤
𝑖
w
i
	​
 = mixture weight for base 
𝑖
i

- 𝐵
𝑖
(
𝜆
)
B
i
	​
 (λ) = spectral basis function of the 
𝑖
i-th pigment

We do not know the base spectra 
𝐵
𝑖
(
𝜆
)
B
i
	​
 (λ).

These are estimated using Non-Negative Least Squares (NNLS).


## Pipeline Overview
```csharp
          ┌──────────────────────────────┐
          │ Reflectance Input (31 dims)  │
          └───────────────┬──────────────┘
                          ↓
 ┌─────────────────────────────────────────────────┐
 │ Neural Network predicts 6 mixture weights       │
 └──────────────────────────────┬──────────────────┘
                                ↓
             ┌────────────────────────────────┐
             │ Spectrum Reconstruction:        │
             │   R_pred = W6 @ B6^T           │
             └──────────────────┬─────────────┘
                                ↓
            ┌─────────────────────────────────┐
            │ Convert reflectance → XYZ → Lab │
            └──────────────────┬──────────────┘
                               ↓
           ┌───────────────────────────────────┐
           │ Compute ΔE2000 vs true Lab        │
           └───────────────────────────────────┘
```

##  Base Spectrum Estimation (NNLS)

We solve:
```
X:,j​=W⋅bj​
```

Where:
- 𝑋
:
,
𝑗
X
:,j
is the reflectance at wavelength 
𝑗
j

- W are the 17 base weights
- b
j
	​ gives the value of each base at that wavelength

We solve for each wavelength using:
```python
b_j, _ = nnls(W_train, X_train[:, j])
```
Result gives 17 base spectra across 31 wavelengths.


## 6-Base Selection
- We reduce 17 bases → 6 by selecting those with highest mean usage:
```python
mean_contrib_train = W_train.mean(axis=0)
selected_idx = np.argsort(mean_contrib_train)[-6:]
```

This improves model conditioning and accelerates training.

## Ground-Truth 6-Weight Generation
We compute, for each sample:
```
w6​=argw≥0min​∥X−B6​w∥
```

using:
```python
nnls(B6, X[i])
```
These 6-weight vectors are used as training targets.

## Neural Network Architecture

A fully-connected regression network:
```python
Input (31 dims)
↓
Dense 256, ReLU
↓
Dense 128, ReLU
↓
Dense 64, ReLU
↓
Dense 32, ReLU
↓
Dropout(0.1)
↓
Dense 64, ReLU
↓
Dense(6) → Softmax  → predicted W6
```

Notes:
- Softmax ensures weights sum to 1
- No LAB head (to avoid shortcut learning)
- Model learns mapping: Reflectance → Formulation


## Training Strategy

Loss:
- 𝐿
=
MSE
(
𝑤
6
true
,
𝑤
6
pred
)


Optimiser: Adam 1e−3
Early stopping + ReduceLR callbacks

## Spectral Reconstruction
After predicting 6 weights:

- 𝑅
pred
=
𝑊
6
pred
𝐵
6
𝑇


Enforced to be 
[
0
,
1
]
[0,1] via clipping.

## Reflectance → CIELAB Conversion
Using colour-science:
```python
sd = SpectralDistribution({λ: R})
XYZ = colour.sd_to_XYZ(sd)
Lab = colour.XYZ_to_Lab(XYZ / 100)
```
Conversion uses:
- CIE 1931 2° Standard Observer
- Illuminant D65


## ΔE2000 Computation

The perceptual difference metric:
```python
deltaE_ciede2000(true_lab, pred_lab)
```
Used to evaluate formulation quality.

## K-Fold Cross-Validation

We perform K = 5 folds:
For each fold:
- Split data
- Estimate base spectra
- Select top 6 bases
- Compute NNLS 6-weights
- Train model
- Predict weights
- Reconstruct reflectance
- Convert → Lab
- Compute ΔE2000

Returned metrics:
- Mean ΔE
- % samples ΔE < 1
- % samples ΔE < 2
- Selected bases per fold


##  Project Structure
```powershell
project/
│
├── cleaned_notebook_v3.ipynb
├── README.md
│
├── data/
│   ├── merged_trainval.csv (confidential)

```


## Installation & Dependencies
```python
pip install numpy pandas scipy scikit-learn scikit-image colour-science tensorflow matplotlib tqdm
```
Core dependencies:
- TensorFlow
- Scipy (NNLS)
- Colour-Science
- Skimage (ΔE2000)
- Pandas / NumPy


## Usage
1. Load dataset
    ```python
    df = pd.read_csv('merged_trainval.csv')
    ```

2. Run full K-fold
    ```python
    results = run_all_folds_v3(df, reflectance_cols, lab_cols, base_cols, wavelengths)
    ```

3. Check results
    ```python
    print(results['mean_DE'])
    print(results['pct_DE_lt_1'])
    print(results['pct_DE_lt_2'])
    ```


## Performance Metrics
| Metric        | Value                 |
| ------------- | --------------------- |
| Mean ΔE2000   | ~1.0–1.5              |
| % ΔE < 1.0    | 75-80%                |
| % ΔE < 2.0    | 90–98%                |
| Spectrum RMSE | Low (model-dependent) |


## Limitations

- Base pigments estimated from data — not physically exact

- Using mean contribution for base selection may miss rare pigments

- Colour conversion approximation depends on wavelength sampling

- NNLS reconstruction may be slow for large datasets

- Neural network does not enforce strict physical constraints

## Future Work
- Add reconstruction loss to training

    - L2 loss between true and reconstructed reflectance.

- Learn base spectra jointly via deep matrix factorization

    - (Instead of NNLS)

- Add multi-illuminant consistency

    - Convert spectra under D50, A, TL84, F11, etc.

- Use physics-based Kubelka–Munk mixing model

    - For more accurate coatings/inks.


## References

- CIELAB & ΔE2000 → Sharma et al. 2005

- Colour-Science Python Library → https://www.colour-science.org

- NNLS (Lawson–Hanson) → Solving least-squares with non-negative constraints

- Spectral Colorimetry → Wyszecki & Stiles, Color Science