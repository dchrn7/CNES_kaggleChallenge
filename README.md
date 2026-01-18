# ARIEL Data Challenge 2025 - Exoplanet Atmosphere Characterization

This repository contains the solution developed for the CNES 2025 AI Challenge. The objective of the competition was to classify simulated light spectra from the future ARIEL space mission to detect the presence of water (H2O) and clouds.

The proposed solution relies on a hybrid architecture combining Gradient Boosting (XGBoost) and Deep Learning (ResNet-CBAM), achieving a final score of **98.5%** on the private leaderboard.

## Project Structure

The project is divided into three distinct notebooks, corresponding to the modeling stages:

* **01_XgBoost.ipynb**: Statistical approach. This notebook contains feature engineering based on physical parameters and the training of the XGBoost model.
* **02_DeepResnet.ipynb**: Deep Learning approach. Implementation of a residual neural network (1D ResNet) with an attention mechanism (CBAM) and data fusion (spectrum + scalars).
* **03_Blending.ipynb**: Ensemble strategy. This notebook handles prediction fusion, correlation analysis, and threshold optimization for the final submission.

## Methodology

Our approach leverages the complementarity between physical feature engineering and automatic pattern extraction via neural networks.

### 1. XGBoost and Stellar Physics
Spectrum normalization removes certain absolute intensity information. To compensate, we injected explicit physical features (Star Temperature, Planet Radius, Surface Gravity) into a Gradient Boosting model. This model performs particularly well on tight orbits and sharp decision boundaries.

### 2. Deep ResNet with Attention (CBAM)
For the spectral analysis, we use a deep 1D CNN architecture (ResNet). The addition of CBAM (Convolutional Block Attention Module) allows the network to dynamically focus on relevant channels (absorption lines) and ignore instrumental noise. The model also integrates scalar metadata via a dense branch connected prior to the classification layer.

### 3. Hybridization and Error Analysis
The final performance relies on a weighted average of both models. Error analysis highlighted a significant uncertainty decorrelation:

* **XGBoost** shows performance degradation on cold stars (< 4000K), due to a covariate shift between the train and test sets regarding this parameter.
* **ResNet**, which analyzes signal morphology rather than raw physical values, remains robust on these cold stars. Conversely, it exhibits higher uncertainty on wide-orbit planets.

Combining both approaches corrects these respective biases and improves generalization.

## Results

The table below summarizes the performance obtained (mean F1-score metric). The significant gain on the private leaderboard confirms the robustness of the ensemble strategy.

| Model | Public Score | Private Score |
| :--- | :---: | :---: |
| Random Forest (Baseline) | 95.0% | - |
| XGBoost alone | 98.2 % | 98.2% |
| **Ensemble (Blending)** | **97.7%** | **98.5%** |

## Installation and Reproduction

To reproduce the results, the following dependencies are required:

```bash
pip install numpy pandas xgboost tensorflow scikit-learn matplotlib seaborn
