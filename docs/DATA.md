# Dataset Provenance, Schemas, and Preprocessing Specifications

This document details the provenance, licensing, attributes, and preprocessing protocols for all empirical datasets utilized in the reproduction experiments.

---

## 1. Overview of Empirical Datasets (RQ2)

| Dataset | Primary Source | License | Total Rows ($m$) | Features ($n$) | Train Partition | Test Partition | Target Variable | Units | Target Characteristics |
|---|---|---|---|---|---|---|---|---|---|
| **Real Estate Valuation (Taiwan)** | UCI ML Repository (ID 477) | CC BY 4.0 | 414 | 6 | 331 (80%) | 83 (20%) | `Y house price of unit area` | 10,000 NTD / Ping | Unbounded, moderate spread |
| **California Housing** | scikit-learn / 1990 Census | Public Domain | 20,640 | 8 | 16,512 (80%) | 4,128 (20%) | `MedHouseVal` | $100,000 USD | Top-coded (capped at 5.0) |
| **King County House Sales** | OpenML (ID 42092) | CC0: Public Domain | 21,613 | 17 | 17,290 (80%) | 4,323 (20%) | `price` | USD | Heavily right-skewed |

---

## 2. Dataset Detailed Profiles

### 2.1 Real Estate Valuation (Taiwan)
- **Source**: UCI Machine Learning Repository, Dataset ID 477.
- **Reference**: Yeh, I-Cheng (2018). *Building Real Estate Valuation Models with Comparative Approach and Artificial Neural Networks*. Journal of Asian Architecture and Building Engineering.
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Input Features (6)**:
  1. `X1 transaction date`: Transaction date (e.g., 2013.250 = 2013 March).
  2. `X2 house age`: Age of the house in years.
  3. `X3 distance to the nearest MRT station`: Distance to closest metro station in meters.
  4. `X4 number of convenience stores`: Count of convenience stores within walking distance.
  5. `X5 latitude`: Geographic latitude in decimal degrees.
  6. `X6 longitude`: Geographic longitude in decimal degrees.
- **Excluded Features**: `No` (sample index dropped).
- **Target**: `Y house price of unit area` (10,000 New Taiwan Dollar / Ping, where 1 Ping = 3.3 square meters).

### 2.2 California Housing
- **Source**: Distributed via `sklearn.datasets.fetch_california_housing`.
- **Reference**: Pace, R. Kelley and Ronald Barry (1997). *Sparse Spatial Autoregressions*. Statistics & Probability Letters, 33(3):291–297.
- **License**: Public Domain.
- **Input Features (8)**:
  1. `MedInc`: Median income in block group ($10,000s).
  2. `HouseAge`: Median house age in block group.
  3. `AveRooms`: Average number of rooms per household.
  4. `AveBedrms`: Average number of bedrooms per household.
  5. `Population`: Block group population.
  6. `AveOccup`: Average number of household members.
  7. `Latitude`: Block group centroid latitude.
  8. `Longitude`: Block group centroid longitude.
- **Target**: `MedHouseVal` (Median house value for California districts in hundreds of thousands of dollars, $100,000s).
- **Domain Anomaly**: The target variable has an artificial ceiling at 5.00001 ($500,001), producing a noticeable cluster of top-coded values.

### 2.3 King County House Sales
- **Source**: OpenML Dataset ID 42092 (derived from King County Department of Assessments public property records).
- **License**: CC0: Public Domain.
- **Input Features (17)**:
  1. `bedrooms`: Number of bedrooms.
  2. `bathrooms`: Number of bathrooms.
  3. `sqft_living`: Square footage of the apartments interior living space.
  4. `sqft_lot`: Square footage of the land space.
  5. `floors`: Number of floors.
  6. `waterfront`: Binary flag for waterfront view (0/1).
  7. `view`: Index from 0 to 4 of how good the view of the property was.
  8. `condition`: Overall condition index from 1 to 5.
  9. `grade`: Classification grade by King County construction index (1 to 13).
  10. `sqft_above`: Square footage of interior living space above ground level.
  11. `sqft_basement`: Square footage of interior living space below ground level.
  12. `yr_built`: Year property was initially built.
  13. `yr_renovated`: Year property was last renovated.
  14. `lat`: Latitude coordinate.
  15. `long`: Longitude coordinate.
  16. `sqft_living15`: Living room area in 2015 (implies renovations).
  17. `sqft_lot15`: Lot size in 2015.
- **Excluded Features**: `id` (identifier), `date` (timestamp string), `zipcode` (high-cardinality postal code integer category).
- **Target**: `price` (Sale price in USD).
- **Domain Anomaly**: Strongly skewed right tail (min: $75,000; median: $450,000; max: $7,700,000). Relative tolerance thresholds are strictly necessary when evaluating error metrics.

---

## 3. Preprocessing Protocol

1. **Partitioning**: Stratified random splitting with `train_test_split(test_size=0.2, random_state=42, shuffle=True)`. Split indices are archived in `split.npz`.
2. **Feature Standardization**: Features are scaled using Z-score standardization:
   $$x_{ij} \leftarrow \frac{x_{ij} - \mu_{j,\text{train}}}{\sigma_{j,\text{train}}}$$
   Parameters $\mu_j$ and $\sigma_j$ (computed with $\text{ddof} = 0$) are fit exclusively on the training partition to prevent data leakage.
3. **Target Standardization**:
   $$y_i \leftarrow \frac{y_i - \mu_{y,\text{train}}}{\sigma_{y,\text{train}}}$$
   Training occurs on normalized $(X_{\text{std}}, y_{\text{std}})$. Test evaluation transforms predictions back to original physical units:
   $$\hat{y}_{\text{orig}} = \hat{y}_{\text{std}} \cdot \sigma_{y,\text{train}} + \mu_{y,\text{train}}$$
4. **Data Serialization**:
   - `train.bin`: Standardized training matrix encoded into `SGDCSR01` binary format for C++ execution.
   - `arrays.npz`: Standardized train/test arrays along with unstandardized test targets for Python metrics calculation.
   - `meta.json`: Dataset dimensions, column names, scaling parameters, and file checksums.
