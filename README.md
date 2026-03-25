# Water Quality Analysis
### Predictive Modeling & Environmental Monitoring Dashboard
**PBL 2026 | CSE | Manipal University Jaipur**

> **Student:** Sanskriti Saxena | Reg No: 23FE10CSE00689  
> **Project Guide:** Dr. Shweta Sharma  
> **Live Site:** [pp-eta-orcin.vercel.app](https://pp-eta-orcin.vercel.app/)

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Loading & Exploration](#1-data-loading--exploration)
  - [2. Handling Missing Values](#2-handling-missing-values)
  - [3. Outlier Detection](#3-outlier-detection)
  - [4. Feature Engineering & Preprocessing](#4-feature-engineering--preprocessing)
  - [5. Visualizations (EDA)](#5-visualizations-eda)
  - [6. Model Building](#6-model-building)
  - [7. Model Evaluation & Comparison](#7-model-evaluation--comparison)
- [Results](#results)
- [Key Findings](#key-findings)
- [Literature Review](#literature-review)
- [Research Gap & Innovation](#research-gap--innovation)
- [How to Run](#how-to-run)
  

---

## Problem Statement

Water quality degradation poses significant risks to aquatic ecosystems and public health. Real-time monitoring and **accurate prediction of key parameters like Dissolved Oxygen (DO)** are critical for effective environmental management.

**Dissolved Oxygen (DO)** is one of the most important indicators of water health — it directly affects the survival of aquatic organisms. When DO drops below safe thresholds, it triggers hypoxic conditions that can cause large-scale die-offs of fish and other aquatic life.

The goal of this project is to:
- Analyse the Brisbane water quality dataset to understand environmental patterns
- Build machine learning models that can **accurately predict Dissolved Oxygen levels** from other measurable parameters
- Enable proactive water quality management through early prediction

---

## Dataset

| Property | Details |
|----------|---------|
| **Name** | Brisbane Water Quality Dataset |
| **Source** | Environmental monitoring sensors, Brisbane |
| **Records** | 30,894 rows |
| **Features** | 20 columns |
| **Time Range** | Multiple months (starting August 2023) |
| **Frequency** | 30-minute intervals |

### Features in the Dataset

| Feature | Description |
|---------|-------------|
| `Timestamp` | Date and time of measurement |
| `Record number` | Unique identifier |
| `Average Water Speed` | Speed of water flow (cm/s) |
| `Average Water Direction` | Direction of water flow (degrees) |
| `Chlorophyll` | Chlorophyll concentration (indicator of algae) |
| `Temperature` | Water temperature (°C) |
| `Dissolved Oxygen` |  **Target variable** — DO concentration (mg/L) |
| `Dissolved Oxygen (%Saturation)` | DO as percentage of saturation |
| `pH` | Acidity/alkalinity level |
| `Salinity` | Salt concentration (ppt) |
| `Specific Conductance` | Electrical conductivity (µS/cm) |
| `Turbidity` | Water clarity (NTU) |
| `[quality]` columns | Data quality flags for each sensor reading |

---

## Tech Stack

```
Python 3.x
├── pandas           — Data manipulation
├── numpy            — Numerical operations
├── matplotlib       — Static visualizations
├── seaborn          — Statistical visualizations
├── scipy            — Z-score outlier detection
├── scikit-learn     — ML models, preprocessing, metrics
├── xgboost          — Gradient boosting model
├── imbalanced-learn — SMOTE (imported, used for class balance exploration)
├── networkx         — Correlation network graph
└── mpl_toolkits     — 3D surface plotting
```

**Dashboard:** Power BI  
**Deployment:** Vercel (landing page)

---

## Project Structure

```
Water-Quality-Analysis/
│
├── brisbane_water_quality.csv      # Raw dataset
├── water_quality_analysis.ipynb    # Main Jupyter Notebook
├── README.md                       # This file
└── dashboard/
    └── dash.png                    # Power BI dashboard screenshot
```

---

## Methodology

### 1. Data Loading & Exploration

```python
df = pd.read_csv("brisbane_water_quality.csv")
df.shape  # (30894, 20)
```

**Why:** We first load the raw data and inspect its shape, column names, and first few rows. This gives us an initial understanding of what we're working with — how many records, what parameters are measured, and what the data looks like before any cleaning.

The dataset has **30,894 rows and 20 columns**, including both sensor readings and their corresponding quality flags.

---

### 2. Handling Missing Values

```python
missing_values = df.isnull().sum().sort_values(ascending=False)
df = df.dropna()
```

**Why:** Several columns had significant missing values — for example, `Dissolved Oxygen (%Saturation) [quality]` had 5,950 missing entries. Missing data can mislead models and produce inaccurate predictions.

**Decision — Drop vs. Impute:** We chose to **drop rows with missing values** (`df.dropna()`) rather than impute them (fill with mean/median) because:
- The dataset is large enough (30,894 rows) that dropping reduces it to ~19,149 rows — still a substantial and representative sample.
- Quality flags (the `[quality]` columns) were the most heavily missing, suggesting these are intentional absence of metadata, not sensor failures in the core measurements.
- Imputing these quality flags with mean values would be statistically meaningless.

After dropping, the dataset had **19,149 clean rows** with zero missing values across all columns.

---

### 3. Outlier Detection

```python
from scipy import stats

z_scores = stats.zscore(df.select_dtypes(include='number'))
outliers = (z_scores > 3).sum(axis=1)
outlier_rows = df[outliers > 0]
```

**Why:** Outliers can severely distort model training, especially for regression tasks. If extreme values are real sensor errors or anomalies, they introduce noise; if they're genuine events (like floods or pollution spikes), they may need special treatment.

**Method — Z-Score:** We used Z-score thresholding (`> 3 standard deviations`) because:
- It's a well-established, statistically grounded method for continuous data.
- It's interpretable: any value more than 3 standard deviations from the mean is flagged.

**Findings:** 1,049 rows were flagged as outliers. Many of these had quality flags of `1020.0` or `2010.0` (non-zero quality codes) — indicating the monitoring system itself marked these readings as potentially questionable. Notable outliers included `Average Water Speed` values above 58 cm/s, which were significantly higher than the typical range.

> **Note:** Outlier rows were identified and inspected. The decision to retain or remove them can be revisited depending on the use case (e.g., anomaly detection models might want to keep them).

---

### 4. Feature Engineering & Preprocessing

#### Dropping Quality Flag Columns

```python
df = df.drop(columns=['Dissolved Oxygen [quality]', 'Chlorophyll [quality]',
                      'Temperature [quality]', 'Dissolved Oxygen (%Saturation) [quality]',
                      'pH [quality]', 'Salinity [quality]',
                      'Specific Conductance [quality]', 'Turbidity [quality]'])
```

**Why:** The `[quality]` columns are metadata flags, not actual environmental measurements. They tell us about sensor reliability, not about the water itself. Including them in the feature set would:
- Introduce data leakage (quality flags may be assigned after DO is measured)
- Add noise with no predictive physical meaning

#### Defining Target and Features

```python
y = df['Dissolved Oxygen']
X = df.drop(columns=['Dissolved Oxygen', 'Timestamp', 'Record number'])
```

**Why:** `Dissolved Oxygen` is our prediction target. `Timestamp` and `Record number` are dropped because they are identifiers, not physical predictors — including them would cause the model to memorise row numbers rather than learn real patterns.

#### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Why:** We use an **80/20 split** — 80% of data for training, 20% held out for testing. This is a standard practice that:
- Provides enough training data for the model to learn patterns
- Keeps a fair unseen test set to evaluate generalisation (how well the model performs on new data)
- `random_state=42` ensures reproducibility — same split every time the code runs

#### Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why (and when we use it):** StandardScaler normalises features to have **mean = 0 and standard deviation = 1**. This is critical for **Linear Regression** because it is sensitive to the scale of features (e.g., Salinity is in the range 29–35, while Specific Conductance is in the 45–53 range — without scaling, the model could disproportionately weight features with larger numerical ranges).

> **Important:** Random Forest and XGBoost are tree-based models and are **scale-invariant** — they work on data splits, not distances. So they are trained on the *unscaled* `X_train` directly.

---

### 5. Visualizations (EDA)

#### Polar Plot — Water Direction & Speed
**Why:** Water flow is directional data. A polar (circular) plot is the most natural representation for direction (0–360°) paired with speed magnitude — standard Cartesian plots would misrepresent the cyclical nature of direction.

#### Violin Plot — Chlorophyll by Quality Flag
**Why:** Violin plots show the full distribution of values (not just median and quartiles like box plots). By grouping on quality flags, we can see if different quality ratings correspond to different chlorophyll distributions — helping validate data quality.

#### Correlation Heatmap
**Why:** Before modelling, it is essential to understand which variables are correlated with the target (Dissolved Oxygen) and with each other. This prevents multicollinearity surprises and guides feature selection.

Key correlation insights (from the heatmap):
- `Dissolved Oxygen (%Saturation)` has near-perfect correlation with `Dissolved Oxygen` — expected, since one is a percentage form of the other
- `Temperature` has a strong negative correlation with DO (warmer water holds less oxygen)
- `Salinity` also negatively correlates with DO (saltier water holds less oxygen)

#### Network Graph — Correlation Matrix
**Why:** Standard heatmaps can be hard to read for many variables. A network graph (threshold = 0.6) makes clusters of correlated variables visually obvious. Edges between nodes represent strong correlations.

#### Time Series Plot
**Why:** Water quality is temporally dynamic — temperature follows day/night cycles, seasonal patterns affect salinity and DO. Time series plots reveal trends, seasonality, and anomalies that inform whether time-based features should be engineered.

#### 3D Surface Plot — Temperature vs. Salinity vs. Time
**Why:** This allows us to see how two continuous environmental variables interact with each other over time, revealing seasonal patterns and spatial structure that 2D plots cannot capture.

#### Boxplots & Histograms
**Why:** Boxplots visually confirm outliers detected by Z-scores. Histograms show the distribution shape of each variable — useful for checking whether features are normally distributed or skewed (which affects some model assumptions).

---

### 6. Model Building

We trained three models of increasing complexity to compare their performance:

#### Model 1: Linear Regression (Baseline)

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
```

**Why start here:** Linear Regression is the simplest predictive model. It assumes a linear relationship between features and the target. We use it as a **baseline** — if more complex models don't significantly outperform it, the simpler model is preferred (Occam's Razor).

**Input:** Scaled features (required for linear models)

---

#### Model 2: Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

**Why:** Random Forest is an ensemble of decision trees that:
- Captures **non-linear relationships** that linear regression cannot
- Is **robust to overfitting** because it averages predictions from 100 trees
- Handles feature interactions naturally without manual feature engineering
- Provides **feature importance** scores, which help explain which variables matter most

**Parameters chosen:**
- `n_estimators=100` — 100 trees provide a good balance between accuracy and computation time
- `random_state=42` — reproducibility

---

#### Model 3: XGBoost Regressor

```python
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
```

**Why:** XGBoost (Extreme Gradient Boosting) is a state-of-the-art gradient boosting algorithm. Unlike Random Forest (parallel trees), XGBoost builds trees **sequentially** — each new tree corrects the errors of the previous one.

**Parameters chosen:**
- `n_estimators=200` — more trees than Random Forest because each tree is weaker (shallow); boosting benefits from more iterations
- `learning_rate=0.05` — small learning rate means the model learns slowly but more accurately (reduces overfitting)
- `max_depth=5` — limits how deep each tree can grow, preventing overfitting to noise

---

### 7. Model Evaluation & Comparison

**Metrics used:**

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **R² Score** | 1 - (SS_res / SS_tot) | Proportion of variance explained by the model (1.0 = perfect) |
| **RMSE** | √(mean squared error) | Average prediction error in the same units as DO (mg/L). Lower is better. |

**Why these two metrics:**
- **R²** gives the big picture — how much of the variation in DO does the model explain?
- **RMSE** gives practical significance — if RMSE = 0.027 mg/L, predictions are within ~0.027 mg/L of actual values on average, which is excellent for environmental monitoring thresholds.

---

## Results

| Model | R² Score | RMSE | Training Data |
|-------|---------|------|--------------|
| Linear Regression | 0.9971 (99.71%) | 0.0340 mg/L | Scaled |
| **Random Forest**  | **0.9982 (99.82%)** | **0.0267 mg/L** | Unscaled |
| XGBoost | 0.9973 (99.73%) | 0.0324 mg/L | Unscaled |

**Best Model: Random Forest** with the highest R² and lowest RMSE.

---

## Key Findings

**1. All models achieved >99.7% accuracy.** This means environmental parameters like temperature, salinity, and DO % saturation are very strong predictors of dissolved oxygen — the physics of water chemistry is well-captured in this dataset.

**2. Why is R² so high?**  
`Dissolved Oxygen (%Saturation)` is essentially a normalised version of `Dissolved Oxygen`. Their near-perfect correlation (r ≈ 0.99+) makes prediction almost trivially accurate. This is physically expected — one is derived from the other. In a real deployment, you would choose either as the target, not keep both as features.

**3. Random Forest outperforms XGBoost** here because:
- The dataset is large enough (~15,000 training samples) for bagging to work well
- The target relationships are smooth and non-noisy — boosting's strength is in correcting errors in complex/noisy datasets
- XGBoost may benefit from further hyperparameter tuning (e.g., `subsample`, `colsample_bytree`)

**4. Feature importance (from correlation analysis):**
- `Dissolved Oxygen (%Saturation)` — strongest predictor
- `Temperature` — strong negative relationship (Henry's Law: warmer water → less dissolved gas)
- `Salinity` — moderate negative relationship (saltwater holds less DO than freshwater)

**5. Temporal patterns matter.** Time series analysis shows DO fluctuates with daily cycles (photosynthesis during day increases DO) and seasonal changes — suggesting time-based features could further improve future models.

---

## Literature Review

| # | Authors | Study | Methods | Key Results |
|---|---------|-------|---------|-------------|
| 1 | Chukwuemeka et al. (2026) | Hybrid AI for water quality in Nigeria's Niger Delta | LSTM, XGBoost, K-Means, PCA | R²=0.95; dominant drivers: iron, nitrate, EC |
| 2 | Nefla et al. (2026) | Pollution source evaluation in China's Luoqing River | Random Forest, XGBoost, SHAP | AUC=0.96; nitrates and organics dominate |
| 3 | Zhang et al. (2026) | Ensemble models for pollution zone prediction in Chinese river basins | Random Forest, Gradient Boosting, Feature Importance | Effective spatial delineation of pollution zones |

**Common thread across literature:** Ensemble methods (Random Forest, XGBoost) consistently outperform single models for water quality prediction. SHAP values and feature importance analysis are emerging as best practices for model interpretability.

---

## Research Gap & Innovation

**What existing studies lack:**
- Most studies focus on classification (polluted/not polluted) rather than regression (predicting exact DO levels)
- Few studies provide interactive dashboards accessible to non-technical stakeholders
- Limited comparative analysis across multiple model families on the same dataset

**What this project adds:**
1. **Multi-model regression comparison** — systematic evaluation of three model families (Linear, Bagging, Boosting) on the same dataset
2. **Advanced EDA visualizations** — polar plots, 3D surface plots, and network graphs that go beyond standard scatter plots
3. **Interactive Power BI dashboard** — makes findings accessible to water quality managers, not just data scientists
4. **Focus on Brisbane** — a specific, well-monitored real-world dataset from a city with known water quality challenges due to urbanisation and seasonal flooding

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn networkx scipy
```

### Steps
1. Clone the repository:
```bash
git clone https://github.com/Sanskriti0109/Water-Quality-Analysis.git
cd Water-Quality-Analysis
```

2. Place `brisbane_water_quality.csv` in the root directory.

3. Open and run the Jupyter Notebook:
```bash
jupyter notebook water_quality_analysis.ipynb
```

4. Run all cells in order — the notebook is structured to flow from data loading → EDA → modelling → evaluation.



---

<p align="center">
  <b>Manipal University Jaipur | Department of Computer Science & Engineering</b><br>
  Project-Based Learning 2026
</p>
