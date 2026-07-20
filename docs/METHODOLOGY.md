# Methodology

## 1. Introduction and Problem Formulation
Agricultural price prediction in developing economies is characterized by high volatility induced by climatic shocks, disjointed supply chains, and localized market variations. We formalize the prediction of agricultural commodity prices as a multi-variate time-series forecasting problem.

Let $P_{i,t}$ denote the mean retail price of commodity $i$ at time $t$. We model the price generation process as a function of temporal components $T$, climatic variables $C$, and commodity-specific fixed effects $E$:

$$P_{i,t} = f(T_{t}, C_{t}, E_{i}) + \epsilon_{t}$$

Where $\epsilon_{t}$ is the random error term.

## 2. Dataset and Preprocessing
The dataset integrates three primary sources across Tamil Nadu (2015-2024):
1. **Retail Price Data**: Monthly average prices for major agricultural commodities (Rice, Wheat, Tomato, Onion, Potato, Milk, Sugar).
2. **Climate Data**: District-level and state-aggregate monthly rainfall data (actual precipitation and deviation from normal).
3. **Vulnerability Data**: Composite climate vulnerability indices covering water stress, rainfall variability, and income margins.

### Preprocessing Steps
- **Missing Value Imputation**: Continuous variables (e.g., rainfall) were imputed using median values to preserve robustness against outliers, while deviations were zero-filled.
- **Normalization**: Continuous features were standardized using Z-score normalization: $z = \frac{x - \mu}{\sigma}$.
- **Encoding**: Categorical variables (commodities) were label-encoded. Time variables (month, year) were decomposed into cyclical seasonal markers.

## 3. Exploratory Data Analysis (EDA)
To establish the relationship between climate shocks and price volatility, we performed:
- **Pearson and Spearman Correlation Analysis**: Evaluated both linear and monotonic relationships between rainfall deviation and price spikes.
- **Vulnerability Assessment via PCA**: Principal Component Analysis (PCA) was employed to compress 4 vulnerability indicators into 2 principal components. K-Means clustering ($k=3$) was subsequently applied to stratify districts into Low, Medium, and High-risk zones.

## 4. Machine Learning Pipeline
To prevent temporal data leakage—a common flaw in agricultural prediction literature—we strictly employed **TimeSeriesSplit (5-fold)** for cross-validation. 

### Model Selection
We benchmarked the following algorithms:
1. **Naive Baseline**: Predicts the historical mean, serving as a lower-bound for model utility.
2. **Ridge Regression**: Linear model with L2 regularization to handle multicollinearity among climate features.
3. **Random Forest Regressor**: An ensemble of decision trees to capture non-linear interactions between seasonality and rainfall.
4. **XGBoost & LightGBM**: State-of-the-art gradient boosting frameworks optimized for tabular data. 

### Evaluation Metrics
Models were evaluated using:
- **R-squared ($R^2$)**: Proportion of variance explained.
- **Root Mean Square Error (RMSE)**: Penalizes large prediction errors.
- **Mean Absolute Percentage Error (MAPE)**: Provides a scale-independent error metric.

### Statistical Significance
To validate model superiority, we utilized the **Wilcoxon signed-rank test** ($p < 0.05$) to compare the distribution of absolute errors between the best-performing model and the alternatives.

## 5. Model Explainability
To transition the model from a "black box" to a decision-support tool, we employed **SHapley Additive exPlanations (SHAP)**. SHAP values calculate the marginal contribution of each feature to the final prediction, allowing us to quantify the exact premium added to commodity prices during monsoon deficits.
