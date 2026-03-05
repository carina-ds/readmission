🏥 **30-Day Hospital Readmission Risk Model**

An end-to-end, production-ready machine learning pipeline to predict 30-day hospital readmissions using structured patient journey data.

This project demonstrates:
- Robust preprocessing
- Model comparison and selection
- Threshold optimization
- Probability calibration
- Explainability
- Production packaging and monitoring readiness
- Automated model monitoring
- Real-time API deployment

**Tech Stack:**
- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborm
- joblib
- FastAPI
- logging

🎯 **Problem Statement**

Hospital readmissions are costly and often preventable. Predicting high-risk patients enables:
- Targeted care management
- Reduced financial penalties
- Improved patient outcomes
- More efficient resource allocation

The objective to predict: "readmitted_30d"
- A binary outcome indicating whether a patient is readmitted within 30 days of discharge

📊 **Dataset Overview**

Source: healthcare_patient_journey.csv
Features include:
- Demographics (age, gender)
- Clinical factors (chronic_condition, complications)
- Utilization (length_of_stay_days, procedures_count)
- Operational metrics (wait_time_min)
- Cost (total_cost_€)
- Satisfactor Score

🏗 **Pipeline Architecture**

The modeling workflow includes:
1. Data loading
2. Feature engineering
3. Preprocessing
4. Train/test split (stratified)
5. Model training & hyperparameter tuning
6. Champion model selection
7. Threshold optimization
8. Calibration
9. Explainability
10. Production packaging
11. Drift & performance monitoring
12. FastAPI deployment
 
**Preprocessing**

- Custom Winsorization
  - Extreme values were capped at the 5th and 95th percentiles using a custom Winsorizer transformer.
  - Prevents extreme outliers from distorting linear models
  - Preserves sample size (no row removal)
  - Improves numerical stability
- Feature Engineering
  - A modular EngineerFeatures transformer is included in the pipeline
  - Clean separation of logic
  - Reproducible feature generation
  - Production consistency
- Model-Specific Pipelines
  - Linear Model Pipeline
    - Winsorization
    - Median imputation
    - Scaling
    - One-hot encoding (sparse)
  - Tree-Based Pipeline
    - Median imputation
    - One-hot encoding (dense)
    - No scaling

**Model Development**

- Model A: Logistic Regression (Elastic Net)
  - Solver: saga
  - Class-weighted
  - Hyperparameter tuning via RandomizedSearchCV
  - Scoring metric: Average Precision (PR-AUC)
  - Why PR-AUC?
    - Better metric for imbalanced data
    - Focuses on positive case detection
- Model B: HistGradientBoosting
  - Gradient boosting classifier
    - Optimized with:
      - Learning rate tuning
      - Tree depth search
      - Leaf size tuning
      - L2 regularization
      - Binning strategy
    - Scoring metric: Average Precision (PR-AUC)
- Champion Selection
   | Model                | ROC-AUC    | PR-AUC     | Brier      |
   | -------------------- | ---------- | ---------- | ---------- |
   | Logistic Regression  | 0.7207     | 0.4732     | 0.2069     |
   | HistGradientBoosting | **0.7248** | **0.4871** | **0.2050** |
  - Champion Model: HistGradientBoosting (highest PR-AUC)

**Threshold optimization**

- Used Youden’s J statistic: _J=Sensitivity−(1−Specificity)_
- This balances:
  - Detecting readmissions (recall)
  - Avoiding excessive false positives
- Resulting test performance:
  - Recall (readmitted): 0.688
  - Precision (readmitted): 0.416
  - Accuracy: 0.700
- Reflects a healthcare-appropriate tradeoff: prioritize catching high-risk patients

**Calibration**

- Healthcare models must produce reliable probabilities.
- Raw tree-based models are often:
  - Overconfident
  - Poorly calibrated
- Code: CalibratedClassifierCV(method="isotonic", cv=3)
- Results:
  | Model      | Brier Score |
  | ---------- | ----------- |
  | Raw        | 0.2050      |
  | Calibrated | **0.1531**  |
  - Significant improvement in probability reliability.
- Calibration curve demonstrates strong alignment between:
  - Predicted risk
  - Observed readmission rate
- Calibration reference file saved for monitoring drift

**Model Explainability**

- Permutation Importance
  - Top drivers of readmission risk:
    - complications
    - chronic_condition
    - age
    - length_of_stay_days
    - total_cost_€
  - Key insights:
    - Clinical severity dominates prediction
    - Utilization intensity is predictive
    - Some operational variables had negligible or negative contribution
- Partial Dependence Plots
  - Used to visualize marginal feature impact on:
    - Complications
    - Chronic condition
    - Age

**Production Packaging**

- Final artifact includes:
  {
  "model": calibrated_pipeline,
  "threshold": optimal_threshold,
  "features": feature_list,
  "model_name": "hgb"
  }
  saved as: readmission_production.joblib
- Enables:
  - Consistent scoring
  - Threshold-based decisions
  - Deployment in batch or real-time systems
 
**Future Improvements**

- Remove features with negative permutation importance
- Add SHAP for local explainability
- Add fairness analysis across demographics
- Implement automated drift alerts
- A/B test intervention thresholds

**Automated Model Monitoring**

- Includes a production-grade monitoring script designed to run daily
- Monitors:
  - Data drift (PSI)
  - Performance degradation
  - Calibration drift
  - Threshold degradation
  - Population risk shift
 
**Real-Time Deployment (FastAPI)**

- Model is deployable as a REST API using FastAPI







