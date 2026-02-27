import os               # file paths, directories
import joblib           # save/load trained models
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# create artifacts folder to store trained model, thresholds, metrics, and feature lists
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET = "readmitted_30d" # 0/1

# ----- Helpher: Youden's J threshold ----- 
def youden_threshold(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    J = tpr - fpr
    ix = int(np.argmax(J)) # select threshold that maximizes separation between positives and negatives
    return float(thr[ix]), float(J[ix]), float(tpr[ix]), float(1 - fpr[ix]) # returns threshold, J score, sensitivity, specificity

# ----- Optional feature engineering -----
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()