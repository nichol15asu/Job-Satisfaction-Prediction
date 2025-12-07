# ===================================================================================
# 1. SETUP & INSTALLATION
# ===================================================================================
# We include all packages needed for the full experimentation
%pip install -q mlflow scikit-learn pandas matplotlib numpy openpyxl xgboost
dbutils.library.restartPython() 

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, make_scorer) # Import make_scorer for GridSearch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier # Requires xgboost library

# ===================================================================================
# 2. CONFIGURATION & SERVING FIX
# ===================================================================================
# NOTE: Replace 'nichol15@asu.edu' with your actual username if different
experiment_name = "/Users/nichol15@asu.edu/JobSatisfaction_Prediction_Finals"
mlflow.set_experiment(experiment_name)

# CRITICAL FIX: Forces Python 3.10 and explicitly lists all core ML libraries 
# to prevent "Container creation failed" or environment resolution errors during model serving.
serving_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.10", 
        "pip",
        {"pip": ["scikit-learn", "pandas", "mlflow", "cloudpickle", "xgboost", "openpyxl", "numpy"]}
    ],
    "name": "mlflow-env"
}

# ===================================================================================
# 3. LOAD DATA (Using your specific Workspace path)
# ===================================================================================

# The path you provided
FILE_PATH = "/Workspace/Users/nichol15@asu.edu/HREmployee_data.xlsx"

print(f"\n⬆️ Loading file: '{FILE_PATH}'")

try:
    # Use read_excel for .xlsx files
    df = pd.read_excel(FILE_PATH)
    print("✅ Excel file loaded successfully!")
    
except Exception as e:
    print(f"\n❌ FILE ERROR: Could not read '{FILE_PATH}'.")
    print("Double check that the file is physically located in your Workspace.")
    raise e

# Data Cleanup
TARGET_COLUMN = "JobSatisfaction"
COLUMNS_TO_DROP = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df.drop(columns=COLUMNS_TO_DROP, errors='ignore', inplace=True)

y = df[TARGET_COLUMN].copy()
X = df.drop(columns=[TARGET_COLUMN]).copy()

# === CRITICAL FIX FOR XGBOOST AND MULTICLASS CLASSIFICATION ===
# XGBoost requires class labels to start at 0 (i.e., 0, 1, 2, 3).
# We must subtract 1 from the JobSatisfaction target (1, 2, 3, 4) to map it to (0, 1, 2, 3).
y = y - 1
# =============================================================

# ===================================================================================
# 4. PREPROCESSING & SPLITTING
# ===================================================================================
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# FIX: After y = y - 1, 'y' is a numpy array, which uses np.unique() instead of y.unique()
CLASS_NAMES = [str(c) for c in sorted(np.unique(y))] 

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Disable autologging to prevent conflicts with our manual serving fix
mlflow.sklearn.autolog(log_models=False)

# Define the scoring metric for GridSearch (Multinomial Classification requires weighted F1)
f1_weighted_scorer = make_scorer(f1_score, average='weighted')

# ===================================================================================
# 5. TRAINING LOOP (FULL EXPERIMENTATION WITH GRIDSEARCH)
# ===================================================================================
print(f"🚀 Starting Full Experimentation Loop (9 Models)...")

best_estimators = {}
run_counter = 0

with mlflow.start_run(run_name="Main Experiment Parent"):
    
    # --- MODEL DEFINITIONS AND HYPERPARAMETER GRIDS (9 Models) ---
    # Note: These grids are based on your Experiment_Summary.md file and ensure 
    # at least 2 distinct values for up to 3 relevant hyperparameters.
    
    models_to_test = [
        # 1. LOGISTIC REGRESSION
        ("LogisticRegression", LogisticRegression(solver='saga', multi_class='multinomial', max_iter=2000, random_state=42), 
         {'clf__C': [0.1, 1.0, 10], 'clf__penalty': ['l2', 'l1'], 'clf__solver': ['saga']}),
        
        # 2. RANDOM FOREST
        ("RandomForest", RandomForestClassifier(random_state=42), 
         {'clf__n_estimators': [50, 100], 'clf__max_depth': [5, 10], 'clf__min_samples_leaf': [1, 5]}),
        
        # 3. CLASSIFICATION TREE
        ("DecisionTree", DecisionTreeClassifier(random_state=42), 
         {'clf__max_depth': [3, 7, 10], 'clf__min_samples_split': [2, 5], 'clf__criterion': ['gini', 'entropy']}),
         
        # 4. SUPPORT VECTOR MACHINE (SVC)
        # SVC is slow; limiting the grid
        ("SVC", SVC(random_state=42), 
         {'clf__C': [0.5, 1.0], 'clf__kernel': ['rbf', 'linear'], 'clf__gamma': ['scale', 'auto']}),
         
        # 5. NEURAL NETWORK (MLP)
        ("MLP", MLPClassifier(max_iter=500, random_state=42), 
         {'clf__hidden_layer_sizes': [(50,), (100,)], 'clf__activation': ['relu', 'tanh'], 'clf__alpha': [0.0001, 0.01]}),
         
        # 6. NAIVE BAYES (GaussianNB)
        # NB has very few hyperparameters, only testing two values for var_smoothing
        ("GaussianNB", GaussianNB(), 
         {'clf__var_smoothing': [1e-9, 1e-8]}),
         
        # 7. XGBOOST
        ("XGBoost", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), 
         {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.1, 0.01], 'clf__max_depth': [3, 6]}),
         
        # 8. K-NEAREST NEIGHBORS (kNN)
        ("kNN", KNeighborsClassifier(), 
         {'clf__n_neighbors': [3, 5, 7], 'clf__weights': ['uniform', 'distance'], 'clf__p': [1, 2]}),
    ]
    
    # Placeholder for Ensemble Model (needs best estimators from others)
    ensemble_estimators = [] 
    
    for model_name, model_estimator, param_grid in models_to_test:
        current_pipe = Pipeline([("prep", preprocess), ("clf", model_estimator)])
        
        run_counter += 1
        run_tag = f"Run {run_counter}: {model_name} (Tuning)"

        with mlflow.start_run(run_name=run_tag, nested=True):
            print(f"   Running GridSearch for: {model_name}...")
            
            # Use GridSearch for hyperparameter tuning
            grid_search = GridSearchCV(
                current_pipe, 
                param_grid, 
                cv=3, # Use 3-fold cross-validation
                scoring=f1_weighted_scorer, 
                n_jobs=-1,
                verbose=0
            )
            
            # This fit will now use the 0-indexed y_train, resolving the XGBoost error
            grid_search.fit(X_train, y_train)
            
            # The best model and score after tuning
            best_model = grid_search.best_estimator_
            best_f1 = grid_search.best_score_ # Use best_score_ from CV for robust comparison
            
            # --- LOGGING BEST RESULT ---
            y_pred_test = best_model.predict(X_test)
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            
            # Log Metrics
            mlflow.log_metric("test_f1_score", test_f1)
            mlflow.log_metric("cv_best_f1_score", best_f1)
            mlflow.log_params(grid_search.best_params_)
            
            # Store the best estimator (needed for the VotingClassifier later)
            if model_name in ["RandomForest", "LogisticRegression", "XGBoost"]: # Select three diverse models for the ensemble
                # NOTE: The stored estimator here is the full pipeline, which is correct
                ensemble_estimators.append((f'{model_name}_best', best_model))

            # Generate Signature
            # Note: The signature is generated using the 0-indexed output of the model
            signature = infer_signature(X_train, best_model.predict(X_train))
            
            # --- REGISTER MODEL ---
            mlflow.sklearn.log_model(
                best_model,
                "model",
                conda_env=serving_env, # The critical fix
                signature=signature,
                input_example=X_train.head(5),
                registered_model_name="workspace.default.jobsatisfaction_prediction"
            )
            
            print(f"   ✅ [{run_tag}] Best CV F1: {best_f1:.4f} | Test F1: {test_f1:.4f} - Model Registered.")
            
    # --- 9. ENSEMBLE MODEL (Voting Classifier) ---
    if len(ensemble_estimators) >= 2:
        # Extract only the classifier steps from the stored pipelines for the VotingClassifier
        
        # We also need to map the stored pipeline name to the classifier object
        voting_estimators = [(name, pipe.named_steps['clf']) for name, pipe in ensemble_estimators]

        ensemble_pipe = VotingClassifier(
            estimators=voting_estimators,
            voting='hard' # Starting with hard voting
        )
        
        # Create a new pipeline with the shared preprocessor and the ensemble classifier
        final_ensemble_pipe = Pipeline([("prep", preprocess), ("clf", ensemble_pipe)])

        run_counter += 1
        run_tag = f"Run {run_counter}: VotingClassifier"

        with mlflow.start_run(run_name=run_tag, nested=True):
            print(f"   Running Ensemble Model: {run_tag}...")
            
            # Simple fit for the Voting Classifier
            final_ensemble_pipe.fit(X_train, y_train)
            
            y_pred_ensemble = final_ensemble_pipe.predict(X_test)
            ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
            
            # Log Metrics
            mlflow.log_metric("test_f1_score", ensemble_f1)
            
            signature = infer_signature(X_train, final_ensemble_pipe.predict(X_train))
            
            # --- REGISTER MODEL ---
            mlflow.sklearn.log_model(
                final_ensemble_pipe,
                "model",
                conda_env=serving_env,
                signature=signature,
                registered_model_name="workspace.default.jobsatisfaction_prediction"
            )
            
            print(f"   ✅ [{run_tag}] Test F1: {ensemble_f1:.4f} - Model Registered.")

print("\n\n✅ Success! All 9 models and hyperparameter tuning have been executed and logged to MLflow.")
print("👉 Next Step: Review your MLflow runs to identify the absolute best model (likely Random Forest) and ensure its latest version is set for deployment.")
