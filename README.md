Job Satisfaction Prediction using Machine Learning

Project Summary

This project addresses the critical business problem of Mitigating High-Risk Turnover by predicting employee job satisfaction (levels 0-3) based on key internal HR and demographic metrics. The goal is to provide HR management with a proactive, data-driven tool to forecast areas of potential dissatisfaction and guide targeted policy interventions.

The analytical task is Supervised Classification. We established a performance baseline by training several models, and the final best model identified was the k-Nearest Neighbors (kNN) Classifier.

Target Variable: JobSatisfaction (0: Low, 1: Medium, 2: High, 3: Very High)
Final Best F1-Weighted Score: 0.2478
Key Business Insight: The success of kNN suggests that satisfaction is driven by localized "Peer Group" effects rather than simple global policy rules.

1. Project Components and Deliverables

Component

Status

Presentation Slides

Submitted on Canvas

[INSERT LINK TO PRESENTATION SLIDES HERE]

Databricks Workspace (MLflow)

Complete

https://dbc-b1609c98-c6b8.cloud.databricks.com/ml/experiments/85878062004456/runs?o=2335049932326856&searchFilter=&orderByKey=metrics.%60test_f1_score%60&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

Deployed Solution (Web App)

Complete

https://hr-predict-insight.lovable.app/

Source Code & Docs

Complete

This repository

2. Repository Structure

The repository is structured to enable full reproducibility and clear separation of concerns:

JobSatisfaction-Prediction-Project/
├── README.md                 <-- (This file) Project documentation.
├── environment.yml           <-- Conda environment for training reproducibility.
├── data/
│   └── HREmployee_data.xlsx  <-- Your raw data file.
├── notebooks/
│   └── job_satisfaction_experiment.py  <-- The complete analysis/training script (Includes MLflow Fix).
└── deployment/
    ├── predictor_app.html            <-- The final web application for prediction.
    └── requirements.txt              <-- Minimal dependencies for deployment server.


3. Environment Setup (Reproducibility)

To recreate the exact environment used for training and ensure the deployed model loads correctly (preventing version mismatch errors), follow these steps:

Create the Conda Environment:
The environment definition, including the specific scikit-learn version used for training, is in environment.yml.

conda env create -f environment.yml
conda activate job_satisfaction_env


Data Location:
Ensure your raw data file (HREmployee_data.xlsx) is placed inside the ./data/ folder.

4. Execution Workflow

A. Training and Tracking

The entire ML workflow is managed via the Python script:

Execute the notebooks/job_satisfaction_experiment.py script.

The script performs preprocessing, trains multiple models, and logs all metrics and the final complete model pipeline to MLflow, ensuring the model is saved with the correct dependency versions.

B. Deployment and Interaction

The model is deployed via the MLflow Model Registry as a REST API endpoint.

The interactive web application, ./deployment/predictor_app.html, communicates with the deployed API endpoint, allowing end-users to input employee features and receive a real-time prediction.

5. Deployment Details (For Reviewer)

These credentials are used by predictor_app.html to communicate with the model server.

Parameter

Value

PREDICTION_ENDPOINT

https://dbc-b1689c98-c6b8.cloud.databricks.com/serving-endpoints/job-satisfaction-endpoint/invocations

TOKEN_ID

dapi93ddc1ce92a774474231709b74d6b23

Note: For real-world projects, API tokens should always be stored securely as environment variables, not in public documentation.
