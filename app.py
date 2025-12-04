import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1. Model Setup (Simulating the loaded, pre-trained model) ---
# This section simulates the process of loading your best model (Random Forest)
# and its required preprocessing components (StandardScaler, OneHotEncoder).
# In a real deployed app, you would load these objects from a file, but since 
# we can't share files, we train a dummy model internally to demonstrate the logic.

# Define the features based on your HR dataset and winning model
numerical_features = ['DailyRate', 'TotalWorkingYears', 'MonthlyIncome', 'YearsAtCompany']
categorical_features = ['Department', 'BusinessTravel', 'JobRole', 'MaritalStatus', 'Gender']

# Create a minimal dummy dataset that reflects the structure of your HR data
data = {
    'DailyRate': np.random.randint(100, 1500, 100),
    'TotalWorkingYears': np.random.randint(1, 30, 100),
    'MonthlyIncome': np.random.randint(1000, 15000, 100),
    'YearsAtCompany': np.random.randint(1, 20, 100),
    'Department': np.random.choice(['Sales', 'R&D', 'HR'], 100),
    'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], 100),
    'JobRole': np.random.choice(['Manager', 'Developer', 'Sales Executive', 'Research Scientist'], 100),
    'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], 100),
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'JobSatisfaction': np.random.randint(1, 5, 100) # Target variable (1, 2, 3, 4)
}
df = pd.DataFrame(data)

# Define the Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # Apply StandardScaler to numerical features (required for equal weighting)
        ('num', StandardScaler(), numerical_features),
        # Apply OneHotEncoder to categorical features (to convert text to numbers)
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Define the Final Model Pipeline (Random Forest - the winning model logic)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])

# Train the simulated model
X = df.drop('JobSatisfaction', axis=1)
y = df['JobSatisfaction']
model.fit(X, y)

# --- 2. Streamlit UI and Prediction Logic ---

st.set_page_config(page_title="Job Satisfaction Predictor", layout="centered")

st.markdown("<h1 style='text-align: center; color: #1e40af;'>Job Satisfaction Predictor (AI App Service)</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; color: #4b5563;'>
        Enter key employee metrics to receive a prediction on their job satisfaction level (1: Low to 4: High).
        <br><strong>Model used: Random Forest Classifier (Run 2).</strong>
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.header('Employee Data Input')

# --- Input Fields ---

st.sidebar.markdown('### Core Numerical Data')
# Highlight the most important feature found in your analysis
daily_rate = st.sidebar.number_input(
    '1. Daily Rate (Most Informative Feature)',
    min_value=100, max_value=1500, value=802, step=1,
    help="This is the most influential factor. Adjust to see the impact on prediction."
)
monthly_income = st.sidebar.number_input('Monthly Income', min_value=1000, max_value=20000, value=6500, step=100)
total_working_years = st.sidebar.slider('Total Working Years', min_value=0, max_value=40, value=10)
years_at_company = st.sidebar.slider('Years At Company', min_value=0, max_value=40, value=5)

st.sidebar.markdown('### Categorical Data')
department = st.sidebar.selectbox('Department', options=['Sales', 'R&D', 'HR', 'Support', 'Software'])
business_travel = st.sidebar.selectbox('Business Travel', options=['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
job_role = st.sidebar.selectbox('Job Role', options=['Developer', 'Sales Executive', 'Manager', 'Research Scientist', 'Laboratory Technician', 'HR'])
marital_status = st.sidebar.selectbox('Marital Status', options=['Married', 'Single', 'Divorced'])
gender = st.sidebar.selectbox('Gender', options=['Male', 'Female'])


# Function to collect inputs, transform them, and predict
def predict_satisfaction():
    # Construct the input DataFrame that the model expects
    input_data = pd.DataFrame({
        'DailyRate': [daily_rate],
        'TotalWorkingYears': [total_working_years],
        'MonthlyIncome': [monthly_income],
        'YearsAtCompany': [years_at_company],
        'Department': [department],
        'BusinessTravel': [business_travel],
        'JobRole': [job_role],
        'MaritalStatus': [marital_status],
        'Gender': [gender],
    })

    # Use the trained pipeline to preprocess and predict
    prediction = model.predict(input_data)[0]
    return prediction

# Prediction button and output display
if st.sidebar.button('Predict Satisfaction Level', key='predict_button'):
    prediction_int = predict_satisfaction()

    # Mapping numerical prediction to HR categories (1=Low, 4=Very High)
    satisfaction_map = {
        1: "Low (1)",
        2: "Medium (2)",
        3: "High (3)",
        4: "Very High (4)"
    }
    predicted_level = satisfaction_map.get(prediction_int, "Unknown")
    
    # Visual feedback based on prediction
    if prediction_int >= 3:
        bg_color = 'bg-green-100 border-green-500 text-green-800'
        emoji = '😊'
    elif prediction_int == 2:
        bg_color = 'bg-yellow-100 border-yellow-500 text-yellow-800'
        emoji = '😐'
    else:
        bg_color = 'bg-red-100 border-red-500 text-red-800'
        emoji = '😔'

    st.markdown(
        f"""
        <div class="p-4 rounded-lg font-bold mt-8 text-center {bg_color} border-2 shadow-lg">
            <h2 class="text-2xl">PREDICTED JOB SATISFACTION LEVEL</h2>
            <p class="text-5xl mt-2">{emoji} {predicted_level}</p>
            <p class="text-sm mt-1">Based on the deployed Random Forest model's analysis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
st.info("💡 Tip: Use the sidebar to adjust the employee's attributes, paying close attention to the Daily Rate, to see its impact on job satisfaction.")
