import pandas as pd
from io import BytesIO
from zipfile import ZipFile
import requests

# Download the ZIP file
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip'
response = requests.get(url)

# Extract the 'diabetic_data.csv' file from the ZIP
with ZipFile(BytesIO(response.content)) as zip_file:
    with zip_file.open('dataset_diabetes/diabetic_data.csv') as file:
        df = pd.read_csv(file)

# Display the first few rows of the dataset
print(df.head())
# Drop unnecessary columns
columns_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
df = df.drop(columns_to_drop, axis=1)

# Replace missing values with NaN
df = df.replace('?', pd.NA)

# Drop rows with missing values
df = df.dropna()

# Convert certain columns to appropriate data types
df['admission_type_id'] = df['admission_type_id'].astype('category')
df['discharge_disposition_id'] = df['discharge_disposition_id'].astype('category')
df['admission_source_id'] = df['admission_source_id'].astype('category')

# Perform one-hot encoding
categorical_columns = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
df_encoded = pd.get_dummies(df, columns=categorical_columns)

from sklearn.model_selection import train_test_split

# Split the data into features (X) and target variable (y)
X = df_encoded.drop('readmitted', axis=1)
y = df_encoded['readmitted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

# Filter the dataset based on the given criteria
df_filtered = df[(df['admission_type_id'] != 5) & (df['discharge_disposition_id'] != 11) & (df['discharge_disposition_id'] != 13) & (df['discharge_disposition_id'] != 14) & (df['discharge_disposition_id'] != 19) & (df['discharge_disposition_id'] != 20) & (df['discharge_disposition_id'] != 21)]

# Encode the target variable 'readmitted' using label encoding
le = LabelEncoder()
df_filtered['readmitted'] = le.fit_transform(df_filtered['readmitted'])

# Select the desired columns for the model
columns_to_keep = ['race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'insulin', 'change', 'diabetesMed', 'readmitted']
df_selected = df_filtered[columns_to_keep].copy()

# Perform one-hot encoding for the categorical features
categorical_columns = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'insulin', 'change', 'diabetesMed']

# Check if categorical_columns are present in df_selected.columns
missing_columns = [col for col in categorical_columns if col not in df_selected.columns]
if missing_columns:
    raise KeyError(f"The following columns are missing in the DataFrame: {missing_columns}")

df_encoded = pd.get_dummies(df_selected, columns=categorical_columns)

# Split the data into features (X) and target variable (y)
X = df_encoded.drop('readmitted', axis=1)
y = df_encoded['readmitted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Decode the predicted labels back to their original values
y_pred_decoded = le.inverse_transform(y_pred)

# Evaluate the model
print(classification_report(le.inverse_transform(y_test), y_pred_decoded))

import joblib

# Save the trained model to a file
joblib.dump(model, 'diabetes_model.joblib')

import streamlit as st
import joblib

# Load the trained model
model = joblib.load('diabetes_model.joblib')

# Define the prediction function
def predict_diabetes(data):
    # Reorder columns to match the training data
    data = data.reindex(columns=X_train.columns, fill_value=0)

    prediction = model.predict(data)
    return prediction

# Create the Streamlit app
def main():
    # Add a title and description
    st.title('Diabetes Readmission Prediction')
    st.write('Enter patient data to predict readmission.')

    # Add input fields for patient data
age = st.selectbox('Age', ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
gender = st.selectbox('Gender', ['Male', 'Female'])
admission_type = st.selectbox('Admission Type', ['Emergency', 'Urgent', 'Elective', 'Other'])

# Prepare the input data as a DataFrame
data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'admission_type_id': [admission_type]
})

# Make predictions and display the result
if st.button('Predict'):
    prediction = predict_diabetes(data)
    st.write('Predicted Readmission:', prediction)

# Run the app
if __name__ == '__main__':
    main()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

# Initialize the Random Forest Classifier
model = RandomForestClassifier()

# Initialize RandomizedSearchCV with the model and parameter grid
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,  # Number of parameter combinations to try
    scoring='accuracy',
    cv=5,  # Number of cross-validation folds
    random_state=42
)

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)

# Get the best model with the optimized hyperparameters
best_model = random_search.best_estimator_

# Make predictions on the test data using the best model
y_pred_best = best_model.predict(X_test)

# Calculate the accuracy of the best model
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracy score
print(f"Accuracy of the best model: {accuracy_best}")

import streamlit as st
import pickle

import os

model_file_path = r'c:\Users\user\Desktop\best_model.pkl'

# Check if the file exists
if os.path.exists(model_file_path):
    print("Model file exists")
else:
    print("Model file does not exist or the path is incorrect")

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the label encoder for gender and admission type
le_gender = LabelEncoder()
le_gender.fit(['Male', 'Female'])

le_admission_type = LabelEncoder()
le_admission_type.fit(['Emergency', 'Urgent', 'Elective', 'Other'])

# Preprocess the input data
input_data = pd.DataFrame({'Age': [age], 'Gender': [gender], 'Admission Type': [admission_type]})

# Encode categorical features
input_data['Gender'] = le_gender.transform(input_data['Gender'])
input_data['Admission Type'] = le_admission_type.transform(input_data['Admission Type'])

# Reorder columns to match the training data
input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

# Make predictions using the preprocessed input data
prediction = model.predict(input_data)

# Display the prediction
if prediction[0] == 0:
    st.write('The patient is not likely to be readmitted')
else:
    st.write('The patient is likely to be readmitted')

if __name__ == '__main__':
    main()
