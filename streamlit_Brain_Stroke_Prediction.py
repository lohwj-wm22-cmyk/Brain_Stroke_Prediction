import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

# Load the trained models
# lr_model = load('logreg_model.joblib')
rf_model = load('rf_model.joblib')
# svm_model = load('svm_model.joblib')
scaler = load('scaler.pkl') 
# scaler = MinMaxScaler()

def dataset_preview_page():
    st.title('📊 Dataset Preview')
    st.header('Brain Stroke Prediction Dataset')
    
    # Link to the dataset
    dataset_link = 'https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset/data'
    st.write(f'You can download the full dataset from [Kaggle]({dataset_link}).')
    
    # Load a sample dataset for preview
    df = pd.read_csv('full_data.csv')  # Update this with the path to your sample data file
    st.write('Here is a preview of the dataset:')
    st.dataframe(df.head(20))

def prediction_page():
    st.title('🩺 Patient Health Prediction App')
    st.write('Fill in the details to predict the patient\'s health outcome.')

    # Input fields for user data
    gender = st.selectbox('Gender 👤', ['Male', 'Female', 'Other'])
    age = st.number_input('Age 🎂', min_value=0, max_value=120, value=25)
    hypertension = st.selectbox('Hypertension 🩺', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    heart_disease = st.selectbox('Heart Disease ❤️', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    ever_married = st.selectbox('Ever Married 💍', ['No', 'Yes'])
    work_type = st.selectbox('Work Type 🏢', ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
    residence_type = st.selectbox('Residence Type 🏡', ['Rural', 'Urban'])
    avg_glucose_level = st.number_input('Average Glucose Level (mg/dL) 🍬', min_value=0.0, step=0.1, value=100.0)
    bmi = st.number_input('BMI ⚖️', min_value=0.0, step=0.1, value=25.0)
    smoking_status = st.selectbox('Smoking Status 🚬', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    # When user clicks Predict button
    if st.button('Predict 🔮'):
        # Create a dictionary for the input
        input_data = {
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status]
        }


        # Convert the input to a DataFrame
        input_df = pd.DataFrame(input_data)

        # Define the model columns
        model_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                 'gender_Female', 'gender_Male', 'ever_married_No', 'ever_married_Yes',
                 'work_type_Govt_job', 'work_type_Private', 'work_type_Self-employed',
                 'work_type_children', 'Residence_type_Rural', 'Residence_type_Urban',
                 'smoking_status_Unknown', 'smoking_status_formerly smoked',
                 'smoking_status_never smoked', 'smoking_status_smokes']
        
        # Create a DataFrame to hold the encoded features
        encoded_input_df = pd.DataFrame(0, index=input_df.index, columns=model_columns)

        # Copy continuous variables
        encoded_input_df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']] = input_df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]
        
        # Hardcode categorical mappings : LIMITATIONS 
        categorical_data = {
            'gender': {'Male': 'gender_Male', 'Female': 'gender_Female'},
            'ever_married': {'Yes': 'ever_married_Yes', 'No': 'ever_married_No'},
            'work_type': {'Govt_job': 'work_type_Govt_job', 'Private': 'work_type_Private',
                        'Self-employed': 'work_type_Self-employed', 'children': 'work_type_children'},
            'Residence_type': {'Urban': 'Residence_type_Urban', 'Rural': 'Residence_type_Rural'},
            'smoking_status': {'smokes': 'smoking_status_smokes', 'formerly smoked': 'smoking_status_formerly smoked',
                            'never smoked': 'smoking_status_never smoked', 'Unknown': 'smoking_status_Unknown'}
        }

        # Populate categorical variables
        for col in categorical_data:
            # Set all columns to 0
            for column in categorical_data[col].values():
                encoded_input_df[column] = 0
            # Set the column for the specific input to 1
            value = input_df[col].iloc[0]
            encoded_input_df[categorical_data[col].get(value, '')] = 1

        # Ensure all columns are present
        encoded_input_df = encoded_input_df.reindex(columns=model_columns, fill_value=0)
        
        # st.write("encoded_input_df")
        # st.write(encoded_input_df)
        # print(encoded_input_df)

        # Check if scaler is fitted
        if scaler:
            # Scale the input data
            input_df_scaled = scaler.transform(encoded_input_df)

            # Predict using the Random Forest model
            rf_prediction = rf_model.predict(input_df_scaled)[0]

            # st.write("input_df_scaled")
            # st.write(input_df_scaled)
            
            # Display the prediction result
            st.success(f'🌟 Random Forest Prediction: {"At risk of stroke" if rf_prediction == 1 else "Not at risk"}')
        else:
            st.error("⚠️ Scaler not loaded properly. Please check the scaler file.")

def about_page():
    st.title('📚 About the Project')
    st.header('Stroke Prediction using Machine Learning Models')
    st.write("""
    This project aims to predict the risk of stroke based on patient data using a Random Forest model. 
    The dataset includes features such as age, gender, medical history (hypertension, heart disease), 
    lifestyle factors (smoking status, BMI), and others that help in predicting the likelihood of stroke.
    
    The model is trained using a stroke prediction dataset, and the goal is to assist healthcare professionals 
    in identifying high-risk individuals early on.
    """)

# Main function with sidebar navigation
def main():
    # Sidebar for navigation
    st.sidebar.title('🗂️ Navigation')
    menu_options = ['Prediction Page', 'Dataset Preview', 'About the Project']
    choice = st.sidebar.selectbox('Go to', menu_options)

    # Navigation based on user selection
    if choice == 'Prediction Page':
        prediction_page()
    elif choice == 'Dataset Preview':
        dataset_preview_page()
    elif choice == 'About the Project':
        about_page()

if __name__ == '__main__':
    main()
