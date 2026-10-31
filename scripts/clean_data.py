import pandas as pd
import numpy as np
import os

# 1. Load the Excel dataset using correct relative path from 'scripts' folder
file_path = '../data/AI_Usage_and_Impact_on_Students_and_Professionals.xlsx'
df = pd.read_excel(file_path)

# --- STEP 1: Remove exact duplicates and ensure unique per User_ID ---
df = df.drop_duplicates()
if 'User_ID' in df.columns:
    df = df.drop_duplicates(subset=['User_ID'], keep='first')

print(f"Total rows after removing duplicates: {len(df)}")


# --- STEP 2: Convert text to numeric columns ---
numeric_cols = [
    'Age', 'Monthly_Income', 'Monthly_AI_Cost', 
    'AI_Usage_Hours_Per_Day', 'Time_Saved_Hours_Per_Week', 
    'Work_or_Study_Hours_Per_Day'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace('yrs', '', case=False, regex=False)
            .str.replace('hrs', '', case=False, regex=False)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')


# --- STEP 3: Standardize Categorical Columns ---

# Gender Standardization
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()
    gender_mapping = {
        'm': 'Male', 'male': 'Male',
        'f': 'Female', 'female': 'Female'
    }
    df['Gender'] = df['Gender'].map(gender_mapping).fillna('Other')

# Country Standardization
if 'Country' in df.columns:
    df['Country'] = df['Country'].astype(str).str.strip().str.title()
    df['Country'] = df['Country'].replace({
        'U.K.': 'United Kingdom', 
        'United States Of America': 'United States',
        'United Statesofamerica': 'United States'
    })

# AI_Tool Standardization
if 'AI_Tool' in df.columns:
    df['AI_Tool'] = df['AI_Tool'].astype(str).str.strip()
    ai_tool_mapping = {
        'Chat GPT': 'ChatGPT', 'chatgpt': 'ChatGPT', 'Chatgpt': 'ChatGPT', 'CHATGPT': 'ChatGPT',
        'Copilot': 'Microsoft Copilot', 'copilot': 'Microsoft Copilot',
        'Gemini': 'Google Gemini', 'gemini': 'Google Gemini',
        'claude': 'Claude', 'Claude': 'Claude'
    }
    df['AI_Tool'] = df['AI_Tool'].replace(ai_tool_mapping)

# Education_Level Standardization
if 'Education_Level' in df.columns:
    def standardize_education(val):
        if pd.isna(val):
            return val
        v = str(val).strip().lower()
        if any(x in v for x in ['master', 'ma', 'msc']):
            return "Master's"
        elif any(x in v for x in ['bachelor', 'ba', 'bsc']):
            return "Bachelor's"
        elif 'ph.d' in v or 'phd' in v:
            return 'PhD'
        elif v == 'hs' or 'high school' in v:
            return 'High School'
        elif 'undergrad' in v:
            return 'Undergraduate'
        else:
            return str(val).strip().title()

    df['Education_Level'] = df['Education_Level'].apply(standardize_education)


# --- STEP 4: Blank out impossible values (Set to NaN) ---
if 'Age' in df.columns:
    df.loc[df['Age'] > 80, 'Age'] = np.nan

if 'AI_Usage_Hours_Per_Day' in df.columns:
    df.loc[df['AI_Usage_Hours_Per_Day'] > 16, 'AI_Usage_Hours_Per_Day'] = np.nan

if 'Work_or_Study_Hours_Per_Day' in df.columns:
    df.loc[df['Work_or_Study_Hours_Per_Day'] > 16, 'Work_or_Study_Hours_Per_Day'] = np.nan

if 'Monthly_Income' in df.columns:
    df.loc[df['Monthly_Income'] > 50000, 'Monthly_Income'] = np.nan


# --- STEP 5: Add Helper / Derived Columns ---

# 1. Age_Group (16-20, 21-25, 26-35, 36-50, 51+)
if 'Age' in df.columns:
    age_bins = [0, 20, 25, 35, 50, 150]
    age_labels = ['16-20', '21-25', '26-35', '36-50', '51+']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)

# 2. Usage_Band (under 1 h, 1-2 h, 2-4 h, 4 h+)
if 'AI_Usage_Hours_Per_Day' in df.columns:
    usage_bins = [-1, 1, 2, 4, 24]
    usage_labels = ['under 1 h', '1-2 h', '2-4 h', '4 h+']
    df['Usage_Band'] = pd.cut(df['AI_Usage_Hours_Per_Day'], bins=usage_bins, labels=usage_labels)

# 3. Income_Band (Assigned Paid User / Free User based on Monthly_AI_Cost)
if 'Monthly_AI_Cost' in df.columns:
    df['Income_Band'] = df['Monthly_AI_Cost'].apply(lambda x: 'Paid User' if pd.notna(x) and x > 0 else 'Free User')


# --- STEP 6: Export Cleaned Data to CSV ---
output_dir = '../data/clean'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'ai_usage_clean.csv')

df.to_csv(output_path, index=False)
print(f"Cleaned data successfully exported to {output_path}")