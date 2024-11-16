# Story 6 : What Is The State of Food Security and Nutrition in the US

# load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# borrowed code from Tony Faser
def clean_cps_fss(df):
    """
    Clean and subset CPS Food Security Supplement data to key variables of interest,
    with additional calculation for a poverty indicator.
    
    Args:
        df: pandas DataFrame with original CPS FSS data
    
    Returns:
        DataFrame with cleaned and renamed columns, subset to key variables,
        and an added poverty indicator column.
    """
    state_fips = {
        1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
        8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia',
        12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
        18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana',
        23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan',
        27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri', 30: 'Montana',
        31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey',
        35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota',
        39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania',
        44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota',
        47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia',
        53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'
    }

    # Key variables to keep and their readable names
    columns_to_keep = {
        # Identifiers
        'HRHHID': 'household_id',
        'HRHHID2': 'household_id_2',
        
        # Demographic characteristics
        'PRTAGE': 'age',
        'PESEX': 'sex',
        'PEEDUCA': 'education',
        'PTDTRACE': 'race',
        'PEHSPNON': 'hispanic',
        'HEFAMINC': 'family_income',
        'HRNUMHOU': 'household_size',
        'HETENURE': 'housing_tenure',
        
        # Geography
        'GESTFIPS': 'state_fips',
        'GEREG': 'region',
        'GTMETSTA': 'metro_status',
        
        # Food Security Status
        'HRFS12M1': 'food_security_status',
        'HRFS12MC': 'child_food_security',
        'HRFS12M8': 'adult_food_security',
        
        # Food Spending
        'HETS8O': 'weekly_food_spending',
        'HETS8OU': 'usual_weekly_food_spending',
        
        # Program Participation  
        'HESP1': 'received_snap',
        'HESP6': 'received_school_lunch',
        'HESP7': 'received_school_breakfast',
        'HESP8': 'received_wic',
        
        # Weights
        'PWSUPWGT': 'person_supplement_weight',
        'HHSUPWGT': 'household_supplement_weight'
    }
    
    # Create subset with renamed columns
    df_clean = df[columns_to_keep.keys()].copy()
    df_clean = df_clean.rename(columns=columns_to_keep)
    
    # Value labels for categorical variables
    value_labels = {
        'food_security_status': {1: 'Food Secure', 2: 'Low Food Security', 3: 'Very Low Food Security'},
        'sex': {1: 'Male', 2: 'Female'},
        'hispanic': {1: 'Hispanic', 2: 'Non-Hispanic'},
        'housing_tenure': {1: 'Owned/Being Bought', 2: 'Rented', 3: 'Occupied without payment'},
        'region': {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'},
        'metro_status': {1: 'Metropolitan', 2: 'Non-metropolitan', 3: 'Not Identified'},
        'family_income': {
            1: 'Less than $5,000', 2: '$5,000 to $7,499', 3: '$7,500 to $9,999',
            4: '$10,000 to $12,499', 5: '$12,500 to $14,999', 6: '$15,000 to $19,999',
            7: '$20,000 to $24,999', 8: '$25,000 to $29,999', 9: '$30,000 to $34,999',
            10: '$35,000 to $39,999', 11: '$40,000 to $49,999', 12: '$50,000 to $59,999',
            13: '$60,000 to $74,999', 14: '$75,000 to $99,999', 15: '$100,000 to $149,999',
            16: '$150,000 or more'
        },
        'received_snap': {1: 'Yes', 2: 'No'},
        'received_school_lunch': {1: 'Yes', 2: 'No'},
        'received_school_breakfast': {1: 'Yes', 2: 'No'},
        'received_wic': {1: 'Yes', 2: 'No'},
        'education': {
            -1: 'Not_relevant', -2: 'Dont_know', -3: 'Refused_to_answer', -9: 'No_response',
            31: 'Less_than_1st_grade', 32: '1st-4th_grade', 33: '5th-6th_grade', 34: '7th-8th_grade',
            35: '9th_grade', 36: '10th_grade', 37: '11th_grade', 38: '12th_grade,_no_diploma',
            39: 'High_school_graduate_diploma_or_GED', 40: 'Some_college_no_degree',
            41: 'Associate_degree_occupational_vocational', 42: 'Associate_degree_academic_program',
            43: "Bachelors_degree", 44: "Masters_degree", 45: 'Professional_school_degree_MD_DDS_DVM_etc',
            46: 'Doctorate_degree_PhD_EdD'
        }
    }
    
    # Apply value labels
    for col, val_map in value_labels.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(val_map).fillna(df_clean[col])
            
    # Convert weights by dividing by 10000
    weight_cols = ['person_supplement_weight', 'household_supplement_weight']
    for col in weight_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col] / 10000
    
    # Create a state column
    df_clean['state'] = df_clean['state_fips'].map(state_fips)
    
    # Define poverty income threshold categories
    poverty_income_levels = [
        'Less than $5,000', '$5,000 to $7,499', '$7,500 to $9,999', '$10,000 to $12,499',
        '$12,500 to $14,999', '$15,000 to $19,999', '$20,000 to $24,999', '$25,000 to $29,999',
        '$30,000 to $34,999', '$35,000 to $39,999'
    ]
    
    # Create poverty indicator based on income and program participation
    df_clean['poverty_indicator'] = df_clean['family_income'].apply(lambda x: 1 if x in poverty_income_levels else 0)
    
    # Add to poverty indicator if received benefits (any program participation marked 'Yes')
    program_columns = ['received_snap', 'received_school_lunch', 'received_school_breakfast', 'received_wic']
    for col in program_columns:
        df_clean['poverty_indicator'] = df_clean.apply(lambda row: 1 if row[col] == 'Yes' else row['poverty_indicator'], axis=1)
    
    return df_clean

# main
def main():
    cps_url = 'https://www2.census.gov/programs-surveys/cps/datasets/2023/supp/dec23pub.csv'
    cps_raw_data = pd.read_csv(cps_url)
    cps_data = clean_cps_fss(cps_raw_data)
    cps_data.to_csv('Resources/cps_data.csv')


if __name__=="__main__":
    main()

