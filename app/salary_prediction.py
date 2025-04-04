import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('G:\\FIAP\\3_Arquitetura_ML_e_Aprendizado\\tech_challenge\\pkl_files\\regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

regressor_loaded = regressor['model']
le_country = regressor['le_country']
le_education = regressor['le_education']
le_age = regressor['le_age']
le_industry = regressor['le_industry']
ed_co = regressor['education_country']
co_ind = regressor['country_industry']
year_age = regressor['yearcode_age']
year_ind = regressor['yearcode_industry']
co_years = regressor['country_yearscode']
ed_years = regressor['education_yearscode']
co_age = regressor['country_age']

st.set_page_config(layout="wide")

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Previsão de Remuneração Pesquisa Stackoverflow 2024</h1>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

countries_list = ['Other','United States of America','Germany','UK','Ukraine','India','France','Canada','Brazil','Spain','Italy','Netherlands','Australia','Sweden','Poland','Switzerland','Austria']

education_list = ['Post Grad', 'Master’s degree', 'Less than a Bachelors','Bachelor’s degree']

age_list = ['25-34 years old','35-44 years old','18-24 years old','45-54 years old','55-64 years old','Under 18 years old','65 years or older',
'Prefer not to say']

industry_list = ['Software Development','Other:','Fintech','Internet, Telecomm or Information Services',"Banking/Financial Services",'Healthcare','Manufacturing','Retail and Consumer Services','Government','Media & Advertising Services','Higher Education','Transportation, or Supply Chain','Computer Systems Design and Services','Energy','Insurance']


country = col1.selectbox("País", [""] + sorted(countries_list))

education = col2.selectbox("Nível de Educação", [""] + sorted(education_list))

experience = col3.number_input('Experiência', min_value = 0.5)

age = col4.selectbox('Faixa Etária',[""] + sorted(age_list) )

industry = col5.selectbox("Indústria", [""] + sorted(industry_list))

if st.button('Previsão'):
    input_data = [[country, education, experience, age, industry]]
    input_data_transformed = input_data.copy()
    input_data_transformed[0][0] = le_country.transform([input_data[0][0]])[0]
    input_data_transformed[0][1] = le_education.transform([input_data[0][1]])[0]
    input_data_transformed[0][3] = le_age.transform([input_data[0][3]])[0]
    input_data_transformed[0][4] = le_industry.transform([input_data[0][4]])[0]

    education_country = f'{input_data_transformed[0][1]}_{input_data_transformed[0][0]}'
    try:
        education_country_encoded = ed_co.transform([education_country])[0]

    except KeyError:
        education_country_encoded = ed_co._mean

    country_industry = f'{input_data_transformed[0][0]}_{input_data_transformed[0][4]}'
    try:
        country_industry_encoded = co_ind.transform([country_industry])[0]
    except KeyError:
        country_industry_encoded = co_ind._mean

    yearscodepro_age = f'{input_data_transformed[0][2]}_{input_data_transformed[0][3]}'
    try:
        yearscodepro_age_encoded = year_age.transform([yearscodepro_age])[0]
    except KeyError:
        yearscodepro_age_encoded = year_age._mean

    #YearsCodePro_Industry	
    yearscodepro_industry = f'{input_data_transformed[0][2]}_{input_data_transformed[0][4]}'
    try:
        yearscodepro_industry = year_ind.transform([yearscodepro_industry])[0]
    except KeyError:
        yearscodepro_industry = year_ind._mean

    #Country_YearCodePro
    country_yearscodepro = f'{input_data_transformed[0][0]}_{input_data_transformed[0][2]}'
    try:
        country_yearscodepro = co_years.transform([country_yearscodepro])[0]
    except KeyError:
        country_yearscodepro = co_years._mean	

    #EdLevel_YearsCodePro
    education_yearcodepro = f'{input_data_transformed[0][1]}_{input_data_transformed[0][2]}'
    try:
        education_yearcodepro = ed_years.transform([education_yearcodepro])[0]
    except KeyError:
        education_yearcodepro = ed_years._mean	

    #Country_Age
    country_age = f'{input_data_transformed[0][0]}_{input_data_transformed[0][3]}'
    try:
        country_age = co_age.transform([country_age])[0]
    except KeyError:
        country_age = co_age._mean	

    input_data_transformed[0].append(education_country_encoded)
    input_data_transformed[0].append(country_industry_encoded)
    input_data_transformed[0].append(yearscodepro_age_encoded)
    input_data_transformed[0].append(yearscodepro_industry)
    input_data_transformed[0].append(country_yearscodepro)
    input_data_transformed[0].append(education_yearcodepro)
    input_data_transformed[0].append(country_age)

    input_data_transformed = np.array(input_data_transformed, dtype=float)

    y_pred = regressor_loaded.predict(input_data_transformed)

   
    st.success(f'O salário para uma pessoa de {country}, com o nível de educação {education}, com experiência de {experience} anos e que trabalha em {industry} é de US$ {y_pred[0]:,.2f}')
