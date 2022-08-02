import streamlit as st

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import io

from prophet import Prophet

# LOADING THE DATA:
@st.cache
def load_data():
    df = pd.read_csv('top17crimes_9col.csv')
    df['start_datetime_of_event'] = pd.to_datetime(df['start_datetime_of_event'])
    df = df.loc[~df['lat'].isna()]
    top_twenty = df
    return top_twenty

top_twenty = load_data()

# USER INPUT FOR DEMOGRAPHICS:
with st.sidebar:
    input_boro = st.selectbox('Select NYC borough:', ('Manhattan', 'Bronx', 'Queens', 'Brooklyn', 'Staten Island'), index = 0)
    
    if input_boro == 'Manhattan':
        boro = 'MANHATTAN'
    elif input_boro == 'Bronx':
        boro = 'BRONX'
    elif input_boro == 'Queens':
        boro = 'QUEENS'
    elif input_boro == 'Brooklyn':
        boro = 'BROOKLYN'
    else:
        boro = 'STATEN ISLAND'
    
    age = st.selectbox('Select age group:', ('<18', '18-24', '25-44', '45-64', '65+'))
    
    input_gender = st.selectbox('Select sex:', ('Female', 'Male'))
    
    if input_gender == 'Female':
        gender = 'F'
    else:
        gender = 'M'

    input_ethn = st.selectbox('Select ethnicity:', ('Black', 'White', 'White Hispanic', 'Asian/Pacific Islander', 'Black Hispanic', 'American Indian/Alaskan Native'))

    if input_ethn == 'Black':
        ethn = 'BLACK'
    elif input_ethn == 'White':
        ethn = 'WHITE'
    elif input_ethn == 'White Hispanic':
        ethn = 'WHITE HISPANIC'
    elif input_ethn == 'Asian/Pacific Islander':
        ethn = 'ASIAN / PACIFIC ISLANDER'
    else:
        ethn = 'AMERICAN INDIAN/ALASKAN NATIVE'
    
    input_date = st.date_input("Select date:", min_value = datetime.date(2006, 1, 1), max_value = datetime.date(2023, 12, 31))
    input_date = pd.Timestamp(input_date)

# SLICING THE DATA ACCORDING TO THE USER INPUT:
slice_ = top_twenty.loc[(top_twenty['boro'] == boro) & (top_twenty['victim_age'] == age) & (top_twenty['victim_sex'] == gender) & (top_twenty['victim_race'] == ethn)]
slice_ = slice_.drop(columns = ['description', 'boro', 'victim_age', 'victim_race', 'victim_sex'])
slice_['ymd'] = (slice_['start_datetime_of_event'].dt.strftime('%Y') + "-" +slice_['start_datetime_of_event'].dt.strftime('%m') + "-" +slice_['start_datetime_of_event'].dt.strftime('%d'))

slice_ymd = (slice_[['ymd', 'id']].groupby(slice_['ymd']).agg({'id': 'count'}).reset_index().copy())
slice_ymd = slice_ymd.reindex(columns={'id', 'ymd'})
slice_ymd = slice_ymd.rename(columns={'id' :'y', 'ymd': 'ds'})
slice_ymd['ds'] = pd.to_datetime(slice_ymd['ds'])

# FACEBOOK PROPHET FORECASTING:
m = Prophet(interval_width=0.95, daily_seasonality=True)
model = m.fit(slice_ymd)
future = m.make_future_dataframe(periods=1000,freq='D')
forecast = m.predict(future)

# STREAMLIT APP ELEMENTS OF OUTPUT:
if input_date < pd.Timestamp("2021-12-31 23:59:59"):
    try:
        h_crime_count = slice_ymd.loc[slice_ymd['ds'] == input_date]['y'].values[0]
        if h_crime_count== 1:
            st.markdown("## There was 1 crime reported on that day against an individual with the selected demographics.")
        else:
            st.markdown("## There were {} crimes reported on that day against individuals with the selected demographics.".format(h_crime_count))
        st.markdown("### The map below shows the total number of crimes in {} reported with the selected demographics.".format(input_date.strftime('%B, %Y')))
        st.map(slice_.loc[slice_['start_datetime_of_event'].dt.strftime('%Y-%m') == input_date.strftime('%Y-%m')])
    except:
        st.markdown("## There were no crimes reported on that day.")
elif input_date > pd.Timestamp("2021-12-31 23:59:59"):
    f_crime_count = round(forecast.loc[forecast['ds'] == input_date]['yhat'].values[0], 2)
    st.markdown("## There are {} crimes forecast on that day against individuals with the selected demographics.".format(f_crime_count))
