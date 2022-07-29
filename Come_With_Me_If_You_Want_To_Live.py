import streamlit as st

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import io

from prophet import Prophet

##########################################################################################

with st.sidebar:
    input_boro = st.selectbox(
         'Select NYC borough:',
         ('Manhattan',
          'Bronx',
          'Queens',
          'Brooklyn',
          'Staten Island'))
    
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
    
    age = st.selectbox(
         'Select age group:',
         ('<18',
          '18-24',
          '25-44',
          '45-64',
          '65+'))
    
    input_gender = st.selectbox(
         'Select sex:',
         ('Female',
          'Male'))
    
    if input_gender == 'Female':
        gender = 'F'
    else:
        gender = 'M'

    input_ethn = st.selectbox(
         'Select ethnicity:',
         ('Black',
          'White',
          'White Hispanic',
          'Asian/Pacific Islander',
          'Black Hispanic',
          'American Indian/Alaskan Native'))

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
    
    input_date = st.date_input(
        "Select date:",
        min_value = datetime.date(2006, 1, 1),
        max_value = datetime.date(2023, 12, 31))

##########################################################################################

@st.cache
def load_data():
    df = pd.read_csv('top17crimes_9col.csv')
    df['start_datetime_of_event'] = pd.to_datetime(df['start_datetime_of_event'])
    #df_fel_misd = df.loc[(df['offense_level'] == 'FELONY') | (df['offense_level'] == 'MISDEMEANOR')].copy()
    #twenty_crimes = df_fel_misd['description'].value_counts().head(17).index.to_list()
    top_twenty = df #df_fel_misd.loc[df['description'].isin(twenty_crimes)].copy()
    return top_twenty

top_twenty = load_data()

buffer = io.StringIO()
top_twenty.info(buf=buffer)
s = buffer.getvalue()
#st.text(s)

##########################################################################################

def df_slicer(boro,
              age,
              gender,
              ethn
             ):
    slice_ = top_twenty.loc[
        (top_twenty['boro'] == boro) &
        (top_twenty['victim_age'] == age) &
        (top_twenty['victim_sex'] == gender) &
        (top_twenty['victim_race'] == ethn)
    ]
    slice_ = slice_.drop(columns = ['description', 'boro', 'lat', 'lon', 'victim_age', 'victim_race', 'victim_sex'])
    
    slice_['ymd'] = (
    slice_['start_datetime_of_event'].dt.strftime('%Y') + "-" +
    slice_['start_datetime_of_event'].dt.strftime('%m') + "-" +
    slice_['start_datetime_of_event'].dt.strftime('%d')
    )
    
    slice_ymd = (
        slice_[['ymd', 'id']]
        .groupby(slice_['ymd'])
        .agg({'id': 'count'})
        .reset_index()
        .copy()
    )
    slice_ymd = slice_ymd.reindex(columns={'id', 'ymd'})
    slice_ymd = slice_ymd.rename(columns={'id' :'y', 'ymd': 'ds'})
    return slice_ymd

slice_ymd = df_slicer(boro, age, gender, ethn)

#buffer = io.StringIO()
#slice_ymd.info(buf=buffer)
#s = buffer.getvalue()
#st.text(s)

##########################################################################################

def tsa(df):
    m = Prophet(interval_width=0.95, daily_seasonality=True)
    model = m.fit(df)
    future = m.make_future_dataframe(periods=1000,freq='D')
    forecast = m.predict(future)
    return forecast

forecast = tsa(df_slicer(boro, age, gender, ethn))

##########################################################################################

if input_date < pd.Timestamp("2021-12-31 23:59:59"):
    try:
        input_date = input_date.strftime("%Y-%m-%d")
        h_crime_count = slice_ymd.loc[slice_ymd['ds'] == input_date]['y'].values[0]
        if h_crime_count== 1:
            st.markdown("## There was 1 crime reported on that day against individual with the selected demographics.")
        else:
            st.markdown("## There were {} crimes reported on that day against individuals with the selected demographics".format(h_crime_count))
    except:
        st.markdown("## There were no crimes reported on that day.")
elif input_date > pd.Timestamp("2021-12-31 23:59:59"):
    input_date = pd.to_datetime(input_date)
    f_crime_count = round(forecast.loc[forecast['ds'] == input_date]['yhat'].values[0], 2)
    st.markdown("## There are {} crimes forecast on that day against individuals with the selected demographics".format(f_crime_count))
    
    
    
    
    
