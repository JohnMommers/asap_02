
# setup
import streamlit as st
import openpyxl
import scipy
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import math

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

col1, col2, col3 = st.columns(3)
with col1:
    intensity_cutoff = st.number_input('Intensity cutoff: ', value=1)
    ru = st.number_input('Repeating unit: ', value=108.0028, format="%.5f")
    size_times = st.slider('Marker Size', 0.1, 1.0, 0.1)

with col2:
    er = st.number_input('Delta mass error: ', value=0.001, step=0.0005, format="%.5f")
    mz_delta_cutoff = st.number_input('Lowest delta mass: ', value=14.01, format="%.5f")

# load datafile
with st.sidebar:
    data_file = st.file_uploader('Upload ASAP file', type=['xlsx', 'csv'])

    if data_file is not None:
        df = pd.read_excel(data_file)
        df.columns = ['mass', 'intensity']
        df_filter = df[df['intensity'] >= intensity_cutoff]
        st.write(df_filter)
        st.write('INFO DATA:')
        st.write(df_filter.describe())

        mz = np.array(df_filter['mass'])
        mz_reshape = mz.reshape(len(mz),1)
        deltas = mz-mz_reshape
        deltas = deltas[deltas >= mz_delta_cutoff]

        # Generate sorted list
        mz_range = int (deltas.max() - deltas.min() )
        deltas_hist = np.histogram(deltas, bins=int (mz_range/(er)))
        a = np.append(deltas_hist[0],0)
        b = np.round(deltas_hist[1], decimals=4)
        data = {'delta mz':b, 'frequency':a}
        delta_hist = pd.DataFrame(data=data)
        delta_hist = delta_hist.sort_values('frequency', ascending=False)
        delta_hist = delta_hist.reset_index()
        delta_hist = delta_hist.drop(['index'], axis=1)
        main_delta = delta_hist['delta mz'][0]
        st.write('DELTAS', delta_hist)

csv = convert_df(df_filter)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='data_processed.csv',
    mime='text/csv', )

# plot mass versus end-groups
if data_file is not None:
    km = df['mass'] * (round(ru, 0) / ru)
    kmd = round(df['mass'], 0) - km

    mass_intensity = df['intensity']
    st.text('Kendrick Plot')
    fig = px.scatter(x=km, y=kmd, title='Kendrick Plot')
    fig.update_traces(marker=dict(size=mass_intensity * size_times,
                                  line=dict(width=1,
                                color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    st.plotly_chart(fig, use_container_width=True)

if st.button('Process data'):
    # Check for delta and calculate end-groups
    # st.write('df_filter =', len(df_filter))
    result_end_group = np.zeros(len(df_filter))
    result_mz1 = np.zeros(len(df_filter))
    result_mz2 = np.zeros(len(df_filter))
    result_delta = np.zeros(len(df_filter))
    result_detect = np.zeros(len(df_filter))

    for i in range(len(df_filter)):
        mz1 = df_filter.iloc[i][0]
        it1 = df_filter.iloc[i][1]
        for j in range(len(df_filter)):
            mz2 = df_filter.iloc[j][0]
            it2 = df_filter.iloc[j][1]
            d = mz2 - mz1
            if (d > (ru - er)) and (d < (ru + er)):
                result_end_group[j] = (mz2 / ru - math.floor(mz2 / ru)) * ru
                result_mz1[j] = mz1
                result_mz2[j] = mz2
                result_delta[j] = d

    # st.write(result_end_group)
    df_filter['end group'] = result_end_group
    df_filter['mz1'] = result_mz1
    df_filter['mz2'] = result_mz2
    df_filter['delta'] = result_delta
    st.write(df_filter)

    # plot mass versus end-groups
    x = df_filter['mz1']
    y = df_filter['end group']

    fig = px.scatter(df_filter, x='mz1', y='end group', size='intensity', width=5000, height=400)
    st.plotly_chart(fig, use_container_width=True)

    csv = convert_df(df_filter)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='data_processed.csv',
        mime='text/csv', )



