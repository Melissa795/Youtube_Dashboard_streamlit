# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 20:08:27 2024

@author: mely7
"""
# Import required libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# Define some functions

# This function is used to color in red the negative numbers on the df_agg_diff_final dataframe
def style_negative(v, props=''):
    """Style negative values in dataframe"""
    try:
        return props if v < 0 else None
    except:
        pass

# This function is used to color in green the positive numbers on the df_agg_diff_final dataframe
def style_positive(v, props=''):
    """Style positive values in dataframe"""
    try:
        return props if v > 0 else None
    except:
        pass
    
# This function is used to show some country on the individual video analysis
def audience_simple(country):
    """Show top represented countries"""
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    elif country == 'IT':
        return 'Italy'
    else:
        return 'Other'

# Load data
@st.cache_data
def load_data():    
    df_agg_sub = pd.read_csv('Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    # Make some feature engineering on the df_agg dataframe
    df_agg = pd.read_csv('Aggregated_Metrics_By_Video.csv').iloc[1:,:]
    df_agg.columns = ['Video','Video title','Video publish time','Comments added','Shares','Dislikes','Likes',
                          'Subscribers lost','Subscribers gained','RPM(USD)','CPM(USD)','Average % viewed','Average view duration',
                          'Views','Watch time (hours)','Subscribers','Your estimated revenue (USD)','Impressions','Impressions ctr(%)']
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'], format='%b %d, %Y')
    df_agg['Video publish time'] = df_agg['Video publish time'].dt.date
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
    df_agg['Avg_duration_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)
    df_agg['Engagement_ratio'] =  (df_agg['Comments added'] + df_agg['Shares'] +df_agg['Dislikes'] + df_agg['Likes']) /df_agg.Views
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    df_agg.sort_values('Video publish time', ascending = False, inplace = True)
    # Load other dataframe
    df_comments = pd.read_csv('Aggregated_Metrics_By_Video.csv')
    df_time = pd.read_csv("Video_Performance_Over_Time.csv")
    df_time['Date'] = pd.to_datetime(df_time['Date'], dayfirst=True, errors='coerce')
    df_time['Date'] = df_time['Date'].dt.date
    return df_agg, df_agg_sub, df_comments, df_time 


# Create dataframes from the function 
df_agg, df_agg_sub, df_comments, df_time = load_data()

#Engineering features

# Create a copy of df_agg dataframe
df_agg_diff = df_agg.copy()
# Select only the data relatives to the last 12 months
metric_date_12mo = (df_agg_diff['Video publish time'].max() - pd.DateOffset(months=12)).date()
# Calculate the median values for the numeric columns
median_agg = df_agg_diff[df_agg_diff['Video publish time'] >= metric_date_12mo].median(numeric_only=True)

# Merge the dataframe df_time and only the columns Video and Video publish time of the df_agg dataframe
df_time_diff = pd.merge(df_time, df_agg.loc[:,['Video','Video publish time']], left_on ='External Video ID', right_on = 'Video')
# Convert the column Date in a datetime format
df_time_diff['Date'] = pd.to_datetime(df_time_diff['Date'])
# Convert the Video publish time column in datetime format
df_time_diff['Video publish time'] = pd.to_datetime(df_time_diff['Video publish time'])
# Calculate the difference beetwen the Date and Video publish columns and the dt.days method return the number of days.
# The result is stored in a new column days_published
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

# Calculate the difference beetwen the max value from the Video publish time column and an offset of 12 months
date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months=12)
# Select only the data from Video publish time that are greater than or equal to date_12mo
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]

# Create a pivot table than calculate the mean, median, percentile 80 and 20.
# The lambda function is used to calculate the  percentile, then the reset_index method is used to 
# Reset the index of the dataframe such that days_published becomes a column
views_days = pd.pivot_table(df_time_diff_yr,index= 'days_published',values ='Views', aggfunc = ['mean', 'median', lambda x: np.percentile(x, 80),lambda x: np.percentile(x, 20)]).reset_index()
# Rename the colums
views_days.columns = ['days_published','mean_views','median_views','80pct_views','20pct_views']
# Filter the rows such that the dataframe includes only the data relatives to the first 30 days
views_days = views_days[views_days['days_published'].between(0,30)]
# Select the columns needed for the cumulative sum
views_cumulative = views_days.loc[:,['days_published','median_views','80pct_views','20pct_views']] 
# Calculate the cumulative sum
views_cumulative.loc[:,['median_views','80pct_views','20pct_views']] = views_cumulative.loc[:,['median_views','80pct_views','20pct_views']].cumsum()

# Create a boolean array which shows the columns that are numeric type (float64 or int64)
numeric_cols = np.array((df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64'))
# Subtract from the df_agg_diff df and only for the numeric columns, the median value then divides the obtained result for the median
df_agg_diff.iloc[:, numeric_cols] = (df_agg_diff.iloc[:, numeric_cols] - median_agg).div(median_agg).astype('float64')

# Buil dashboard
add_sidebar = st.sidebar.selectbox('Aggregate or Individual Video', ('Aggregate Metrics', 'Individual Video Analysis'))

# Build the dashboard for the aggregate metrics
if add_sidebar == 'Aggregate Metrics':
    st.write('Ken Jee YouTube Aggregated Data')
    # Select the columns to show
    df_agg_metrics = df_agg[['Video publish time','Views','Likes','Subscribers','Shares','Comments added','RPM(USD)','Average % viewed',
                             'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']]
    # Convert the Video publish time column in the datetime type 
    df_agg_metrics.loc[:, 'Video publish time'] = pd.to_datetime(df_agg_metrics['Video publish time'])
    # Calculate the difference beetwen the max value from the Video publish time column and an offset of 6 months
    metric_date_6mo = (df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=6))
    # Convert metric_date_6mo to a datetime format
    metric_date_6mo = pd.Timestamp(metric_date_6mo)
    # Calculate the difference beetwen the max value from the Video publish time column and an offset of 12 months
    metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=12)
    # Convert metric_date_12mo to a datetime format
    metric_date_12mo = pd.Timestamp(metric_date_12mo)
    # Select only the data from Video publish time that are greater than or equal to date_6mo and then calculate the median values (applied only for the numeric values)
    metric_medians6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo].median(numeric_only=True)
    # Select only the data from Video publish time that are greater than or equal to date_12mo and then calculate the median values (applied only for the numeric values)
    metric_medians12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo].median(numeric_only=True)
    
    # Create 5 columns on the streamlit page to shows the metrics in 2 rows
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]
    
    count = 0
    # Create a for loop that iterates over the indexes of the metric_medians6mo dataframe
    for i in metric_medians6mo.index:
        # Enter the context of the specific column using the 'with' statement
        with columns[count]:
            # Calculate the delta between the median values of 6 months and 12 months
            delta = (metric_medians6mo[i] - metric_medians12mo[i])/metric_medians12mo[i]
            # Display the metric value and its delta in the current column, in the format of % with two decimals
            st.metric(label= i, value = round(metric_medians6mo[i],1), delta = "{:.2%}".format(delta))
            # Increment the count to move to the next column
            count += 1
            # When the count is equal to 5, the first row is complete so we pass to the second row by setting count equal to 0
            if count >= 5:
                count = 0
    
    # # Creating a new column 'Publish_date' in df_agg_diff containing the video publishing dates, using the 'Video publish time' column
    df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x: x)
    # Selecting relevant columns for analysis 
    df_agg_diff_final = df_agg_diff.loc[:,['Video title','Publish_date','Views','Likes','Subscribers','Shares','Comments added','RPM(USD)','Average % viewed',
                             'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']]
    
    # Selecting numeric columns and creating a dictionary to format them as percentages
    numeric_columns = df_agg_diff_final.select_dtypes(include=['number'])
    df_agg_numeric_list = numeric_columns.columns.tolist()
    df_to_pct={}
    for i in df_agg_numeric_list:
        df_to_pct[i] = '{:.1%}'.format
    # Displaying the formatted DataFrame using a specific style, with different colors for negative and positive values
    st.dataframe(df_agg_diff_final.style.applymap(style_negative, props='color:red;').applymap(style_positive, props='color:green;').format(df_to_pct))

# If 'Individual Video Analysis' is selected from the sidebar
if add_sidebar == 'Individual Video Analysis':
    # Creating a tuple of video titles
    videos = tuple(df_agg['Video title'])
    # Selecting a video title using a selectbox
    video_select = st.selectbox('Pick a Video:', videos)
    
    # Filtering the DataFrame to contain only the selected video title
    agg_filtered = df_agg[df_agg['Video title'] == video_select]
    agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]
    agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)
    agg_sub_filtered.sort_values('Is Subscribed', inplace= True)
    
    # Creating a bar plot to visualize the relationship between views and subscription status by country
    fig = px.bar(agg_sub_filtered, x ='Views', y='Is Subscribed', color ='Country', orientation ='h')
    st.plotly_chart(fig)
    
    # Creating a line plot to compare cumulative views for the first 30 days
    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
    first_30 = first_30.sort_values('days_published')
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
                    mode='lines',
                    name='20th percentile', line=dict(color='purple', dash ='dash')))
    
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
                        mode='lines',
                        name='50th percentile', line=dict(color='black', dash ='dash')))
    
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
                        mode='lines', 
                        name='80th percentile', line=dict(color='royalblue', dash ='dash')))
    
    fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
                        mode='lines', 
                        name='Current Video', line=dict(color='firebrick',width=8)))
        
    fig2.update_layout(title='View comparison first 30 days',
                   xaxis_title='Days Since Published',
                   yaxis_title='Cumulative views')
    
    st.plotly_chart(fig2)