import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#--------------------------------------------------#   

st.header(" General Machine Learning Package")

#--------------------------------------------------#   
st.sidebar.image('img.svg', width=100)
st.sidebar.header("AutoStreamML")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=['csv', 'xlsx', 'json'], accept_multiple_files=False)

if uploaded_file is not None:
    @st.cache_data  
    def load_data(file):
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(file)

    df = load_data(uploaded_file)

#--------------------------------------------------#   

    st.sidebar.subheader("Preview Options")
    show_nrows = st.sidebar.slider('Number of rows to display:', min_value=5, max_value=len(df), step=1)
    show_ncols = st.sidebar.multiselect('Columns to display:', df.columns.to_list(), default=df.columns.to_list())

    st.subheader("Preview of the Dataset:")
    st.write(df[:show_nrows][show_ncols])

#--------------------------------------------------#   

    st.subheader("Exploratory Data Analysis:")
    
    st.write("#### Shape of Dataset:")
    st.write(df.shape)

    st.write("#### Descriptive Statistics:")
    st.write(df.describe())

    st.write("#### Missing Values:")
    st.write(df.isna().sum())

    st.write("#### Exploratory Data Analysis Plots:")
    tab1, tab2, tab3, tab4 = st.tabs(['Scatter Plot', 'Histogram Plot', 'Box Plot', 'Bar Plot']) 
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        with col1:
            x_column = st.selectbox('Select column on x axis: ',num_cols)
        with col2:
            y_column = st.selectbox('Select column on y axis: ',num_cols)
        with col3:
            size = st.selectbox('Select column for size: ', num_cols)
        with col4:
            color = st.selectbox('Select column to be color: ',df.columns)
        fig_scatter = px.scatter(df, x=x_column, y=y_column,color=color, size=size)
        st.plotly_chart(fig_scatter)
    with tab2:
        x_hist = st.selectbox('Select feature to drow histogram plot: ',num_cols)
        fig_hist=px.histogram(df,x=x_hist)
        st.plotly_chart(fig_hist)        
    with tab3:
        x_box = st.selectbox('Select feature to draw box plot: ', df.columns)
        fig_box = px.box(df, y=x_box)
        st.plotly_chart(fig_box)
    with tab4:      
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        x_bar = st.selectbox('Select feature to draw bar plot: ', cat_cols)
        fig_bar = px.bar(df, x=x_bar)
        st.plotly_chart(fig_bar)

#--------------------------------------------------#   

    st.subheader("Preprocessing for Dataset:")
    
    st.sidebar.subheader("Preprocessing Options")

    remove_columns = st.sidebar.multiselect('Select columns to remove:', df.columns.to_list())
    if remove_columns:
        df.drop(remove_columns, axis=1, inplace=True)

    missing_values_option = st.sidebar.radio('Missing Values Handling:', ['Delete Rows', 'Fill with Mean and Mode'])

    if missing_values_option == 'Fill with Mean and Mode':
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
    elif missing_values_option == 'Delete Rows':
        df.dropna(inplace=True)

    st.write(df.head(10))

    if df.isnull().sum().sum() == 0:
        st.success("No missing values remaining.")
    else:
        st.warning("There are still missing values in the data.")
    
    st.write("#### Shape of Dataset:")
    st.write(df.shape)

#--------------------------------------------------# 
#   
    st.sidebar.subheader("Target Variable ")
    st.write("#### The Target Variable of Dataset:")
    target_var = st.sidebar.selectbox('Select the target variable:', df.columns)
    st.write(target_var)

#--------------------------------------------------#   
   
    st.write("#### The machine learning model for Dataset:")
    if df[target_var].dtype == 'object':
        model_type = 'Classifier'
    else:
        model_type = 'Regressor'

    st.write(f"Selected Model Type: {model_type}")

#--------------------------------------------------#   
   
    st.sidebar.subheader("Encoding Options")
    st.write("#### Data Encoding:")
       
    hot_encode_columns = st.sidebar.multiselect("Select columns for One-Hot Encoding:", df.columns)
    label_encode_columns = st.sidebar.multiselect("Select columns for Label Encoding:", df.columns)

    if st.sidebar.button("Apply Encoding"):
        if hot_encode_columns:
            encoded_data = pd.get_dummies(df[hot_encode_columns], dtype=float, drop_first=True)
            df.drop(hot_encode_columns, axis=1, inplace=True)
            df = pd.concat([df, encoded_data], axis=1)
        
        if label_encode_columns:
            le = LabelEncoder()
            for col in label_encode_columns:
                df[col] = le.fit_transform(df[col])
        st.write(df)

#--------------------------------------------------#   

        st.write("### Correlation Heatmap:")
        figure = plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot(figure)

#--------------------------------------------------#   

        st.subheader("Training Machine Learning Model:")
        
        if model_type == 'Classifier':
            from pycaret.classification import *
            # init setup
            st.write('The Classifier')
            setup(data = df, target = target_var, session_id = 123)
            setup_df = pull()
            # model training
            st.info('This is the ML Experiment settings')
            st.dataframe(setup_df)
            
            best_model = compare_models()
            compare_df = pull()
            st.info('This is the ML Models')
            st.dataframe(compare_df)
            best_model

        elif model_type == 'Regressor':
            from pycaret.regression import *
            st.write('The Regressor')
            # init setup
            setup(data = df, target = target_var, session_id = 123)
            setup_df = pull()
            # model training
            st.info('This is the ML Experiment settings')
            st.dataframe(setup_df)
            
            best_model = compare_models()
            compare_df = pull()
            st.info('This is the ML Models')
            st.dataframe(compare_df)
            best_model