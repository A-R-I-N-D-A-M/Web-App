import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import base64
import io
import pickle
import numpy as np
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.linear_model import LassoLarsIC,BayesianRidge
st.set_option('deprecation.showPyplotGlobalUse', False)
from PIL import Image





def main():
    st.set_page_config(page_title="Bike rental prediction", layout='wide')

    options=['Home','EDA','Visualization','Model building and evaluation','Prediction']
    choice=st.sidebar.selectbox('Choose the followings',options)

    if choice=='Model building and evaluation':
        st.subheader('Build **AutoML** models with 30 different algorithms and corresponding evaluation')
        uploaded_file = st.file_uploader('', type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            with st.beta_expander('Expand dataframe'):
                st.dataframe(df)

            X = df.drop(['cnt','instant','dteday'],axis=1)
            Y = df['cnt']

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
            reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
            models_train, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
            models_test, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

            st.subheader('2. Table of Model Performance on Test set')
            st.write(predictions_test)

            st.subheader('3. Plot of Model Performance (Test set)')

            with st.markdown('**R-squared**'):
                plt.figure(figsize=(9, 3))

                ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
                ax1.set(ylim=(0, 1))
                plt.xticks(rotation=90)
                st.pyplot(plt)


            with st.markdown('**RMSE (capped at 50)**'):

                plt.figure(figsize=(9, 3))

                ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
                plt.xticks(rotation=90)
                st.pyplot(plt)


    elif choice=='Prediction':
        st.subheader('Prediction for unseen data')
        st.sidebar.header('User Input Features')
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
        else:
            st.sidebar.subheader('Or input your features manually')
            def user_input_features():
                season=st.sidebar.selectbox('Season',np.arange(1,5))
                yr=st.sidebar.selectbox('Year',np.arange(0,2))
                month=st.sidebar.selectbox('Month',np.arange(1,13))
                holiday=st.sidebar.selectbox('Is Holiday',(0,1))
                weekday=st.sidebar.selectbox('Number of day',np.arange(1,8))
                workingday=st.sidebar.selectbox('Is workind day',(0,1))
                weathersit=st.sidebar.selectbox('Weather Number',np.arange(1,5))
                temp=st.sidebar.slider('Tempareture',0.05,0.86,0.20)
                atemp=st.sidebar.slider('Atemp',0.07,0.84,0.15)
                hum=st.sidebar.slider('Humadity',0.0,0.97,0.55)
                windspeed=st.sidebar.slider('Windspeed',0.02,0.5,0.08)
                casual=st.sidebar.slider('Casual',2,3410,50)
                registered=st.sidebar.slider('Registered',20,6946,5589)
                data = {'season': season,
                        'yr':yr ,
                        'mnth': month,
                        'holiday': holiday,
                        'weekday': weekday,
                        'workingday': workingday,
                        'weathersit':weathersit,
                        'temp':temp,
                        'atemp':atemp,
                        'hum':hum,
                        'windspeed':windspeed,
                        'casual':casual,
                        'registered':registered}
                features=pd.DataFrame(data,index=[0])
                return features

            input_df = user_input_features()

            st.subheader('User input features :')
            st.dataframe(input_df)


            if st.button('Start prediction'):
                model=pickle.load(open('LassoLarsIC.pkl','rb'))

                pred=model.predict(input_df)
                st.write('The prediction is :',pred)

    elif choice=='EDA':
        st.subheader('Explanatory data analysis')
        uploaded_file = st.file_uploader('', type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            with st.beta_expander('Expand dataframe'):
                st.dataframe(df)

            with st.beta_expander('Full profile information'):
                st_profile_report(ProfileReport(df,explorative=True))

            with st.beta_expander('Display basic summary'):
                st.write(df.describe().T)
            with st.beta_expander('Display data type'):
                st.write(df.dtypes)

    elif choice=='Visualization':
        st.subheader('Data Visualization')
        uploaded_file = st.file_uploader('', type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            with st.beta_expander('Expand dataframe'):
                st.dataframe(df)

            with st.beta_expander('Display bike rental along with time axis'):
                df2=df.copy(deep=True)
                df2.dteday=pd.to_datetime(df2.dteday)
                df2.set_index('dteday',inplace=True)
                plt.figure(figsize=(20,6))
                df2['cnt'].plot()
                st.pyplot()
                st.write('These shows that bike rental counts has seasonality and quite upwards trend.')
            with st.beta_expander('Display heatmap'):
                plt.figure(figsize=(10,6))
                sns.heatmap(df.corr(), annot=True)
                st.pyplot()
                st.write('There are some multicolliearity.')
            col1,col2=st.beta_columns(2)
            with col1:
                with st.beta_expander('Display total bike rental counts with different seasons'):
                    df.groupby('season')['cnt'].sum().plot(kind='bar')
                    st.pyplot()
                    st.write('Maximum bike rent was in season 3.')
                with st.beta_expander('Display total bike rental counts along with months and years'):
                    df.groupby(['mnth', 'yr'])['cnt'].sum().unstack().plot(kind='bar')
                    st.pyplot()
                    st.write('This plot shows the total bike rental count of every month of 2011 and 2012')
                    st.write('From MAY to OCTOBER the total bike rental count was high in every year and total rental in every month has increased from 2011 to 2012')
                with st.beta_expander('Display the pie chart of weathersit based on bike rental'):
                    plt.pie(df.groupby('weathersit')['cnt'].sum(), labels=['1', '2', '3'], explode=(0.05, 0, 0),
                            radius=1, autopct='%0.2f%%', shadow=True)
                    plt.tight_layout()
                    plt.legend(loc='upper left')
                    plt.axis('equal')
                    plt.show()
                    st.pyplot()
                    st.write('we have found total out of total bike rental count, 68.57% count was in "Clear, Few clouds, Partly cloudy, Partly cloudy" weatherand 30.27% was in " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist" weather.')
                with st.beta_expander('Display the outliers'):
                    num_var = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
                    for i in num_var:
                        sns.boxplot(y=i, data=df)
                        plt.title('Boxplot of ' + i)
                        plt.show()
                        st.pyplot()
                    st.write('We have found some outliers on the features - casual,windspeed and humidity')
            with col2:
                with st.beta_expander('Display the relationship between bike rental count and temperature'):
                    sns.scatterplot(x='temp', y='cnt', data=df)
                    st.pyplot()
                    st.write('We found almost linear relation between temp and count.')
                with st.beta_expander('Display the relationship between bike rental count and windspeed'):
                    sns.scatterplot(x='windspeed', y='cnt', data=df)
                    st.pyplot()
                    st.write('There is not much interpretation')
                with st.beta_expander('Display violine plot of seasons along with bike rental count'):
                    sns.violinplot(x=df.season, y=df.cnt)
                    st.pyplot()
                    st.write('Less count was in season 1 and it is right skewed and rest 3 seasons has not exactly any long tail and more or less season 2,3,4 have similar distribution')

    elif choice=='Home':

        image = Image.open('RentBike_25-09-17_02.jpg')
        st.image(image,use_column_width=True)
        st.title('Bike rental analysis, visualization, model building, evaluation and prediction in a single window')
        #st.subheader('Try the side bar options according to your choice')





if __name__=='__main__':
    main()


