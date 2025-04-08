
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from PIL import Image

gold_data = pd.read_csv('gld_price_data.csv')

gold_data.head()

gold_data.shape

gold_data.isnull().sum()

gold_data.describe()

corr = gold_data.corr(numeric_only=True)

plt.figure(figsize=(8,8))
sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.1f', annot_kws={'size':8}, cmap='plasma')

print(corr['GLD'])

sns.distplot(gold_data['GLD'],color='green')

"""In essence, this line creates a new DataFrame called x that contains all the columns from gold_data EXCEPT 'Date' and 'GLD'. These remaining columns will be used as the input features for the machine learning model.bold text.
A new variable called y that contains only the 'GLD' column. This is the target variable, which is the variable the machine learning model will try to predict.
"""

x = gold_data.drop(['Date','GLD'],axis=1)
y = gold_data['GLD']

print(x)

print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=2)

regressor = RandomForestRegressor(n_estimators=100)

regressor.fit(x_train,y_train)

prediction = regressor.predict(x_test)
print(prediction)

error_score = metrics.r2_score(y_test, prediction)
print("R squared error : ", error_score)

st.title('Gold Price Predictor')
img = Image.open('Gold_image.jpg')
st.image(img,width=200,use_column_width=True)
st.subheader('using randomforestregressor')
st.write(gold_data)
st.subheader('Model performance')
st.write('The accuracy of the model is',error_score)