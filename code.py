# -*- coding: utf-8 -*-
"""

1. Business Understanding:
The first step is to thoroughly understand the business elements and challenges that Data Science aims to solve or improve.

2. Data Understanding:
This phase aims to precisely determine the data to be analyzed, identify the quality of available data, and establish the connection between the data and its business meaning. Since Data Science relies solely on data, business problems related to existing data, whether internal or external, can be addressed through Data Science.

3. Data Preparation:
This phase involves activities related to constructing the specific dataset for analysis from raw data. It includes sorting the data based on selected criteria, cleaning the data, and especially recoding the data to make it compatible with the algorithms to be used.

4. Modeling:
This is the core phase of Data Science. Modeling includes the selection, parameterization, and testing of different algorithms, as well as their sequencing, which forms a model. This process starts with descriptive modeling to generate knowledge by explaining why things happened. It then becomes predictive by explaining what will happen and finally prescriptive by optimizing a future situation.

5. Evaluation:
Evaluation aims to verify the model(s) or knowledge obtained to ensure they meet the objectives set at the beginning of the process. It also contributes to the decision of deploying the model or, if necessary, improving it. At this stage, the robustness and accuracy of the obtained models are tested, among other things.

6. Deployment:
This is the final step of the process. It involves putting the obtained models into production for end users. The goal is to package the knowledge obtained through modeling in an appropriate form and integrate it into the decision-making process.

Deployment can range from simply generating a report describing the obtained knowledge to implementing an application.



Importing Libraries :
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



"""Loading Dataset :"""

Dataset = pd.read_csv('C:/Users/KHAIRI/Downloads/SonderListings.csv')

Dataset.head(2)

"""Printing the shape and datatypes of the Dataset :"""

Dataset.shape

Dataset.dtypes

Dataset.info()

"""**3. Data Preparation :** <br>

Removing the Duplicates if any :
"""

Dataset.duplicated().sum()

"""Checking for the null values in each column :

"""

Dataset.isnull().sum()

"""Drop unnecessary columns :"""

Dataset.drop(['name','id','host_name','last_review'], axis=1, inplace=True)

"""Examining Changes :"""

Dataset.head(5)

Dataset.isnull().sum()

"""Replace the 'reviews per month' by zero in the case of NAN :

First of all i will explore the max and the min value of 'reviews_per_month' :
"""

Dataset['reviews_per_month'].max()

Dataset['reviews_per_month'].min()

Dataset.fillna({'reviews_per_month':0}, inplace=True)
Dataset.reviews_per_month.isnull().sum()

Dataset.isnull().sum()

Dataset.head()

Dataset['room_type'].value_counts()

Dataset.shape

Dataset.info()

"""Examine Continous Variables :"""

Dataset.describe()

"""Print all the columns names :"""

Dataset.columns

"""Regression Analysis :

1 - Getting Correlation between different variables :

Kendall :
"""

import seaborn as sns
numeric_dataset = Dataset.select_dtypes(include=[float, int])

# Calculate the correlation matrix using the Kendall method
correlation = numeric_dataset.corr(method='kendall')

# Plot the heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
Dataset.columns

"""Spearman :"""

numeric_dataset = Dataset.select_dtypes(include=[float, int])

# Calculate the correlation matrix using the Spearman method
correlation = numeric_dataset.corr(method='spearman')

# Plot the heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
Dataset.columns

"""Pearson :"""

numeric_dataset = Dataset.select_dtypes(include=[float, int])

# Calculate the correlation matrix using the Pearson method
correlation = numeric_dataset.corr(method='pearson')

# Plot the heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')

Dataset.columns

"""Data Vizualisation :"""

import seaborn as sns

Dataset['neighbourhood_group'].unique()

"""Plotting all neighbourhood Groups :"""

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=Dataset, x='neighbourhood_group', palette='plasma')
plt.title('Neighbourhood Group')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Count')
plt.show()

"""Neighbourhood :"""

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=Dataset, x='neighbourhood', palette='plasma')
plt.title('Neighbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotates the x-axis labels for better readability
plt.show()

"""Room Type"""

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=Dataset, x='room_type', palette='plasma')
plt.title('Room Types')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.show()

"""Relation between neighbourgroup and Availability of Room"""

plt.figure(figsize=(10,10))
ax = sns.boxplot(data=Dataset, x='neighbourhood_group',y='availability_365',palette='plasma')

"""Map of Neighbourhood group"""

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=Dataset, x='longitude', y='latitude', hue='neighbourhood_group')
plt.title('Scatter Plot of Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

"""Map of Neighbourhood"""

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=Dataset, x='longitude', y='latitude', hue='neighbourhood')
plt.title('Scatter Plot of Latitude and Longitude by Neighbourhood')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Neighbourhood')
plt.show()

"""Availability of Room"""

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=Dataset, x='longitude', y='latitude', hue='availability_365')
plt.title('Scatter Plot of Latitude and Longitude by Availability (365 days)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Availability (365 days)')
plt.show()

"""WordCloud :"""

from wordcloud import WordCloud

plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080
                         ).generate(" ".join(Dataset.neighbourhood))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('neighbourhood.png')
plt.show()

"""2 - Drop Columns :"""

Dataset['neighbourhood_group'].value_counts()

Dataset['neighbourhood'].value_counts()

Dataset.drop(['host_id','neighbourhood','number_of_reviews'], axis=1, inplace=True)

Dataset.head(5)

"""Saving Dataset To Power BI"""

Dataset.to_csv('Dataset_Bi_Sonder.csv', encoding='utf-8',sep=',')

"""3 - Encode the input Variables :

First Method : Function :
"""

def Encode(Dataset):
    for column in Dataset.columns[Dataset.columns.isin(['neighbourhood_group', 'room_type'])]:
        Dataset[column] = Dataset[column].factorize()[0]
    return Dataset

Dataset_Method_1 = Encode(Dataset.copy())
Dataset_Method_1.head(5)

"""Second method :"""

from sklearn.preprocessing import LabelEncoder
Dataset_cat=Dataset.drop(['price', 'minimum_nights','calculated_host_listings_count','availability_365'],axis=1)
le=LabelEncoder()
for i in Dataset_cat:
  Dataset_cat[i]=le.fit_transform(Dataset[i])
Dataset_cat.head()

Dataset_cat.head(5)

Dataset.head(5)

Dataset_num=Dataset.drop(['neighbourhood_group','room_type'],axis=1)

Dataset_Method_2=pd.concat([Dataset_num,Dataset_cat],axis=1)

Dataset_Method_2.head(10)

"""Third Method :"""

Dataset.head(5)

Price=Dataset['price']

Price.head(5)

Y=np.array(Price).reshape(-1,1)

Data=Dataset.drop(['price','neighbourhood_group','room_type'],axis=1)

from sklearn.preprocessing import LabelEncoder

X1=Dataset['neighbourhood_group']
le=LabelEncoder()
X1New=le.fit_transform(X1)
X1New=X1New.reshape(-1,1)

X2=Dataset['room_type']
le1=LabelEncoder()
X2New=le1.fit_transform(X2)
X2New=X2New.reshape(-1,1)

Dataset['neighbourhood_group']=X1New
Dataset['room_type']=X2New

"""Comparison between 3 Methods :"""

Dataset['neighbourhood_group'].value_counts()

Dataset_Method_2['neighbourhood_group'].value_counts()

Dataset_Method_1['neighbourhood_group'].value_counts()

"""**3. Modelisation :** <br>

Models :

Importing Libraries :
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

Dataset_Method_1.head(5)

"""Defining X and the Y , and  Getting test and training set :"""

X = Dataset_Method_1.drop(['price'],axis=1)
Y = Dataset_Method_1['price']

X.head()

Dataset_Method_1.head(5)

"""Linear Regression Model :"""

from sklearn.linear_model import LinearRegression
import sklearn.linear_model
from sklearn.metrics import mean_squared_error

reg=LinearRegression()
import time
debut=time.time()
reg.fit(X,Y)
Temps=time.time()-debut
print(Temps)

Y_pred_reg = reg.predict(X)

print('MSE : %.3f' % (mean_squared_error(Y, Y_pred_reg)))

print('R^2 : %.3f' % (
        r2_score(Y, Y_pred_reg),
        ))

from sklearn.metrics import explained_variance_score
from IPython.display import display

# Assuming Y and Y_pred_reg are defined
EV = explained_variance_score(Y, Y_pred_reg)
display("Explained variance: %f" % EV)


"""Decision Tree Regressor :"""

from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor(max_depth=10,random_state=2)

debut=time.time()
DTR.fit(X,Y)
Temps=time.time()-debut
print(Temps)

Y_pred_DT=DTR.predict(X)

print('MSE : %.3f' % (mean_squared_error(Y, Y_pred_DT)))

print('R^2 : %.3f' % (
        r2_score(Y, Y_pred_DT)
        ))

from sklearn.metrics import explained_variance_score
EV=explained_variance_score(Y,Y_pred_DT)
display("Explained variance : %f" %(EV))

"""Random Forest Regressor  :"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

forest = RandomForestRegressor()

debut=time.time()
forest.fit(X, Y)
Temps=time.time()-debut
print(Temps)

Y_pred_RF=forest.predict(X)

print('Mean Squared Error : %.3f' % (mean_squared_error(Y, Y_pred_RF)))

print('R^2 : %.3f' % (r2_score(Y, Y_pred_RF)))

from sklearn.metrics import explained_variance_score
EV=explained_variance_score(Y,Y_pred_RF)
display("Explained variance : %3f" %(EV))

"""KNN Regressor :"""

# Commented out IPython magic to ensure Python compatibility.
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import explained_variance_score

for K in range(20):
    K = K+1
    knn = neighbors.KNeighborsRegressor(n_neighbors = K)
    debut=time.time()
    knn.fit(X, Y)
    Temps=time.time()-debut
    print('CC = ',Temps)
    Y_pred_knn=knn.predict(X)
    MSE = mean_squared_error(Y, Y_pred_knn)
    print('MSE value for k= ' , K , 'is:', MSE)
    print('R^2 : %.3f' % (r2_score(Y, Y_pred_knn)))
    EV=explained_variance_score(Y,Y_pred_knn)
    display("Explained variance : %f" %(EV))

knnFinal = neighbors.KNeighborsRegressor(n_neighbors = 1)
debut=time.time()
knnFinal.fit(X, Y)
Temps=time.time()-debut
print('CC = ',Temps)

Y_pred_knnFinal=knnFinal.predict(X)

MSE = mean_squared_error(Y, Y_pred_knn)
print('MSE value for k= 1'   'is:', MSE)
print('R^2 : %.3f' % (r2_score(Y, Y_pred_knnFinal)))
EV=explained_variance_score(Y,Y_pred_knnFinal)
display("Explained variance : %f" %(EV))

"""XGBOOST :"""

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import xgboost

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=30)

debut=time.time()
xgb.fit(X,Y)
Temps=time.time()-debut
print(Temps)

Y_pred_xgb = xgb.predict(X)

print('Mean squared Error : %.3f' % (mean_squared_error(Y, Y_pred_xgb)))

print('R^2 : %.3f' % (r2_score(Y, Y_pred_xgb)))

from sklearn.metrics import explained_variance_score
EV=explained_variance_score(Y,Y_pred_xgb)
display("Explained variance : %f" %(EV))

"""**5. Evaluation:** <br>

As mentioned earlier, for the evaluation of the price prediction model, we used MSE, R2, explained variance, and computational complexity as evaluation metrics. The table illustrates the evaluation results of five algorithms used (Linear Regression, Decision Tree, Random Forest, KNN Regressor, and XGBoost). According to this table, it is clear that KNN Regressor performs the best in terms of explained variance, R2 statistic, and computational complexity with k=1, as it is generally well-suited for numerical data.

In terms of comparison between Random Forest, Decision Tree, and Linear Regression, we can clearly see that Random Forest performs the best in terms of explained variance and R2 statistic, as it is a good classifier in the bagging category.

In terms of computational complexity, we observe that KNN has the best performance with the shortest execution time. Decision Tree, Random Forest, and XGBoost have longer execution times. XGBoost has the longest execution time, taking around 32 seconds to process and analyze the data. This computational complexity can be attributed to the use of multiple layers, and the learning of the data is executed incrementally.

Our choice is selected based on KNN since it provides the best results in terms of computational complexity, explained variance, MSE, and R2.

**6. Deployment:** <br>

Choosen model :
"""

import pickle

pickle.dump(knnFinal,open('knnSonder.pkl','wb'))