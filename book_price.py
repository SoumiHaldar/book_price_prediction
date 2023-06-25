import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',force=True)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df_train=pd.read_excel('A:\\ML\\BookPrice_prediction\\Data\\Participants_Data\\Data_Train.xlsx')
df_test=pd.read_excel(r'A:\ML\BookPrice_prediction\Data\Participants_Data\Data_Test.xlsx')
print('train data:','\n',df_train.head())
print('test data:','\n',df_test.head())

print(df_train.info())
print(df_test.info())

# Modifying the Ratings and the Reviews features and changing its datatypes to float.
df_train['Ratings']=df_train['Ratings'].str[0].astype('int64')
df_train['Reviews']=df_train['Reviews'].str[0:3].astype('float64')

df_test['Ratings']=df_test['Ratings'].str[0].astype('int64')
df_test['Reviews']=df_test['Reviews'].str[0:3].astype('float64')

# Publication year and age of editions

df_train['Edition year']=df_train['Edition'].str.split('–').str[-1]
df_train['Edition type']=df_train['Edition'].str.split('–').str[0].str.replace(',','')

df_test['Edition year']=df_test['Edition'].str.split('–').str[-1]
df_test['Edition type']=df_test['Edition'].str.split('–').str[0].str.replace(',','')

# We will now analyze Edition year (if it contains any alphabets)
x=df_train[df_train['Edition year'].apply(lambda x: str(x).isalpha())]
print(x)
x1=df_test[df_test['Edition year'].apply(lambda x: str(x).isalpha())]
print(x1)

# variation with Edition type with price
plt.figure(figsize=(20,8))
sns.barplot(x=df_train['Edition type'],y=df_train['Price'])
plt.xticks(rotation=90)
plt.plot()
plt.show()

# check outliers in price column
# df_train['Price'].plot(kind='box',layout=(7,2),figsize=(10,8))
# Book category
df_train['BookCategory'].value_counts()
plt.figure(figsize=(10,8))
y=df_train['Price']
x=df_train['BookCategory']
plt.xticks(rotation=90)
plt.bar(x,y,color='Maroon',width=0.4)

# Edition year is of categorical structure
df_train['Edition year']=df_train['Edition year'].str[-4:]
df_train['Edition year']=df_train['Edition year'].map(df_train['Edition year'].value_counts())

df_test['Edition year']=df_test['Edition year'].str[-4:]
df_test['Edition year']=df_test['Edition year'].map(df_train['Edition year'].value_counts())

#  Both Genre and bookcategory columns have categorized structure
df_train['Genre']=df_train['Genre'].map(df_train['Genre'].value_counts())
df_train['BookCategory']=df_train['BookCategory'].map(df_train['BookCategory'].value_counts())
df_train['Edition type']=df_train['Edition type'].map(df_train['Edition type'].value_counts())

df_test['Genre']=df_test['Genre'].map(df_test['Genre'].value_counts())
df_test['BookCategory']=df_test['BookCategory'].map(df_test['BookCategory'].value_counts())
df_test['Edition type']=df_test['Edition type'].map(df_test['Edition type'].value_counts())

# Drop the columns Title,Author,Edition
train_data=df_train.drop(['Title','Author','Edition','Synopsis'],axis=1)

# Create train and test data
x=train_data.drop(['Price'],axis=1)
y=train_data['Price']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

# Linear regression
from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(xtrain,ytrain)
prediction1=model1.predict(xtest)
print('Mean Squared Error in LR: ', mean_squared_error(ytest, prediction1))

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
model2=DecisionTreeRegressor()
model2.fit(xtrain,ytrain)
prediction2=model2.predict(xtest)
print('Mean Squared Error in Decision Tree: ', mean_squared_error(ytest, prediction2))

# Random Forest
from sklearn.ensemble import RandomForestRegressor
model3=RandomForestRegressor()
model2.fit(xtrain,ytrain)
prediction2=model2.predict(xtest)
print('Mean Squared Error in Decision Tree: ', mean_squared_error(ytest, prediction2))
