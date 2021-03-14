import pandas as pd
import numpy as np

#loading the dataset
df = pd.read_csv("D:/BLR10AM/Assi/15.KNN/Datasets_KNN/glass.csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["userid is unique customer number ,irrelevant",
                "Gender is the range of characteristics pertaining to, and differentiating between, femininity and masculinity,important",
                "age of person , impotant",
                "salary of person,very important",
                "did he purchased the SUV or not"]

d_types =["nominal","binary","ratio","ratio","binary"]

data_details =pd.DataFrame({"column name":df.columns,
                            "data types ":d_types,
                            "description":description})

            #3.	Data Pre-processing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of df 
df.info()
df.describe()          


#data types        
df.dtypes


#checking for na value
df.isna().sum()
df.isnull().sum()

#checking unique value for each columns
df.nunique()

#variance of df
df.var()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": df.columns,
      "mean": df.mean(),
      "median":df.median(),
      "mode":df.mode(),
      "standard deviation": df.std(),
      "variance":df.var(),
      "skewness":df.skew(),
      "kurtosis":df.kurt()}

EDA

# covariance for data set 
covariance = df.cov()
covariance


####### graphical repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(df.iloc[:, :])


#boxplot for every columns
df.columns
df.nunique()

boxplot = df.boxplot(column=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type'])   #no outlier

#creatind dataframe with only with (discrete,continuous ,output)
X = df.iloc[:,0:9] # Predictors 
Y   = df.iloc[:,[9]] # Target 

# Normalization functio,
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


# Normalized data frame (considering the numerical part of data)
X = norm_func(X.iloc[:,:])


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=77)

#converting ouput into  series
y_test = Y_test["Type"]

y_train = Y_train["Type"]

"""
5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform KNN, and use cross validation techniques to get N-neighbors
5.3	Train and Test the data and perform cross validation techniques, compare accuracies, precision and recall and explain about them.
5.4	Briefly explain the model output in the documentation. """


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score 
print(accuracy_score(Y_test, pred))
pd.crosstab(y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,17,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,17,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,17,2),[i[1] for i in acc],"bo-")
