# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MOHANAPRABHA S
RegisterNumber:  212224040197
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict(pd.DataFrame([[5,6]], columns=x.columns))
```

## Output:

<img width="464" height="347" alt="image" src="https://github.com/user-attachments/assets/2e1cf1d6-6be7-4326-8c70-e2dd87bd2a93" />


<img width="489" height="278" alt="image" src="https://github.com/user-attachments/assets/82e4b2b2-a7b9-46ca-982c-480fcbe18f98" />


<img width="305" height="258" alt="image" src="https://github.com/user-attachments/assets/ea1953c7-0531-49a6-b31a-b6a4dca167e7" />


<img width="552" height="374" alt="image" src="https://github.com/user-attachments/assets/f2e66f39-628a-45f6-8f57-c88bdb666681" />


<img width="559" height="158" alt="image" src="https://github.com/user-attachments/assets/7326ba14-c73f-4dbe-a8f9-67a461b24377" />


<img width="678" height="135" alt="image" src="https://github.com/user-attachments/assets/07ecfb45-7520-4b82-b278-1de3bae699a0" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
