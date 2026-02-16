# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
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

<img width="464" height="347" alt="image" src="https://github.com/user-attachments/assets/c2acfeb3-d632-43a9-9466-f0e704605281" />


<img width="489" height="278" alt="image" src="https://github.com/user-attachments/assets/b78bf784-f8c0-4426-8e16-b5beafe74279" />


<img width="305" height="258" alt="image" src="https://github.com/user-attachments/assets/9fed601e-7295-4c67-8ce7-f2b9bb432b07" />


<img width="552" height="374" alt="image" src="https://github.com/user-attachments/assets/5d81caa9-2d95-4334-9c9e-3768002d7b0a" />


<img width="559" height="158" alt="image" src="https://github.com/user-attachments/assets/697106c8-6b1b-4712-888f-3c52e42ce3a9" />


<img width="678" height="135" alt="image" src="https://github.com/user-attachments/assets/2fbfe3c6-9a47-4e0b-be5b-2894db95a47c" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
