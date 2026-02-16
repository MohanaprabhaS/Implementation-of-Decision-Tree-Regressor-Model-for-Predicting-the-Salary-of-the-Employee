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

<img width="416" height="354" alt="image" src="https://github.com/user-attachments/assets/dded282f-284d-4150-be42-69edf3a7033d" />


<img width="412" height="278" alt="image" src="https://github.com/user-attachments/assets/d42f413f-835a-4492-93b5-f8f5dc48d5e3" />


<img width="300" height="272" alt="image" src="https://github.com/user-attachments/assets/8f2edd3e-67fd-4802-90cb-f83bdd6c1127" />


<img width="607" height="381" alt="image" src="https://github.com/user-attachments/assets/9fdf7165-5f00-451f-8c07-6eb1c4ea5c25" />


<img width="839" height="229" alt="image" src="https://github.com/user-attachments/assets/fb9e4620-9e01-4601-9dfc-b980e2eda9d0" />


<img width="557" height="167" alt="image" src="https://github.com/user-attachments/assets/84517c4d-d540-4adb-a245-6402aa7ba55f" />


<img width="624" height="130" alt="image" src="https://github.com/user-attachments/assets/44ea4676-47fd-42ca-b515-1cfddc8eebf3" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
