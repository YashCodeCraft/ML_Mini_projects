# import packages
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# extracting data
salary_data = pd.read_csv("C:\To Read\Data_sets\salaries.csv")
salary_data1 = pd.read_csv('C:\To Read\Data_sets\salaries.csv')
salary_data

# graph
plt.scatter(salary_data['job'], salary_data1['salary_more_then_100k'], marker = '+', color = 'red')
plt.scatter(salary_data['company'], salary_data1['salary_more_then_100k'], marker = '+', color = 'red')
plt.scatter(salary_data['degree'], salary_data1['salary_more_then_100k'], marker = '+', color = 'red')

# input and output
input_salary = salary_data.drop(columns=['salary_more_then_100k'])
input_salary = pd.get_dummies(input_salary, dtype = int)
output_salary = salary_data['salary_more_then_100k']
input_salary

# model and training
input_salary_train, input_salary_test, output_salary_train, output_salary_test = train_test_split(input_salary, output_salary, test_size=0.2)
model = LogisticRegression()
model.fit(input_salary_train, output_salary_train)
prediction = model.predict(input_salary_test)

# accuracy
print("The accuracy of my model is",int(model.score(input_salary_train, output_salary_train)*100),'%')