# import packages
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# extract data
canada_data = pd.read_csv("C:\\To Read\\Data_sets\\canada_per_capita_income.csv")
canada_data

# graph
plt.scatter(canada_data['year'], canada_data['per capita income (US$)'], marker = '+', color = 'red')

# training input and output
input_canada_data = canada_data[['year']]
output_canada_data = canada_data[['per capita income (US$)']]
input_canada_data_train, input_canada_data_test, output_canada_data_train, output_canada_data_test = train_test_split(input_canada_data, output_canada_data, test_size=0.2)

# model
model = LinearRegression()
model.fit(input_canada_data_train, output_canada_data_train)
prediction = model.predict(input_canada_data_test)
answer = model.predict([[2020]])
print("Canada's per capita in 2020 was",int(answer))

# accuracy
print("The accuracy of my model is ",int(model.score(input_canada_data_train, output_canada_data_train)*100),'%')