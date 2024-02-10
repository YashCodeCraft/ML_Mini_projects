# import packages
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings as w
w.filterwarnings('ignore')

# extract data
score_data = pd.read_excel("C:\\To Read\\Data_sets\\Scorecard.xlsx")
score_data
plt.scatter(score_data['Runs'],score_data['Wickets'])

# Setting output using Kmeans
km = KMeans(n_clusters=3)
predicted_op = km.fit_predict(score_data[['Runs','Wickets']])
predicted_op
print(km.inertia_)
sse=[]
krange = range(1,6)
for i in krange:
    km1=KMeans(n_clusters=i)
    km1.fit(score_data[['Runs','Wickets']])
    sse.append(km1.inertia_)
sse

sns.lineplot(sse) # linear line in 3rd data point(elbow line)
score_data['clusters'] = predicted_op
score_data

# Creating a new column
status = ['Batsmen', 'Bowler', 'Allrounder']
status
score_data['style'] = score_data['clusters'].apply(lambda x : status[x])
score_data

# plotting a graph
score_data0 = score_data[score_data['clusters']==0]
score_data1 = score_data[score_data['clusters']==1]
score_data2 = score_data[score_data['clusters']==2]
plt.scatter(score_data0['Runs'],score_data0['Wickets'],color='red',marker='+')
plt.scatter(score_data1['Runs'],score_data1['Wickets'],color='blue',marker='+')
plt.scatter(score_data2['Runs'],score_data2['Wickets'],color='green',marker='+')

# use KNN algorithm to predict
input_ = score_data.drop(columns=['Players','Wickets','style'])
output_= score_data['style']

# trainings and predicting the output
input_train, input_test, output_train, output_test=train_test_split(input_,output_,test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=1)
knn

knn.fit(input_train, output_train)
NoOfPlayers = int(input("Enter the number of players: "))
print()
for i in range(NoOfPlayers):
    Runs = int(input("No of runs scored: "))
    wickets = int(input("No of wickets have taken: "))
    predict = knn.predict([[Runs,wickets]])
    print('The player must be the',predict)
    print()
print("The accuracy is",int(knn.score(input_test,output_test))*100)
sns.scatterplot(score_data[['Runs',"Wickets"]])