
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder
import missingno as msno
import numpy as np

#Loading dataset and checking missing values
iris=pd.read_csv('Iris.csv')    
msno.bar(iris,figsize=(8,6),color='skyblue')


X=iris.iloc[:,0:4].values
y=iris.iloc[:,4].values

le = LabelEncoder()
y = le.fit_transform(y)

#Model Select

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)
accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)


cm = confusion_matrix(y_test, Y_prediction)
accuracy = accuracy_score(y_test,Y_prediction)
precision =precision_score(y_test, Y_prediction,average='weighted')
recall =  recall_score(y_test, Y_prediction,average='weighted')
f1 = f1_score(y_test,Y_prediction,average='weighted')
print('Confusion matrix for Random Forest\n',cm)
print('accuracy_random_Forest : %.3f' %accuracy)
print('precision_random_Forest : %.3f' %precision)
print('recall_random_Forest : %.3f' %recall)
print('f1-score_random_Forest : %.3f' %f1)

#logistics regression start
logreg = LogisticRegression(solver= 'lbfgs',max_iter=400)
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
accuracy_lr=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)


cm = confusion_matrix(y_test, Y_pred,)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='weighted')
recall =  recall_score(y_test, Y_pred,average='weighted')
f1 = f1_score(y_test,Y_pred,average='weighted')
print('Confusion matrix for Logistic Regression\n',cm)
print('accuracy_Logistic Regression : %.3f' %accuracy)
print('precision_Logistic Regression : %.3f' %precision)
print('recall_Logistic Regression: %.3f' %recall)
print('f1-score_Logistic Regression : %.3f' %f1)


#LinearSVC
linear_svc = LinearSVC(max_iter=4000,dual=False)
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
accuracy_svc=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='weighted')
recall =  recall_score(y_test, Y_pred,average='weighted')
f1 = f1_score(y_test,Y_pred,average='weighted')
print('Confusion matrix for SVC\n',cm)
print('accuracy_SVC: %.3f' %accuracy)
print('precision_SVC: %.3f' %precision)
print('recall_SVC: %.3f' %recall)
print('f1-score_SVC : %.3f' %f1)

#DecisionTreeClassifier
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, y_train)  
Y_pred = decision_tree.predict(X_test) 
accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='weighted')
recall =  recall_score(y_test, Y_pred,average='weighted')
f1 = f1_score(y_test,Y_pred,average='weighted')
print('Confusion matrix for DecisionTree\n',cm)
print('accuracy_DecisionTree: %.3f' %accuracy)
print('precision_DecisionTree: %.3f' %precision)
print('recall_DecisionTree: %.3f' %recall)
print('f1-score_DecisionTree : %.3f' %f1)


#GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test) 
accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='weighted')
recall =  recall_score(y_test, Y_pred,average='weighted')
f1 = f1_score(y_test,Y_pred,average='weighted')
print('Confusion matrix for Naive Bayes\n',cm)
print('accuracy_Naive Bayes: %.3f' %accuracy)
print('precision_Naive Bayes: %.3f' %precision)
print('recall_Naive Bayes: %.3f' %recall)
print('f1-score_Naive Bayes : %.3f' %f1)

#KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
accuracy_knn = round(accuracy_score(y_test, Y_pred) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test, Y_pred)
precision_knn = precision_score(y_test, Y_pred, average='weighted')
recall_knn = recall_score(y_test, Y_pred, average='weighted')
f1_knn = f1_score(y_test, Y_pred, average='weighted')
print('Confusion matrix for KNN\n', cm)
print('accuracy_KNN: %.3f' % accuracy)
print('precision_KNN: %.3f' % precision_knn)
print('recall_KNN: %.3f' % recall_knn)
print('f1-score_KNN : %.3f' % f1_knn)



models = [
    ("KNN", KNeighborsClassifier(n_neighbors=3)),
    ("Logistic Regression", LogisticRegression(solver='lbfgs', max_iter=400)),
    ("Random Forest", RandomForestClassifier(n_estimators=100)),
    ("Naive Bayes", GaussianNB()),
    ("Support Vector Machine", LinearSVC(max_iter=4000, dual=False)),
    ("Decision Tree", DecisionTreeClassifier())
]

# Model training and evaluation
results = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred, average='weighted') * 100, 2)
    recall = round(recall_score(y_test, y_pred, average='weighted') * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
    results.append((name, accuracy, precision, recall, f1))

results_df = pd.DataFrame(results, columns=["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)"])
results_df = results_df.sort_values(by='Accuracy (%)', ascending=False)

# Plotting results
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Model', y="Accuracy (%)", data=results_df)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.show()

# Displaying pair plot
sns.pairplot(iris.drop(columns=['Id']), hue="Species")
plt.show()



