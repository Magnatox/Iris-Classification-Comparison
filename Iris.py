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

# Loading dataset and checking missing values
iris = pd.read_csv('Iris.csv')   
iris = iris.drop(columns=['Id']) 
msno.bar(iris, figsize=(8, 6), color='skyblue')
plt.title("Missing Values Check")

# Pairplot
plot = sns.PairGrid(iris)
plot.map_diag(plt.hist)
plot.map_upper(plt.scatter)
plot.map_lower(sns.kdeplot)


# Splitting features and target variable
X = iris.drop(columns=['Species']).values
y = iris['Species'].values

# Encoding target variable
le = LabelEncoder()
y = le.fit_transform(y)


# Splitting data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# Model initialization
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
    cm = confusion_matrix(y_test, y_pred)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred, average='weighted') * 100, 2)
    recall = round(recall_score(y_test, y_pred, average='weighted') * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
    results.append((name, accuracy, precision, recall, f1))
    print('Confusion matrix for {}\n'.format(name), cm)
    print('accuracy {}: %.2f%%'.format(name) % accuracy)
    print('precision {}: %.2f%%'.format(name) % precision)
    print('recall {}: %.2f%%'.format(name) % recall)
    print('f1 score {}: %.2f%%'.format(name) % f1)

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
sns.pairplot(iris, hue="Species")
plt.show()



