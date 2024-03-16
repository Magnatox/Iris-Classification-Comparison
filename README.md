# Iris_Classification_Comparison

Iris Classifier Comparison: Code to compare classification models on Iris dataset, assessing accuracy, precision, recall, and F1-score. Includes visualizations.

# Results

- Confusion matrix for Random Forest  
    [[16  0  0]  
    [ 0 17  1]  
    [ 0  0 11]]  
    accuracy_random_Forest : 0.978  
    precision_random_Forest : 0.980  
    recall_random_Forest : 0.978  
    f1-score_random_Forest : 0.978  

- Confusion matrix for Logistic Regression  
      [[16  0  0]  
      [ 0 17  1]  
      [ 0  0 11]]  
      accuracy_Logistic Regression : 0.978  
       precision_Logistic Regression : 0.980  
       recall_Logistic Regression: 0.978  
       f1-score_Logistic Regression : 0.978

- Confusion matrix for SVC  
      [[16  0  0]  
      [ 0 15  3]  
      [ 0  0 11]]  
      accuracy_SVC: 0.933  
      precision_SVC: 0.948  
      recall_SVC: 0.933  
      f1-score_SVC : 0.934  

- Confusion matrix for DecisionTree  
      [[16  0  0]  
      [ 0 17  1]  
      [ 0  0 11]]  
      accuracy_DecisionTree: 0.978  
      precision_DecisionTree: 0.980  
      recall_DecisionTree: 0.978  
      f1-score_DecisionTree : 0.978  

- Confusion matrix for Naive Bayes  
      [[16  0  0]  
      [ 0 18  0]  
      [ 0  0 11]]  
      accuracy_Naive Bayes: 1.000  
      precision_Naive Bayes: 1.000  
      recall_Naive Bayes: 1.000  
      f1-score_Naive Bayes : 1.000  

- Confusion matrix for KNN  
      [[16  0  0]  
      [ 0 17  1]  
      [ 0  0 11]]  
      accuracy_KNN: 0.978  
      precision_KNN: 0.980  
      recall_KNN: 0.978  
      f1-score_KNN : 0.978  

# Conclusion

Based on the results from our classification models:

- Random Forest, Logistic Regression, Decision Tree, and KNN did a really good job. They all got a high score of 97.8% for accuracy, which means they were mostly right when predicting the type of Iris flower.

- Naive Bayes did even better, getting a perfect score of 100% accuracy. It means it didn't make any mistakes in predicting the Iris flower types. It was precise, remembered all the actual flowers, and found a good balance between precision and recall.

- Support Vector Machine (SVC) did a bit worse than the others with an accuracy of 93.3%. It means it made a few more mistakes compared to the other models. It wasn't as precise, didn't remember all the actual flowers.

Overall, all the models did a good job at predicting the type of Iris flower, but Naive Bayes performed the best. Which model to use might depend on things like how fast it needs to work or how easy it is to understand.
