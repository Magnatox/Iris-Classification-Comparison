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

- Random Forest, Logistic Regression, Decision Tree, and KNN all got a high score of 97.8% for accuracy. These models demonstrated consistent performance across precision, recall, and F1-score metrics, indicating robustness in classifying the Iris species accurately.

- Naive Bayes did even better, getting a perfect score of 100% accuracy, indicating flawless classification performance. This model displayed perfect precision, recall, and F1-score, suggesting it accurately classified all instances of Iris species in the test dataset.

- Support Vector Machine (SVC) achieved a slightly lower accuracy of 93.3% compared to the other models. While it still provided reliable classification results, it showed a marginally lower precision, recall, and F1-score compared to the other models

Overall, the models trained on the Iris dataset performed exceptionally well, with Naive Bayes demonstrating the highest accuracy and the other models closely following suit. The choice of the best model may depend on various factors such as computational efficiency, interpretability, and the specific requirements of the application.
