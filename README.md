# Iris_Classification_Comparison

Iris Classifier Comparison: Code to compare classification models on Iris dataset, assessing accuracy, precision, recall, and F1-score. Includes visualizations.

# Results

- Confusion matrix for Random Forest  
   [[16  0  0]  
    [ 0 17  1]  
    [ 0  0 11]]  
   accuracy Random Forest: 97.78%  
   precision Random Forest: 97.96%  
   recall Random Forest: 97.78%  
   f1 score Random Forest: 97.79%

- Confusion matrix for Logistic Regression  
   [[16  0  0]  
    [ 0 17  1]  
    [ 0  0 11]]  
   accuracy Logistic Regression: 97.78%  
   precision Logistic Regression: 97.96%  
   recall Logistic Regression: 97.78%  
   f1 score Logistic Regression: 97.79%

- Confusion matrix for Support Vector Machine  
   [[16  0  0]  
    [ 0 15  3]  
    [ 0  0 11]]  
   accuracy Support Vector Machine: 93.33%  
   precision Support Vector Machine: 94.76%  
   recall Support Vector Machine: 93.33%  
   f1 score Support Vector Machine: 93.43%

- Confusion matrix for DecisionTree  
   [[16  0  0]  
    [ 0 17  1]  
    [ 0  0 11]]  
   accuracy Decision Tree: 97.78%  
   precision Decision Tree: 97.96%  
   recall Decision Tree: 97.78%  
   f1 score Decision Tree: 97.79%

- Confusion matrix for Naive Bayes  
   [[16  0  0]  
    [ 0 18  0]  
    [ 0  0 11]]  
   accuracy Naive Bayes: 100.00%  
   precision Naive Bayes: 100.00%  
   recall Naive Bayes: 100.00%  
   f1 score Naive Bayes: 100.00%

- Confusion matrix for KNN  
   [[16  0  0]  
    [ 0 17  1]  
    [ 0  0 11]]  
   accuracy KNN: 97.78%  
   precision KNN: 97.96%  
   recall KNN: 97.78%  
   f1 score KNN: 97.79%

# Conclusion

Based on the results from our classification models:

- Random Forest, Logistic Regression, Decision Tree, and KNN all got a high score of 97.8% for accuracy. These models demonstrated consistent performance across precision, recall, and F1-score metrics, indicating robustness in classifying the Iris species accurately.

- Naive Bayes did even better, getting a perfect score of 100% accuracy, indicating flawless classification performance. This model displayed perfect precision, recall, and F1-score, suggesting it accurately classified all instances of Iris species in the test dataset.

- Support Vector Machine (SVC) achieved a slightly lower accuracy of 93.3% compared to the other models. While it still provided reliable classification results, it showed a marginally lower precision, recall, and F1-score compared to the other models

Overall, the models trained on the Iris dataset performed exceptionally well, with Naive Bayes demonstrating the highest accuracy and the other models closely following suit. The choice of the best model may depend on various factors such as computational efficiency, interpretability, and the specific requirements of the application.
