# Module 12 Report Template

## Overview of the Analysis

The purpose of the analysis is to predict whether an individual loan is high-risk or healthy. This prediction is based on the following predictors:

    loan_size
    interest_rate
    borrower_income
    debt_to_income - calculated by dividing total_debt by borrower income
    num_of_accounts 
    derogatory_marks - due to late payments, bankruptcy, charge-offs and other negative marks on your credit report
    total_debt - total debt excluding the loan applied for

Data from previous loanees is split in test and training data with the above predictors as input features and output labels being to predict whether the loan is healthy (0) or high-risk (1).

As this is a two output clustering problem a simple logistic regression algorithm is fitted to the training data (58152 loans) and then the test data (19384 loans) is used to check the predictions made by the model.

An accuracy value is then calculated i.e the percentage of data points of all data points that are correctly assigned to their true labels 

Additionally a confusion matrix is caluclated. This assigns the predictions from the test data to positive-positive, negative-negative, positive-negative and negative-positive results.

The classification report at the end of the regression analysis shows the precision, recall, f1-score and support percentages for healthy and high-risk predictions and their averages.

The initial training and testing data has an imbalance - there are 75036 healthy loans and 2500 high-risk loans in the data. This will lead to the regression algorithm training to be skewed towards the healthy loans. The healthy loans predictions will be more accurate than the high-risk loans. This is indeed the case. Therefore the second part of the analysis tries to rectify the situation by over training on the high-risk data.

The training data is re-sampled using the imbalanced-learn library. This copies the the high-risk data until there is an equal number of high-risk and healthy loans training daata. In this case it gives 56271 instances of each in the training data.

The logistic regression algorithm is again trained but this time with the re-sampled training data. The new regression model is then tested using the original test data. It showed a marginal improvement in in the accurate predictions of the high-risk loans whilst not affecting the predictions of the healthy loans at all. 

Please note that the data was not stratified before training both models. If stratified the results improved for both models.



## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:

  The balanced accuracy score for model 1 was 99.3%
    
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  
                           precision    recall  f1-score   support

                healthy       1.00      0.99      1.00     18765
              high-risk       0.84      0.94      0.89       619

               accuracy                           0.99     19384
              macro avg       0.92      0.97      0.94     19384
           weighted avg       0.99      0.99      0.99     19384  

       * Precision is the ratio of correctly predicted positive observations to the total predicted positives i.e. precison = TP / (TP + FP). For the healthy class, the precision is 1.00, meaning that 100% of all instances predicted as healthy were actually healthy. For the high-risk class, the precision is 0.84, indicating that 84% of instances predicted as high-risk were actually high-risk.

        * Recall is the ratio of correctly predicted positive observations to the all observations in the actual class i.e. recall = TP / (TP + FN). For the healthy class, the recall is 0.99, meaning that 99% of healthy instances were correctly identified. For the high-risk class, the recall is 0.94, indicating that 94% of actual high-risk instances were correctly identified.

        * Support shows that there are 18,765 instances of healthy and 619 instances of high-risk in the test data.

        * Accuracy is the ratio of correctly predicted instances to the total instances in the dataset. In this case, the overall accuracy of the model is 0.99, which means that 99% of instances were classified correctly.

        * Macro average calculates the average for the healthy and high-risk instances for each of precision, recall, and F1-score, the macro average for precision, recall, and F1-score between 92 and 97%.

        * Weighted average take the macro average and weights by the support. In this case, the weighted average precision, recall, and F1-score are all 99%.

    * confusion matrix results:

        array([[18658,   107],
               [   37,   582]]
    
        TN = 18658, FP = 107, FN = 37, TP = 582
           
        The confusion matrix shows that 18658 healthy loans (TN because 0 is healthy) and 582 high-risk loans (TP as a high risk loan is 1) were correctly identified  from the test data. However, 37 high-risk loans were identified as healthy (FN) and 107 healthy loans were identified as high-risk (FP) incorrectly.           


* Machine Learning Model 2:

    The balanced accuracy score for model 1 was 99.4%

  * Description of Model 2 Accuracy, Precision, and Recall scores.

                        precision    recall  f1-score   support

             healthy       1.00      0.99      1.00     18765
           high-risk       0.84      0.99      0.91       619

            accuracy                           0.99     19384
           macro avg       0.92      0.99      0.95     19384
        weighted avg       0.99      0.99      0.99     19384
        
        * Once the data had been re-sampled to equalise the number of healthy and high-risk loans in the training data sets  the predictions changed slightly. The precision values remained the same but the recall values for predictions for high-risk   loans went from 94% to 99%. This means that only 84% of the high-risk loans for identified as high-risk, out of the total sample size which   is much larger both the healthy and high-risk loans were identified with an accuracy of 99%.

    * confusion matrix results:

        array([[18646,   119],
               [    4,   615]]
           
        TN = 18646, FP = 119, FN = 4, TP = 615
  
        The confusion matrix shows that 18646 healthy loans (TN) and 615 high-risk loans (FP) were correctly identified from the test data. However, 119 healthy loans were incorrectly identified as high-risk (FP) and 4 high-risk loans were incorrectly identified as healthy (FN).           


## Summary

The average loan size was $9805. We will do the following calculations using this as the loan size.

In the first model the number of loans wrongly assumed as high-risk and therefore turned down (FP) was 107. This cost the bank interest on $1,049,135 on those loans. The high-risk loans that were wrongly assumed to be healthy (FN) were 37. This total loan amount was $362,785. So the total loss to the bank was the interest on the wrongly turned down loans (assume 6%) and the capital on the wrongly given high-risk loans. This works out to be a total cost to the bank of (0.06 x 1,049,135) + 362,785 = $425,733.

In the second model the number of loans wrongly assumed as high risk and therefore turned down (FP) was 119. This cost the bank   interest on $1,116,795 on those loans. The high-risk loans that were wrongly assumed to be healthy (FN) were 4. This total loan amount   was  $39,220. So the total loss to the bank was the interest on the wrongly turned down loans (assume 6%) and the capital on the   wrongly given high-risk loans. This works out to be a total cost to the bank of (0.06 x $1,116,795) + $39,220 = $106,228.

The oversampled method provides the better results. It is more important to the banks profitablity to reduce the number of  high-risk loans which are incorrectly identified as healthy rather than turning down good applicants incorrectly. So oversampling to identify high-risk loans at the expense of healthy loans works because it reduces the number of high-risk loans incorrectly identified.. 
