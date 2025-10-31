#importing pandas and sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#accessing the dataset csv file 
df = pd.read_csv("loan_approval.csv")

#defining the rule based function
def basic(row):
    if (row['credit_score']> 650 and
        row['income']>40000 and
        row['years_employed']>2 and
        row['loan_amount']<(0.5*row['income'])):
        return True
    else:
        return False

#applying the above function to a new df where the AI will store its predictions
df['predicted'] = df.apply(basic, axis=1)

# evaluating stuff
correct = df['loan_approved']
prediction = df['predicted']
print('performance of basic rule based AI')
print('accuracy:', accuracy_score(correct, prediction))
print("confusion matrix:\n", confusion_matrix(correct, prediction))
print('classification report:\n', classification_report(correct, prediction))

