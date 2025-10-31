#importing the libraries needed
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("loan_approval.csv")

#defining comparatively advanced rules using scores
def advanced(row):
    score = 0
    #rules for credit score, the better it is, the more increase in overall score
    if row['credit_score']>=750:
        score += 3
    elif row['credit_score']>=650:
        score += 2
    elif row['credit_score']>=550:
        score += 1

    #rules for income
    if row['income']>=80000:
        score += 3
    elif row['income']>=50000:
        score += 2
    elif row['income']>=30000:
        score += 1

    #employment duration rules
    if row['years_employed']>=10:
        score+=2
    elif row['years_employed']>=5:
        score+=1

    #ratio of loan to income
    if row['loan_amount']<0.3*row['income']:
        score+=3
    elif row['loan_amount'] <0.5*row['income']:
        score+=2
    else:
        score+=0

    #if 'points' are available, custom rule is applied
    if 'points' in row and row['points'] >= 50:
        score += 1

    #making the decision
    return True if score >= 8 else False

#applying the rules
df['predicted'] = df.apply(advanced, axis=1)

#evaluating
correct = df['loan_approved']
prediction = df['predicted']

print('advanced rule based AI performance')
print('accuracy:', accuracy_score(correct, prediction))
print('confusion matrix:\n', confusion_matrix(correct, prediction))
print('classification report:\n', classification_report(correct, prediction))
