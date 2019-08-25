from template_classy import PsplinesClf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
#
# loading sci-kit learn internal data
#
X,y = load_breast_cancer(return_X_y=True)
x = X[:,1]
x = x.reshape(-1, 1)
model = PsplinesClf(ndx=40, smoothing = 1000) #1,10,100,1000,10000
model.fit(x,y)

#
# logistic regression for comparison
#
clf = OneVsRestClassifier(LogisticRegression(solver = 'lbfgs' ))

# Fit it to the training data
clf.fit(x, y)
print(clf.score(x,y))
print(model.score(x,y))

logit_roc_auc = roc_auc_score(y, model.predict(x))
fpr, tpr, thresholds = roc_curve(y, model.predict_proba(x))
logit_roc_auc1 = roc_auc_score(y, clf.predict(x))
fpr1, tpr1, thresholds1 = roc_curve(y, clf.predict_proba(x)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Smooth Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr1, tpr1, label='Logistic Regression (area = %0.2f)' % logit_roc_auc1)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate2')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#
# plot of smooth curve
#
x = np.linspace(np.min(x),np.max(x),600)
x = x.reshape(-1, 1)
pred = model.predict_proba(x)
plt.plot(x,pred,label="Smooth logistic regression")
lr_pred = clf.predict_proba(x)
plt.plot(x,lr_pred[:,1],label="Logistic regression")
plt.ylim(0, 1)
plt.ylabel('Probability')
plt.xlabel('Mean texture')
plt.legend(loc="best")
plt.show()

