from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,f1_score
def evaluation(y_pred,y_test,plot=True):
  from sklearn.metrics import roc_auc_score
  from sklearn.metrics import roc_curve
  
  print(classification_report(y_test, y_pred))
  if plot:
    logit_roc_auc = roc_auc_score(y_test,y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
  if (y_test.to_numpy()[0] == 't') | (y_test.to_numpy()[0] == 'f'):
    yt = y_test == 't'
    yp = y_pred == 't' 
    return f1_score(yt,yp) 
  return f1_score(y_test,y_pred)

def cal_f1_score(y_test,y_pred):
  if (y_test.to_numpy()[0] == 't') | (y_test.to_numpy()[0] == 'f'):
    yt = y_test == 't'
    yp = y_pred == 't' 
    return f1_score(yt,yp) 
  return f1_score(y_test,y_pred)

