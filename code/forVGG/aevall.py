from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_score

y_test=["a","b","p","b","b","b","b","p","b","p","b","b","p","p","p","a"]
y_pred=["a","b","p","p","p","p","b","p","b","p","b","b","a","a","p","b"]
print(confusion_matrix(y_test, y_pred,labels=["a", "b","p"]))


print(classification_report(y_test,y_pred))
#print(recall_score(y_test, y_pred) )
print(precision_score(y_test, y_pred))


