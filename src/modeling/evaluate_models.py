from sklearn import metrics

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    return {"accuracy": accuracy, "auc": auc, "confusion_matrix": metrics.confusion_matrix(y_test, y_pred)}
