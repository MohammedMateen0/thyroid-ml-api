from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate(model, X, y):
    preds = model.predict(X)

    print(classification_report(y, preds))
    print("Confusion Matrix:\n", confusion_matrix(y, preds))