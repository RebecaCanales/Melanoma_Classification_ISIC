from sklearn import svm
from sklearn.model_selection import cross_val_score

def svm_cross_validation(X_train, y_train, c_values, gamma_values, kernels, cv=5):
    best_score = 0
    best_params = None
    for c in c_values:
        for gamma in gamma_values:
            for kernel in kernels:
                clf = svm.SVC(C=c, kernel=kernel, gamma=gamma)
                scores = cross_val_score(clf, X_train, y_train, cv=cv)
                mean_score = scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {"C": c, "gamma": gamma, "kernel": kernel}
    return best_params, best_score
