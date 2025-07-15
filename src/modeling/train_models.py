from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

def get_models():
    return {
        "SVM": svm.SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "NaiveBayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "DecisionTree": DecisionTreeClassifier(random_state=0)
    }
