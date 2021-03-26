from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

class Boost():
    #TODO: add customizable properties
    def __init__(self, X_train, X_test, y_train, y_test, n_est=500):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_estimators = n_est
        self.tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)
        self.model = AdaBoostClassifier(base_estimator=self.tree, n_estimators=n_est, learning_rate=0.1, random_state=1)

    def fit(self):
        self.svm.fit(self.X_train, self.y_train)
        self.tree = self.tree.fit(self.X_train, self.y_train)
        self.model = self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        return y_pred
