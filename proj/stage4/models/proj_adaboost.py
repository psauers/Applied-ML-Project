from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

class Boost():
    #TODO: add customizable properties
    def __init__(self, n_est=500):
        self.n_estimators = n_est
        self.tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)
        self.model = AdaBoostClassifier(base_estimator=self.tree, n_estimators=n_est, learning_rate=0.1, random_state=1)

    def fit(self, X_train, y_train):
        self.tree.fit(X_train, y_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def tree_score(self, X_test, y_test):
        return self.tree.score(X_test, y_test)
