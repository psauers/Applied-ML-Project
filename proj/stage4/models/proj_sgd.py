from sklearn.linear_model import SGDClassifier

class SGD():
    def __init__(self):
        self.model = SGDClassifier(max_iter=15)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def run(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)