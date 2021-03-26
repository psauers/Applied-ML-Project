from sklearn.linear_model import SGDClassifier

class SGD():
    #TODO: add customizable properties
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = SGDClassifier()

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def run(self):
        y_pred = self.model.predict(X_test)
        return y_pred