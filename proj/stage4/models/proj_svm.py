from sklearn.svm import SVC as svc, LinearSVC as lsvc

class SVM():
    def __init__(self, kernel, c_num, gamma):
        self.model = svc(kernel=kernel, C=c_num, random_state=1, gamma=gamma, max_iter=15)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
