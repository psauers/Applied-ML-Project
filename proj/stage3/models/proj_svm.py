from sklearn.svm import SVC as svc, LinearSVC as lsvc

class SVM():
    #TODO: add customizable properties
    def __init__(self, X_train, X_test, y_train, y_test, linear=True):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if (linear):
            self.model = svc(kernel='linear', C=1.0, random_state=1)
        else:
            self.model = svc(kernel='rbf', C=10.0, random_state=1, gamma=0.10)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        return y_pred
