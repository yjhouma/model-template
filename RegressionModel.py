import numpy
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn import neighbors


class RegModel(object):

    def evaluate(self, X_val, y_val, func=None):
        guess = self.guess(X_val)
        if func != None:
            return func(guess, y_val)
        else:
            return self.rootmeansqrt(guess, y_val)

    def rootmeansqrt(self, y_pred, y_true):
        delta = (y_pred-y_true)**2
        return numpy.sqrt(np.sum(relative_err) / len(y_val))



class LinearModel(RegModel):

    def __init__(self, X, y):
        super().__init__()
        self.clf = linear_model.LinearRegression()
        self.clf.fit(X, y)

    def guess(self, feature):
        return self.clf.predict(feature)


class RF(RegModel):

    def __init__(self, X, y):
        super().__init__()
        self.clf = RandomForestRegressor(n_estimators=200, verbose=True,
                                         max_depth=35, min_samples_split=2,
                                         min_samples_leaf=1)
        self.clf.fit(X,y)

    def guess(self, feature):
        return self.clf.predict(feature)


class SVM(RegModel):
    def __init__(self, X, y):
        super().__init__()
        self.clf = SVR(kernel='linear', degree=3, gamma='auto', coef0=0.0,
                       tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                       cache_size=200, verbose=False, max_iter=-1)

        self.clf.fit(X, y)

    def guess(self, feature):
        return self.clf.predict(feature)


class XGBoost(RegModel):
    def __init__(self, X, y, rnd = 500):
        super().__init__()
        dtrain = xgb.DMatrix(X, label=y)
        evallist = [(dtrain, 'train')]
        param = {'nthread': -1,
                 'max_depth': 7,
                 'eta': 0.02,
                 'silent': 1,
                 'objective': 'reg:linear',
                 'colsample_bytree': 0.7,
                 'subsample': 0.7}
        num_round = rnd
        self.bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)

    def guess(self, feature):
        dtest = xgb.DMatrix(feature)
        return self.bst.predict(dtest)



class KNN(RegModel):

    def __init__(self, X, y, k=10):
        super().__init__()
        self.clf = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance', p=1)
        self.clf.fit(X_train, y_train)

    def guess(self, feature):
        return self.clf.predict(feature)
