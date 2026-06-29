import pandas as pd
import statsmodels.api as sm
class SimpleLinearRegression:
    def __init__(self,X,y):
        self.X=X
        self.y=y
    def fit(self):
        X_with_const=sm.add_constant(self.X)
        model=sm.OLS(self.y,X_with_const).fit()
        return model
    def summary(self,model):
        print(model.summary())