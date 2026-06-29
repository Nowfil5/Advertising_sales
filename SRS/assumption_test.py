import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan

class LinearRegressionAssumptionTest:
    def __init__(self, X, y):
        self.X = sm.add_constant(X)
        self.y = y

        self.model = sm.OLS(self.y, self.X).fit()
        self.predictions = self.model.predict(self.X)
        self.residuals = self.y - self.predictions

    def test_linearity(self):
        """
        Check linearity using R-squared.
        Higher R² indicates a stronger linear relationship.
        """
        print("R-squared:", self.model.rsquared)

    def test_independence(self):
        """
        Durbin-Watson test for independence of errors.
        Value close to 2 indicates independence.
        """
        dw = durbin_watson(self.residuals)
        print("Durbin-Watson Statistic:", dw)

    def test_homoscedasticity(self):
        """
        Breusch-Pagan test for constant variance.
        p-value > 0.05 indicates homoscedasticity.
        """
        bp_test = het_breuschpagan(
            self.residuals,
            self.model.model.exog
        )

        print("Breusch-Pagan p-value:", bp_test[1])

    def test_normality(self):
        """
        Shapiro-Wilk test for normality of residuals.
        p-value > 0.05 indicates normal distribution.
        """
        stat, p = shapiro(self.residuals)

        print("Shapiro-Wilk Statistic:", stat)
        print("p-value:", p)

    def summary(self):
        print(self.model.summary())