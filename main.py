from SRS.assumption_test import LinearRegressionAssumptionTest
from SRS.data_preprocess import DataPreprocess
from SRS.data_ingest import DataIngest
from SRS.model_build import SimpleLinearRegression
if __name__=="__main__":
    # Data Ingestion
    data_ingest = DataIngest(r'E:\ML PROJECTS\Simple Linear regression\Advertising_sales\data\advertising.csv')
    #print(data_ingest.load_data().head())
    X, y, df = data_ingest.get_X_y()
    df.to_csv(r'E:\ML PROJECTS\Simple Linear regression\Advertising_sales\data\processed_data.csv', index=False)
    print(df)
    # Data Preprocessing
    data_preprocessor = DataPreprocess(df)
    print(data_preprocessor.identify_outliers(df["TV"]))

    # Model Building
    model_builder = SimpleLinearRegression(X, y)
    model = model_builder.fit()
    model_builder.summary(model)

    # Assumption Testing
    assumption_tester = LinearRegressionAssumptionTest(X, y)
    assumption_tester.test_linearity()
    assumption_tester.test_independence()
    assumption_tester.test_homoscedasticity()
    assumption_tester.test_normality()
