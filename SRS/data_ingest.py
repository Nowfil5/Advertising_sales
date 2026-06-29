import pandas as pd
class DataIngest:
    def __init__(self,file_path):
        self.file_path = file_path
    def load_data(self):
        data=pd.read_csv(self.file_path)
        return data
    def get_X_y(self):
        data=self.load_data()
        X=data[['TV']]
        y=data['Sales']
        df=pd.concat([X,y],axis=1)
        return X, y,df