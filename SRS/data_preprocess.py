import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class DataPreprocess:
    def __init__(self,df):
        self.df=df
    def identify_outliers(self,Data:pd.DataFrame):
        fig,ax=plt.subplots(figsize=(10,5))
        ax.boxplot(Data)
        ax.set_title('Boxplot for Outlier Detection')
        ax.set_xlabel('Features')
        ax.set_ylabel('Values')
        plt.show()
    def identify_outliers_z_score(self,data:pd.Series,threshold:float=3):
        mean=np.mean(data)
        std=np.std(data)
        z_scores=(data-mean)/std
        outliers=data[np.abs(z_scores)>threshold]
        return outliers
     

        