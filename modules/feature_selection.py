import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class PlotCharts:
    
    def plot_corr_matrix(self, corr_matrix: pd.DataFrame, method: str) -> None:
        '''
        Plot heatmap chart
        '''
        plt.figure(figsize=(15,10))
        sns.heatmap(corr_matrix, annot=True, cmap='vlag')
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.show()
        
    def plot_barh(self, range_importances: list, feature_importances: list, feature_names: list, title: str) -> None:
        '''
        Plot horizontal bar
        '''
        plt.barh(range_importances, feature_importances, align='center')
        plt.yticks(range_importances, feature_names)
        plt.xlabel('Feature Importance')
        plt.title(f'{title}')
        plt.show()


class FeatureSelection:
        
    def __init__(self, df: pd.DataFrame, target: str) -> None:
        self.df = df.copy()
        self.target = target
        
        self.df_features = self.df.drop(self.target, axis=1)
        self.df_target = self.df[[self.target]]
        
        self.feature_names = self.df_features.columns
        
        self.array_features = self.df_features.to_numpy()
        self.array_target = self.df_target[self.target].to_numpy()
        
        self.plt = PlotCharts()
        

    def correlation(self, method:str, n_features:int):
        '''
        Plot correlation methods for the dataset
        '''
        corr_matrix = self.df.corr(method=method).round(2)
        
        # Get the correlation coefficients of each variable with the target (Response)
        target_correlations = corr_matrix[self.target].abs().sort_values(ascending=False)
        
        # Select the number of n_features with the highest correlation with the target (Response)
        top_features = target_correlations[1:n_features+1].index.tolist()
        
        # Print the names of the top n features
        # print("Top", n_features, f"features with the highest correlation (Spearman) with the target ({self.target}): ")
        # for ft in range(len(top_features)):
        #   print(top_features[ft], ":", target_correlations[ft+1])
        
        self.plt.plot_corr_matrix(corr_matrix, method)
        
