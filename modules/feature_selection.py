import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
        

    def correlation(self, method:str):
        '''
        Plot correlation methods for the dataset
        '''
        corr_matrix = self.df.corr(method=method).round(2)
        
        # Get the correlation coefficients of each variable with the target (Response)
        target_correlations = corr_matrix[self.target].sort_values(ascending=False)
        print(target_correlations)
        
        self.plt.plot_corr_matrix(corr_matrix, method)

        return corr_matrix
    

    def calculate_variance_inflation_factor(self):
        # X is the DataFrame with independent variables
        X = self.df.drop(columns=self.target)
        vif_data = pd.DataFrame()

        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data = vif_data.sort_values(by=["VIF"], ascending=False).reset_index(drop=True)

        print(vif_data)
        return vif_data
    

    def pca_feature_selection(self):
        # Initialize PCA and fit the our features data after Feature Engineering steps
        pca = PCA(n_components=None)  # Keep all components initially
        features_pca = pca.fit_transform(self.df.drop(columns=[self.target]))

        # Calculate the cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Select the number of components that explain at least 95% of the variance
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"There are {n_components} components that explain at least 95%.")

        # Redefinition of PCA with the selected number of components
        pca = PCA(n_components=n_components)
        features_pca_reduced = pca.fit_transform(self.df.drop(columns=[self.target]))

        return features_pca_reduced
        
