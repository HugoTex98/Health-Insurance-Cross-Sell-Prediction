import numpy as np
import pandas  as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    
    
    def logistic_regression_coefficients(self):
        '''
        Plot the Logitic Regression Coefficients for feature importance technique
        '''
        lr = LogisticRegression()
        lr.fit(self.array_features, self.array_target)
        
        # Get feature importance scores
        coef = lr.coef_[0]
        feature_importance = abs(coef)

        # Rank features by importance
        feature_importance_ranked = np.sort(feature_importance, kind="stable")
        sorted_featn = []
        for index in range(len(feature_importance_ranked)):
          sorted_featn.append(self.feature_names[index])
        #   print(f"{self.feature_names[index]}: {feature_importance_ranked[index]}")
        
        range_importances = range(self.array_features.shape[1])
        feature_importances = feature_importance_ranked
        feature_names = sorted_featn
        title = 'Logistic Regression Coeficients (LRC)'

        self.plt.plot_barh(range_importances=range_importances, feature_importances=feature_importances, feature_names=feature_names, title=title)
        
        return feature_names
        
    def random_forest(self, n_estimators: int):
        '''
        Plot the Random Forest feature importance technique
        '''      
        rfc = RandomForestClassifier(n_estimators=n_estimators, bootstrap=False, random_state=42)
        rfc.fit(self.array_features, self.array_target)
        
        # Compute feature importance scores
        rfc_fe_importances = rfc.feature_importances_
        sorted_idx = rfc_fe_importances.argsort()
        
        range_importances = range(self.array_features.shape[1])
        feature_importances = rfc_fe_importances[sorted_idx]
        feature_names = self.feature_names[sorted_idx]
        title = 'Random Forest Classifier (RFC) Feature Importance'

        self.plt.plot_barh(range_importances=range_importances, feature_importances=feature_importances, feature_names=feature_names, title=title)
        
        return feature_names
        
    def anova(self, n_features: int):
        '''
        Plot the ANOVA feature importance 
        '''        
        # apply ANOVA feature selection
        anova_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_new = anova_selector.fit_transform(self.array_features, self.array_target)
        
        # get the ANOVA scores and feature names
        scores = anova_selector.scores_
        # sort the features by their scores
        sorted_idx = np.argsort(scores)[::1]
        
        range_importances = range(self.array_features.shape[1])
        feature_importances = scores[sorted_idx]
        feature_names = self.feature_names[sorted_idx]
        title = 'ANOVA Feature Importance'

        self.plt.plot_barh(range_importances=range_importances, feature_importances=feature_importances, feature_names=feature_names, title=title)
        
        return feature_names
        
    def recursive_feature_elimination(self, n_features: int):
        '''
        Run the Recursive Feature Elimination feature importance technique on the RandomForestClassifier
        '''
        rfc = RandomForestClassifier(n_estimators=100, bootstrap=False, random_state=42)
        rfc.fit(self.array_features, self.array_target)
        
        # Define RFE selector with n features
        rfe_selector = RFE(rfc, n_features_to_select=n_features, step=1)
        
        # Fit RFE selector
        selector = rfe_selector.fit(self.array_features, self.array_target)
        
        # Get feature ranking
        rankings = selector.ranking_
        sorted_idx = sorted(rankings) #np.sort(rankings, kind='stable')
        
        # Print results
        for rank, feature in sorted(zip(rankings, feature_names)):
            print(f"Rank {rank}: {feature}")
        
        range_importances = range(self.array_features.shape[1])
        feature_importances = rankings[sorted_idx]
        feature_names = self.feature_names[sorted_idx]
        title = 'Recursive Feature Elimination (RFE)'
                
        self.plt.plot_barh(range_importances=range_importances, feature_importances=feature_importances, feature_names=feature_names, title=title)
        
        return feature_names
        
        
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