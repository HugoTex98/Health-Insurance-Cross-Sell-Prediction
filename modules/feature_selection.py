import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from probatus.feature_elimination import ShapRFECV


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
    

    def recursive_feature_elimination(self, model, keep_features: int, step=int):
        # List to stoer the features to keep
        features_to_keep = []

        # Define RFE selector with n features
        selector = RFE(model, n_features_to_select=keep_features, step=step)

        # Fit RFE selector
        selector = selector.fit(self.df.drop(columns=[self.target]), 
                                self.df[self.target])

        # Get feature ranking
        rankings = selector.ranking_

        # Print results
        for rank, feature in sorted(zip(rankings, self.df.drop(columns=[self.target]).columns)):
            print(f"Rank {rank}: {feature}")
            if rank == 1:
                features_to_keep.append(feature)

        return features_to_keep
    

    def shap_rfecv_selection(self, model, step: float, cv: int, scoring: str, 
                             eval_metric: str, keep_features: int):
        
        # n_jobs=-1 means to run the process in all available cores
        shap_elimination = ShapRFECV(model=model, step=step, cv=cv,
                                     scoring=scoring, eval_metric=eval_metric,
                                     n_jobs=-1) 
        
        # It shows the score results for each iteration of features to maintain
        report = shap_elimination.fit_compute(self.df.drop(columns=[self.target]),
                                              self.df[self.target],
                                              check_additivity=False)
        
        print(report)
        # Plot to compare the score defined in `ShapRFECV` (i.e: F1-Score or ROC-AUC) 
        # in Training and Validation
        shap_elimination.plot()

        # Get final feature set with a nº of features
        features_to_maintain = shap_elimination.get_reduced_features_set(num_features=keep_features)

        return features_to_maintain
    

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
        
