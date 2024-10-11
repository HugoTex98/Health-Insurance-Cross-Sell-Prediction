import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss


class Undersampling:
    """
    A class for performing various data augmentation techniques such as oversampling and undersampling 
    to handle class imbalance in datasets.

    Attributes
    ----------
    train_features : pd.DataFrame
        Features of the training data (excluding the target column).
    train_target : pd.DataFrame
        Target column of the training data.
    target : str
        Name of the target column.

    Methods
    -------
    random_undersampling():
        Perform random undersampling to randomly reduce the number of majority class samples.

    """


    def __init__(self, train_data: pd.DataFrame, target: str):
        """
        Initializes the Augmentation class with training data and the target column.

        Parameters
        ----------
        train_data : pd.DataFrame
            The complete training dataset containing features and target.
        target : str
            The name of the target column.
        """
        self.train_features = train_data.drop(columns=target)
        self.train_target = train_data[[target]]
        self.target = target


    def random_undersampling(self):
        """
        Perform random undersampling to reduce the number of majority class samples.
        
        Random undersampling randomly selects a subset of the majority class to balance the class distribution.
        """
        undersampler = RandomUnderSampler(random_state=42)
        self.undersamp_train_features_resampled, self.undersamp_train_target_resampled = undersampler.fit_resample(self.train_features, 
                                                                                                                   self.train_target)
        self.undersamp_df_train_modified = pd.concat([self.undersamp_train_features_resampled, 
                                                      self.undersamp_train_target_resampled], 
                                                      axis=1, join='inner')
        
    
    def near_miss_undersampling(self, version: int=3):
        # summarize class distribution
        counter = self.train_target.value_counts()
        print(f'Class distribution before undersampling: {counter}')

        # define the undersampling method
        near_miss_undersample = NearMiss(version=version, n_neighbors_ver3=3)

        # transform the dataset
        self.near_miss_train_features, self.near_miss_train_target = near_miss_undersample.fit_resample(self.train_features,
                                                                                                        self.train_target)
        
        self.near_miss_df_train_modified = pd.concat([self.near_miss_train_features, 
                                                      self.near_miss_train_target], 
                                                      axis=1, join='inner')
        
        # summarize the new class distribution
        counter = self.near_miss_train_target.value_counts()
        print(f'\nClass distribution after undersampling: {counter}')

        
        

    def cluster_centroids_undersampler(self, num_clusters: int) -> pd.DataFrame:
        '''
        Method that under samples the majority class by replacing a cluster of majority samples by the cluster centroid 
        of a KMeans algorithm. 
        This algorithm keeps N majority samples by fitting the KMeans algorithm with N cluster to the majority class 
        and using the coordinates of the N cluster centroids as the new majority samples.
        '''
        # summarize class distribution
        counter = self.train_target.value_counts()
        print(f'Class distribution before undersampling: {counter}')

        # Apply Cluster Centroids undersampling
        cc = ClusterCentroids(random_state=42)
        self.cc_train_features, self.cc_train_target = cc.fit_resample(self.train_features,
                                                                       self.train_target)
        
        self.cc_df_train_modified = pd.concat([self.cc_train_features,
                                                    self.cc_train_target],
                                                    axis=1, join='inner')

        # summarize the new class distribution
        counter = self.cc_train_target.value_counts()
        print(f'\nClass distribution after undersampling: {counter}')

        
