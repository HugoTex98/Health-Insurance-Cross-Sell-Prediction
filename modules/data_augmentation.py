import pandas  as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids


class Augmentation:
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
    random_oversampling():
        Perform random oversampling to duplicate random samples from the minority class.

    smote_augmentation():
        Perform oversampling using the SMOTE algorithm to create new synthetic samples for the minority class.

    borderline_smote_augmentation():
        Perform oversampling using the Borderline SMOTE algorithm, which generates synthetic samples near the decision boundary.

    adasyn_augmentation():
        Perform oversampling using the ADASYN algorithm, which generates synthetic samples based on the distribution of the minority class.

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


    def random_oversampling(self):
        """
        Perform random oversampling on the minority class by duplicating random samples.
        
        This method duplicates random samples from the minority class to balance the class distribution.
        """
        oversampler = RandomOverSampler(random_state=42)
        self.rnd_train_features_resampled, self.rnd_train_target_resampled = oversampler.fit_resample(self.train_features, 
                                                                                                      self.train_target)
        self.rnd_df_train_modified = pd.concat([self.rnd_train_features_resampled, 
                                                self.rnd_train_target_resampled], 
                                                axis=1, join='inner')


    def smote_augmentation(self):
        """
        Perform SMOTE (Synthetic Minority Over-sampling Technique) to generate new synthetic samples for the minority class.
        
        SMOTE selects random samples and their nearest neighbors to create new samples by interpolating between them.
        """
        smote = SMOTE(random_state=42)
        self.smote_train_features_resampled, self.smote_train_target_resampled = smote.fit_resample(self.train_features, 
                                                                                                    self.train_target)
        self.smote_df_train_modified = pd.concat([self.smote_train_features_resampled, 
                                                  self.smote_train_target_resampled], 
                                                  axis=1, join='inner')


    def borderline_smote_augmentation(self):
        """
        Perform Borderline SMOTE augmentation to oversample the minority class, focusing on samples near the decision boundary.
        
        Borderline SMOTE generates synthetic samples for minority instances that are close to the decision boundary.
        """
        bordersmote = BorderlineSMOTE(random_state=42)
        self.bordersmote_train_features_resampled, self.bordersmote_train_target_resampled = bordersmote.fit_resample(self.train_features, 
                                                                                                                      self.train_target)
        self.bordersmote_df_train_modified = pd.concat([self.bordersmote_train_features_resampled, 
                                                        self.bordersmote_train_target_resampled], 
                                                        axis=1, join='inner')


    def adasyn_augmentation(self):
        """
        Perform oversampling using ADASYN (Adaptive Synthetic Sampling) to generate new synthetic samples for the minority class.
        
        ADASYN generates synthetic samples based on the local distribution of the minority class, focusing on regions with lower density.
        """
        adasyn = ADASYN(random_state=42)
        self.adasyn_train_features_resampled, self.adasyn_train_target_resampled = adasyn.fit_resample(self.train_features, 
                                                                                                       self.train_target)
        self.adasyn_df_train_modified = pd.concat([self.adasyn_train_features_resampled, 
                                                   self.adasyn_train_target_resampled], 
                                                   axis=1, join='inner')


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