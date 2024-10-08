import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, 
                                   OneHotEncoder, OrdinalEncoder)


class FeatureEngineering:
    """
    A class for performing various feature engineering techniques on a pandas DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame to be transformed.

    Methods
    -------
    minmaxscale_numeric_columns(cols: list):
        Normalizes numerical columns using MinMaxScaler to scale values between 0 and 1.
        
    standardscale_numeric_columns(cols: list):
        Standardizes numerical columns by removing the mean and scaling to unit variance.
        
    truncate_numeric_columns_by_percentis(cols: list, percentile: float):
        Truncates values in numerical columns beyond a specified percentile to handle outliers.
        
    probability_ratio_encoding(feature_column: str, target_column: str):
        Encodes a categorical feature by calculating the probability ratio between a feature and the target column.
        
    target_encoding(feature_column: str, target_column: str):
        Encodes a categorical feature by assigning the mean target value for each category.
        
    leave_one_out_encoding(feature_columns: list, target_column: str):
        Encodes categorical features using leave-one-out strategy, avoiding target leakage by excluding the current row.
        
    frequency_encoding(col: str):
        Encodes a categorical feature by its frequency in the DataFrame.
        
    one_hot_encoding(col: str):
        Performs one-hot encoding on a categorical feature with multiple categories.
        
    ordinal_encoding(col: str, categories: list):
        Encodes ordinal features based on a predefined order of categories.
        
    categorical_encoding(col: str, mapping: dict):
        Encodes binary categorical features using a provided mapping dictionary.
        
    dummies_encoding(col: str):
        Performs one-hot encoding using pandas' `get_dummies` function on a categorical feature.
    """
    
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the FeatureEngineering class with the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be transformed.
        """
        self.df = df.copy()


    def minmaxscale_numeric_columns(self, cols: list):
        """
        Normalizes numerical columns using MinMaxScaler, scaling values between 0 and 1.
        Useful when features do not follow a Gaussian distribution, or when preserving relative distances between data points.

        Parameters
        ----------
        cols : list
            List of numerical columns to be normalized.
        """
        scaler = MinMaxScaler()
        scaler.fit(self.df[cols])
        self.df[cols] = scaler.transform(self.df[cols])


    def standardscale_numeric_columns(self, cols: list):
        """
        Standardizes numerical columns by removing the mean and scaling to unit variance.
        Used when features follow a Gaussian distribution.

        Parameters
        ----------
        cols : list
            List of numerical columns to be standardized.
        """
        scaler = StandardScaler()
        scaler.fit(self.df[cols])
        self.df[cols] = scaler.transform(self.df[cols])


    def truncate_numeric_columns_by_percentis(self, cols: list, percentile: float):
        """
        Truncates values in numerical columns based on a specific percentile to handle outliers.
        Any value above the specified percentile is truncated to that threshold.

        Parameters
        ----------
        cols : list
            List of numerical columns to truncate.
        percentile : float
            The percentile to use for truncating values (e.g., 0.95 for the 95th percentile).
        """
        for c in cols:
            threshold = self.df[c].quantile(percentile)
            self.df[c] = self.df[c].apply(lambda x: threshold if x > threshold else x)


    def probability_ratio_encoding(self, feature_column: str, target_column: str):
        """
        Encodes a categorical feature by calculating the probability ratio of the target variable given the feature.
        The ratio is the probability of the target being positive divided by the probability of it being negative.

        Parameters
        ----------
        feature_column : str
            The categorical column to encode.
        target_column : str
            The target column used to calculate probability ratios.
        """
        category_probs = self.df.groupby(feature_column)[target_column].mean()
        epsilon = 1e-8  # To avoid division by 0
        category_ratios = category_probs / (1 - category_probs + epsilon)
        self.df[feature_column] = self.df[feature_column].map(category_ratios)


    def target_encoding(self, feature_column: str, target_column: str):
        """
        Encodes a categorical feature by assigning the mean target value for each category.
        The encoding replaces each category with the mean of the target variable within that category.

        Parameters
        ----------
        feature_column : str
            The categorical column to encode.
        target_column : str
            The target column used to calculate the mean values.
        """
        target_means = self.df.groupby(feature_column)[target_column].mean()
        self.df[feature_column] = self.df[feature_column].map(target_means)


    def leave_one_out_encoding(self, feature_columns: list, target_column: str):
        """
        Performs leave-one-out encoding, where the mean target value for each category is calculated,
        excluding the current row to avoid overfitting. Adds Gaussian noise for regularization.

        Parameters
        ----------
        feature_columns : list
            List of categorical columns to encode.
        target_column : str
            The target column used to calculate leave-one-out encoding.
        """
        encoder = ce.LeaveOneOutEncoder(cols=feature_columns, sigma=0.1)
        self.df = encoder.fit_transform(self.df, self.df[target_column])


    def frequency_encoding(self, col: str):
        """
        Encodes a categorical feature by assigning values based on the frequency of each category in the DataFrame.

        Parameters
        ----------
        col : str
            The name of the categorical column to encode.
        """
        freq_encoding = self.df[col].value_counts(normalize=True).to_dict()
        self.df[col] = self.df[col].map(freq_encoding)


    def one_hot_encoding(self, col: str):
        """
        Performs one-hot encoding on a categorical feature with more than two categories.
        Converts each category into separate binary columns.

        Parameters
        ----------
        col : str
            The name of the categorical column to encode.
        """
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(self.df[[col]])
        encoded_data = encoder.transform(self.df[[col]])

        # Create a DataFrame with the encoded values
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))

        # Concatenate the encoded DataFrame with the original DataFrame
        self.df = pd.concat([self.df, encoded_df], axis=1)

        # Drop the original column
        self.df = self.df.drop(col, axis=1)


    def ordinal_encoding(self, col: str, categories: list):
        """
        Encodes an ordinal feature by assigning integer values based on the order of categories.

        Parameters
        ----------
        col : str
            The name of the ordinal column to encode.
        categories : list
            List of categories in the order they should be encoded.
        """
        encoder = OrdinalEncoder(categories=[categories])
        encoder.fit(self.df[[col]])
        self.df[col] = encoder.transform(self.df[[col]])
        self.df[col] = self.df[col].astype(int)


    def categorical_encoding(self, col: str, mapping: dict):
        """
        Encodes binary categorical features using a provided mapping dictionary.

        Parameters
        ----------
        col : str
            The name of the binary categorical column to encode.
        mapping : dict
            Dictionary that maps the categorical values to numerical values.
        """
        self.df[col] = self.df[col].replace(mapping)
        self.df[col] = self.df[col].astype(int)
        

    def dummies_encoding(self, col: str):
        """
        Performs one-hot encoding using pandas' `get_dummies` function on a categorical feature, 
        creating binary columns for each category.

        Parameters
        ----------
        col : str
            The name of the categorical column to encode.
        """
        self.df = pd.get_dummies(self.df, columns=[col], prefix=col, drop_first=True)

