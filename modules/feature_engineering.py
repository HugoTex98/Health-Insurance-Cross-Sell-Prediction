import pandas  as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
        Normalizes numerical columns using MinMaxScaler.
        
    truncate_numeric_columns_by_percentis(cols: list, percentile: float):
        Truncates values in numerical columns beyond a specified percentile to handle outliers.
        
    probability_ratio_encoding(feature_columns: list, target_column: str):
        Encodes categorical features based on the probability ratio between a feature and the target column.
        
    frequency_encoding(col: str):
        Encodes a categorical feature based on its frequency in the DataFrame.
        
    categorical_encoding(col: str, mapping: dict):
        Encodes binary categorical features using a provided mapping dictionary.
        
    one_hot_encoding(col: str):
        Performs one-hot encoding on a categorical feature with more than two possible values.
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
        Normalizes numerical columns using MinMaxScaler. When features do not follow a normal 
        (Gaussian) distribution or we need to preserve the relative distance between data points.

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
        Normalizes numerical columns using MinMaxScaler. When features do not follow a normal 
        (Gaussian) distribution or we need to preserve the relative distance between data points.

        Parameters
        ----------
        cols : list
            List of numerical columns to be normalized.
        """
        scaler = StandardScaler()
        scaler.fit(self.df[cols])
        self.df[cols] = scaler.transform(self.df[cols])


    def truncate_numeric_columns_by_percentis(self, cols: list, percentile: float):
        """
        Truncates values in numerical columns based on a specific percentile to handle outliers.

        Parameters
        ----------
        cols : list
            List of numerical columns to truncate.
        percentile : float
            The percentile to use for truncating values (e.g., 0.95 for the 95th percentile).
        """
        for c in cols:
            train_percentis = self.df[c].quantile([percentile])
            train_threshold = train_percentis[percentile]
            self.df[c] = self.df[c].apply(lambda x: train_threshold if x > train_threshold else x)
            

    def probability_ratio_encoding(self, feature_column: str, target_column: str):
        """
        Encodes categorical feature based on the probability ratio between a feature and the target column.

        Parameters
        ----------
        feature_columns : str
            The categorical column to encode.
        target_column : str
            The target column used to calculate probability ratios.
        """
        category_probs = self.df.groupby(feature_column)[target_column].mean()
        epsilon = 1e-8  # To avoid division by 0
        category_ratios = category_probs / (1 - category_probs + epsilon)
        self.df[feature_column] = self.df[feature_column].map(category_ratios)
            

    def frequency_encoding(self, col: str):
        """
        Encodes a categorical feature based on its frequency in the DataFrame.

        Parameters
        ----------
        col : str
            The name of the column to be frequency encoded.
        """
        freq_encoding = self.df[col].value_counts(normalize=True).to_dict()
        self.df[col] = self.df[col].map(freq_encoding)


    def categorical_encoding(self, col: str, mapping: dict):
        """
        Encodes binary categorical features using a provided mapping dictionary.

        Parameters
        ----------
        col : str
            The name of the binary categorical column to encode.
        mapping : dict
            A dictionary that maps the binary values to numerical representations.
        """
        self.df[col].replace(mapping, inplace=True)


    def one_hot_encoding(self, col: str):
        """
        Performs one-hot encoding on a categorical feature with more than two possible values.

        Parameters
        ----------
        col : str
            The name of the column to be one-hot encoded.
        """
        self.df = pd.get_dummies(self.df, columns=[col], prefix=col, drop_first=True)
