import pandas  as pd
from sklearn.model_selection import train_test_split


class Utils:
    """
    A utility class for performing common data manipulation tasks on pandas DataFrames.
    
    Methods
    -------
    drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        Drops specified columns from the DataFrame.
        
    replace_column_values(df: pd.DataFrame, col: str, values: dict) -> pd.DataFrame:
        Replaces values in a specified column according to a dictionary mapping.
        
    fill_column_values(df: pd.DataFrame, col: str, value: any) -> pd.DataFrame:
        Fills missing values (NA/NaN) in a specified column with a given value.
        
    split_df(df: pd.DataFrame, test_size: float) -> pd.DataFrame:
        Splits the DataFrame into train and test sets.
        
    split_df_stratified(df: pd.DataFrame, test_size: float, col: str) -> pd.DataFrame:
        Splits the DataFrame into train and test sets while maintaining class proportions in the specified column.
        
    split_df_by_class(df: pd.DataFrame, col: str, majority_class_value: int, minority_class_value: int) -> pd.DataFrame:
        Separates rows of the DataFrame into two sets based on the values of a specified column, typically used for separating majority and minority classes.
    """


    def drop_columns(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame from which to drop columns.
        cols : list
            A list of column names to be dropped.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame without the specified columns.
        """
        return df.drop(columns=cols)
    

    def replace_column_values(self, df: pd.DataFrame, col: str, values: dict) -> pd.DataFrame:
        """
        Replaces values in a specified column according to a dictionary mapping.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to modify.
        col : str
            The name of the column whose values are to be replaced.
        values : dict
            A dictionary where keys are existing values and values are the replacements.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with replaced values in the specified column.
        """
        return df[col].replace(values)
    
        
    def fill_column_values(self, df: pd.DataFrame, col: str, value: any) -> pd.DataFrame:
        """
        Fills missing values (NA/NaN) in a specified column with a given value.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to modify.
        col : str
            The name of the column whose missing values are to be filled.
        value : any
            The value to use for filling missing entries.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with missing values filled in the specified column.
        """
        return df[col].fillna(value)
        

    def split_df(self, df: pd.DataFrame, test_size: float) -> pd.DataFrame:
        """
        Splits the DataFrame into train and test sets.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be split.
        test_size : float
            The proportion of the dataset to include in the test split.
        
        Returns
        -------
        tuple
            A tuple containing the training and test sets (train, test).
        """
        return train_test_split(df, test_size=test_size, random_state=42)
    
        
    def split_df_stratified(self, df: pd.DataFrame, test_size: float, col: str) -> pd.DataFrame:
        """
        Splits the DataFrame into train and test sets while maintaining class proportions in the specified column.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be split.
        test_size : float
            The proportion of the dataset to include in the test split.
        col : str
            The column to use for stratification.
        
        Returns
        -------
        tuple
            A tuple containing the stratified training and test sets (train, test).
        """
        return train_test_split(df, test_size=test_size, random_state=42, stratify=df[col])
    
        
    def split_df_by_class(self, df: pd.DataFrame, col: str, majority_class_value: int, minority_class_value: int) -> pd.DataFrame:
        """
        Separates rows of the DataFrame into two sets based on the values of a specified column, typically used for separating majority and minority classes.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to separate by.
        col : str
            The name of the column to use for separating classes.
        majority_class_value : int
            The value representing the majority class.
        minority_class_value : int
            The value representing the minority class.
        
        Returns
        -------
        tuple
            A tuple containing two DataFrames: one for the majority class and one for the minority class.
        """
        return df[df[col] == majority_class_value], df[df[col] == minority_class_value]
