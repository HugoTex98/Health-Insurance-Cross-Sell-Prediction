import pandas  as pd
import numpy as np
from math import floor, ceil, log10
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


class NumericalAnalyses:
    """
    A class for performing various numerical analyses and visualizations on a pandas DataFrame.
    
    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    original_df : pd.DataFrame
        A copy of the original DataFrame for reference.
    
    Methods
    -------
    boxplots(col: str):
        Displays a boxplot for a specified column, along with annotated statistics such as mean, median, min, and max.
        
    hist(column: str, **kwargs):
        Displays a histogram for a specified column.
        
    round_to_nearest_order(number: float) -> int:
        Rounds a number to the nearest order of magnitude.
        
    hist_by_hue(column: str, hue: str, binwidth: int):
        Displays a histogram for a specified column, with an optional hue parameter to group the data.
        
    scatter(column_x: str, column_y: str):
        Displays a scatter plot for two specified columns.
    """


    def __init__(self, df: pd.DataFrame):
        """
        Initializes the NumericalAnalyses class with the given DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be analyzed.
        """
        self.df = df
        self.original_df = self.df.copy()


    def boxplots(self, col: str):
        """
        Displays a boxplot for a specified column, with manually written statistics (mean, median, min, and max).
        
        Parameters
        ----------
        col : str
            The name of the column for which to display the boxplot.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(self.df[col], orient='h')
        plt.axvline(x=self.df[col].mean(), color='g', linestyle='--')

        text_metada = [
            {'statistics': 'mean', 'value': 0.48, 'pos': 'center'},
            {'statistics': 'median', 'value': -0.42, 'pos': 'center'},
            {'statistics': 'max', 'value': 0.48, 'pos': 'right'},
            {'statistics': 'min', 'value': 0.48, 'pos': 'left'}
        ]
        
        for text_specifics in text_metada:
            function_reference = getattr(pd.Series, text_specifics['statistics'], None)
            
            if text_specifics['statistics'] == 'mean':
                message = f"{text_specifics['statistics'].capitalize()}: {self.df[col].mean():.2f}"
            elif text_specifics['statistics'] == 'median':
                message = f"{text_specifics['statistics'].capitalize()}: {self.df[col].median():.2f}"
            elif text_specifics['statistics'] == 'max':
                message = f"{text_specifics['statistics'].capitalize()}: {self.df[col].max():.2f}"
            elif text_specifics['statistics'] == 'min':
                message = f"{text_specifics['statistics'].capitalize()}: {self.df[col].min():.2f}"
                
            plt.text(function_reference(self.df[col]), text_specifics['value'], message, ha='center', fontsize=10)
        
        plt.title(f'{col} Distribution in the Dataset')
        plt.xlabel(col)
        plt.show()


    def hist(self, column: str, **kwargs):
        """
        Displays a histogram for a specified column.
        
        Parameters
        ----------
        column : str
            The name of the column for which to display the histogram.
        kwargs : dict
            Additional keyword arguments for the histogram plot.
        """
        plt.figure(figsize=(10, 6))
        minn, maxx = int(floor(self.df[column].min())), int(floor(self.df[column].max()))
        
        step = (maxx - minn) // 50 + 1
        sns.histplot(self.df[column], bins=len(range(minn, maxx, step)))
        # sns.histplot(self.df[column], bins=len(range(minn, maxx)))


    def round_to_nearest_order(self, number: float) -> int:
        """
        Rounds a given number to the nearest order of magnitude.
        
        Parameters
        ----------
        number : float
            The number to be rounded.
        
        Returns
        -------
        int
            The number rounded to the nearest order of magnitude.
        """
        order = 10 ** ceil(log10(number))
        rounded_number = ceil(number / order) * order
        
        return rounded_number
    
    
    def hist_by_hue(self, column: str, hue: str, binwidth: int):
        """
        Displays a histogram for a specified column, with data grouped by a hue parameter.
        
        Parameters
        ----------
        column : str
            The name of the column for which to display the histogram.
        hue : str
            The name of the column used to group the data.
        binwidth : int
            The width of the bins in the histogram.
        """
        plt.figure(figsize=(10, 6))

        ax = sns.histplot(data=self.df, x=column, hue=hue, bins=range(self.df[column].min(), self.df[column].max() + binwidth, binwidth), kde=False)
        
        y_values = list()
        for p in ax.patches:
            y_values.append(p.get_height())
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
        
        percent = 0.1 * max(y_values)
        step = self.round_to_nearest_order(percent)
        
        plt.xticks(range(0, self.df[column].max(), 10))
        plt.yticks(range(0, int(max(y_values)) + int(step), int(step)))
        
        plt.show()

        
    def scatter(self, column_x: str, column_y: str):
        """
        Displays a scatter plot for two specified columns.
        
        Parameters
        ----------
        column_x : str
            The name of the column for the x-axis.
        column_y : str
            The name of the column for the y-axis.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x=column_x, y=column_y)
        plt.show()
