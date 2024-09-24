import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CategoricalAnalyses:
    """
    A class for performing various categorical analyses and visualizations on a pandas DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    originial_df : pd.DataFrame
        A copy of the original DataFrame for reference.

    Methods
    -------
    count_frequency(column: str, **kwargs):
        Displays the frequency distribution of a categorical feature.
        
    groupedbar(column: str, target: str, title: str, feat_by_targ: bool):
        Displays the distribution of a feature conditioned on the target variable using a grouped bar chart.
        
    pie(columns: list[str]):
        Creates pie charts for the specified list of columns.
        
    filter_col_value(column: str, value: str, group_names: list, cols_wanted: list, include_ocurrences: bool) -> pd.DataFrame:
        Filters the DataFrame based on the specified column and value, allowing grouping and column selection.
    """


    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the CategoricalAnalyses class with the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be analyzed.
        """
        self.df = df.copy()
        self.originial_df = self.df.copy()
        

    def count_frequency(self, column: str, **kwargs):
        """
        Displays the frequency distribution of a feature. Uses kwargs to set plot properties.

        Parameters
        ----------
        column : str
            The name of the column for which to display the frequency distribution.
        kwargs : dict
            Additional keyword arguments for customizing the plot.
        """
        title = f'Frequency: {column}'
        
        if len(self.df[column].unique()) > 3:
            counts = self.df[column].value_counts()
            ordered_categories = counts.index
            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.df, x=column, order=ordered_categories)
            plt.xticks(rotation=90)
        else:
            new_column_list = list(self.df[column].value_counts().sort_index().index.values)
            new_count_list = list(self.df[column].value_counts().sort_index().values)
            data = {'Column': new_column_list, 'Count': new_count_list}
            df_plot = pd.DataFrame(data)

            df_plot['Column'] = df_plot['Column'].replace(['0', 'No', 'no', 0], f'{column}: No')
            df_plot['Column'] = df_plot['Column'].replace(['1', 'Yes', 'yes', 1], f'{column}: Yes')

            totals = df_plot['Count'].sum()
            plt.figure(figsize=(10, 6))
            sns.set_style("white")
            ax = sns.barplot(data=df_plot, x='Column', y='Count', color='#ABC9EA')

            y_lim = df_plot['Count'].max()
            ax.set(title=title, ylim=(0, y_lim + 50000))

            for k, p in enumerate(ax.patches):
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                value = int(y)
                percentage = f'{(y / totals) * 100:.0f}%'
                bbox = dict(boxstyle="round", fc="w", ec="black")
                ax.annotate(f'{value:,}  ({percentage})', (x, y + 6000), ha='center', va='bottom', bbox=bbox)
                
            plt.xticks(rotation=0)
        
        plt.title(title)
        plt.xlabel('')
        plt.ylabel('')
        plt.show()


    def groupedbar(self, column: str, target: str, title: str, feat_by_targ: bool):
        """
        Displays the distribution of a feature conditioned to the target variable using a grouped bar chart.

        Parameters
        ----------
        column : str
            The name of the feature for which to display the distribution.
        target : str
            The name of the target variable.
        title : str
            The title of the plot.
        feat_by_targ : bool
            If True, plot feature x target; if False, plot target x feature.
        """
        feature_list = sorted(list(self.df[column].unique()))
        target_list = sorted(list(self.df[target].unique()))
        new_feat_list = list()
        new_targ_list = list()
        new_count_list = list()

        for feat in feature_list:
            for targ in target_list:
                count = self.df.loc[(self.df[column] == feat) & (self.df[target] == targ), target].count()
                new_feat_list.append(feat)
                new_targ_list.append(targ)
                new_count_list.append(count)

        df_plot = pd.DataFrame({
            'Column': new_feat_list,
            'Target': new_targ_list,
            'Count': new_count_list
        })

        df_plot['Column'] = df_plot['Column'].replace(['0', 'No', 'no', 0], f'{column}: No')
        df_plot['Column'] = df_plot['Column'].replace(['1', 'Yes', 'yes', 1], f'{column}: Yes')
        
        plt.figure(figsize=(10, 6))
        sns.set_style("white")

        totals = []
        if feat_by_targ:
            for feat in feature_list:
                totals.append(df_plot[df_plot['Column'] == feat].Count.sum())
            ax = sns.barplot(x='Column', y='Count', hue='Target', data=df_plot, palette=sns.color_palette("pastel"))
        else:
            for targ in target_list:
                totals.append(df_plot[df_plot['Target'] == targ].Count.sum())
            ax = sns.barplot(x='Target', y='Count', hue='Column', data=df_plot, palette=sns.color_palette("pastel"))

        ax.set(title=title, ylim=(0, df_plot['Count'].max() + 50000))
        plt.xlabel('')
        plt.ylabel('')
        plt.legend(loc='upper right', ncols=2)

        for k, p in enumerate(ax.patches):
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            value = int(y)
            percentage = f'{(y / totals[k]) * 100:.0f}%'
            bbox = dict(boxstyle="round", fc="w", ec="black")
            ax.annotate(f'{value:,}  ({percentage})', (x, y + 6000), ha='center', va='bottom', bbox=bbox)
        
        plt.show()


    def pie(self, col: list[str]):
        """
        Creates pie chart for the specified column.

        Parameters
        ----------
        col : str
            The column to create pie charts.
        """
        counts = self.df[col].value_counts()
        total_observations = counts.sum()
        counts.plot.pie(autopct=lambda pct: '{:.2f}%\n({:.0f})'.format(pct, pct / 100 * total_observations))
        plt.title(col)
        plt.ylabel("")
        plt.show()


    def filter_col_value(self, column: str, value='value', group_names=None, cols_wanted=None, include_ocurrences=False):
        """
        Filters the DataFrame based on a specific column and value. Allows for grouping, column selection, and counting occurrences.

        Parameters
        ----------
        column : str
            The name of the column to filter by.
        value : str
            The value to filter on. Default is 'value'.
        group_names : list, optional
            List of columns to group the filtered data by. Default is None.
        cols_wanted : list, optional
            List of columns to include in the filtered DataFrame. Default is None.
        include_ocurrences : bool, optional
            Whether to include the count of occurrences in the filtered data. Default is False.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.
        """
        filtered = self.df[self.df[column] == value]
        if isinstance(group_names, list):
            filtered = filtered.groupby(by=group_names)
            if isinstance(cols_wanted, list):
                filtered = filtered.loc[:, cols_wanted]
            if include_ocurrences:
                filtered = filtered.count().to_frame().rename(columns={'id': 'count'}).reset_index()
        return filtered
