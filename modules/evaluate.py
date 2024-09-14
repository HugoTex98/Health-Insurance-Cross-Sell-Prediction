import pandas as pd
import numpy as np
# np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix, balanced_accuracy_score
from decimal import Decimal


class Evaluate:
    """
    A class for evaluating machine learning models using various metrics and plots.

    Attributes
    ----------
    X_valid : pd.DataFrame
        Validation set features.
    y_valid : pd.DataFrame
        Validation set target values.
    X_test : pd.DataFrame
        Test set features.
    y_test : pd.DataFrame
        Test set target values.
    min_samples : int
        Minimum number of samples to consider when calculating percentiles for predicted probabilities.

    Methods
    -------
    predictions(model):
        Get predictions for the validation and test sets from the model.
        
    feat_probabilities(model):
        Get predicted probabilities for the test set from the model.
        
    evaluation_metrics(model, model_name: str):
        Compute and print balanced accuracy, accuracy, precision, recall, and F1 score for validation and test sets.
        
    roc_auc_metric(model, model_name: str):
        Plot the ROC curve and compute the AUC score for the test set.
        
    confusion_matrix(model, model_name: str):
        Plot the confusion matrix for the test set.
        
    probs_to_percentiles(y_probs: list):
        Map predicted probabilities to percentiles and compute accuracy for each percentile group.
        
    plot_accuracy_by_percentile(model):
        Plot the accuracy by percentile and predicted probabilities for the test set.
    """


    def __init__(self, X_valid: pd.DataFrame, y_valid: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, min_samples: int) -> None:
        """
        Initializes the Evaluate class with the validation and test sets.

        Parameters
        ----------
        X_valid : pd.DataFrame
            Validation set features.
        y_valid : pd.DataFrame
            Validation set target values.
        X_test : pd.DataFrame
            Test set features.
        y_test : pd.DataFrame
            Test set target values.
        min_samples : int
            Minimum number of samples for percentile-based analysis.
        """
        self.X_valid = X_valid.copy()
        self.y_valid = y_valid.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.min_samples = min_samples


    def predictions(self, model):
        """
        Get predictions for the validation and test sets from the model.

        Parameters
        ----------
        model : object
            The machine learning model to generate predictions.

        Returns
        -------
        tuple
            y_pred_valid : ndarray
                Predicted values for the validation set.
            y_pred_test : ndarray
                Predicted values for the test set.
        """
        print(f'Model type: {type(model)}')
        y_pred_valid = model.predict(self.X_valid)
        y_pred_test = model.predict(self.X_test)
        
        return y_pred_valid, y_pred_test
    

    def feat_probabilities(self, model):
        """
        Get predicted probabilities for the test set from the model.

        Parameters
        ----------
        model : object
            The machine learning model to generate predicted probabilities.

        Returns
        -------
        ndarray
            Predicted probabilities for the test set.
        """
        y_probs = model.predict_proba(self.X_test)
        return y_probs
    

    def evaluation_metrics(self, model, model_name: str):
        """
        Compute and print balanced accuracy, accuracy, precision, recall, and F1 score for the validation and test sets.

        Parameters
        ----------
        model : object
            The machine learning model to evaluate.
        model_name : str
            The name of the model for display purposes.
        """
        y_pred_valid, y_pred_test = self.predictions(model)
        
        print('\nValidation Metrics:')
        print(f"Balanced accuracy for {model_name}: {balanced_accuracy_score(self.y_valid, y_pred_valid)}")
        print(f"Accuracy for {model_name}: ", accuracy_score(self.y_valid, y_pred_valid))
        print(f"Precision for {model_name}: ", precision_score(self.y_valid, y_pred_valid))
        print(f"Recall for {model_name}: ", recall_score(self.y_valid, y_pred_valid))
        print(f"F1-Score for {model_name}: ", f1_score(self.y_valid, y_pred_valid))
    
        print('\nTest Metrics:')
        print(f"Balanced accuracy for {model_name}: {balanced_accuracy_score(self.y_test, y_pred_test)}")
        print(f"Accuracy for {model_name}: {accuracy_score(self.y_test, y_pred_test)}")
        print(f"Precision for {model_name}: {precision_score(self.y_test, y_pred_test)}")
        print(f"Recall for {model_name}: {recall_score(self.y_test, y_pred_test)}")
        print(f"F1-Score for {model_name}: {f1_score(self.y_test, y_pred_test)}")


    def roc_auc_metric(self, model, model_name: str):
        """
        Plot the ROC curve and compute the AUC score for the test set.

        Parameters
        ----------
        model : object
            The machine learning model to evaluate.
        model_name : str
            The name of the model for display purposes.
        """
        _, y_pred_test = self.predictions(model)
        roc_auc = roc_auc_score(self.y_test, y_pred_test)
        
        y_probs = self.feat_probabilities(model)[:, 1]  # Get predicted probabilities for class 1
        fpr, tpr, thresholds = roc_curve(self.y_test, y_probs)
        roc_auc = roc_auc_score(self.y_test, y_probs)
        
        plt.plot(fpr, tpr, label='ROC curve (AUC = {:.3f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], linestyle='--') 
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc='lower right')
        plt.show()


    def confusion_matrix(self, model, model_name: str):
        """
        Plot the confusion matrix for the test set.

        Parameters
        ----------
        model : object
            The machine learning model to evaluate.
        model_name : str
            The name of the model for display purposes.
        """
        _, y_pred_test = self.predictions(model)
        
        cm = confusion_matrix(self.y_test, y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'{model_name} Confusion Matrix')
        plt.show()


    def probs_to_percentiles(self, y_probs: list):
        """
        Map predicted probabilities to percentiles and compute accuracy for each percentile group.

        Parameters
        ----------
        y_probs : list
            List of predicted probabilities.

        Returns
        -------
        tuple
            df_classes_acc : pd.DataFrame
                DataFrame with the number of samples in each percentile.
            accuracies : dict
                Dictionary with accuracy values for each percentile group.
        """
        x = np.arange(0.05, 1.05, 0.1)
        classes_acc = {round(i, 2): [] for i in x}
        mapping = {v: round(x[i], 2) for i, v in enumerate(range(0, 10))}
        
        for i, prob in enumerate(y_probs):
            percentile = min(int(Decimal(str(prob)) // Decimal('0.1')), 9)
            classes_acc[mapping[percentile]].append(self.y_test.values[i])

        rows = [[key, len(class_probs)] for key, class_probs in classes_acc.items()]
        df_classes_acc = pd.DataFrame(rows, columns=['Prob Class', 'Size'])
        
        accuracies = {i: sum(prob_class) / len(prob_class) for i, prob_class in classes_acc.items() if len(prob_class) > self.min_samples}
        
        return df_classes_acc, accuracies
    

    def plot_accuracy_by_percentile(self, model):
        """
        Plot the accuracy by percentile and predicted probabilities for the test set.

        Parameters
        ----------
        model : object
            The machine learning model to evaluate.

        Returns
        -------
        pd.DataFrame
            DataFrame with percentile classes and their corresponding sample sizes.
        """
        y_probs = self.feat_probabilities(model)[:, 1]
        df_classes_acc, accuracies = self.probs_to_percentiles(y_probs=y_probs)

        plt.scatter(list(accuracies.keys()), list(accuracies.values()), color='red')
        plt.plot(np.linspace(0, 1, len(y_probs)), sorted(y_probs), color='blue')
    
        plt.gca().invert_xaxis()
        plt.xticks(np.arange(0.0, 1.1, 0.1))
        plt.ylabel('Probabilities')
        plt.show()
        
        return df_classes_acc        