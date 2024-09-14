import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from modules.evaluate import Evaluate
# Add LightGBM


class Modeling:
    """
    A class to implement and evaluate various machine learning models, such as Logistic Regression, Decision Trees, 
    Random Forests, Gradient Boosting, SVM, and MLP Classifier.

    Attributes
    ----------
    X_train : pd.DataFrame
        Features of the training set.
    y_train : pd.DataFrame
        Target of the training set.
    X_valid : pd.DataFrame
        Features of the validation set.
    y_valid : pd.DataFrame
        Target of the validation set.
    X_test : pd.DataFrame
        Features of the test set.
    y_test : pd.DataFrame
        Target of the test set.
    eval : Evaluate
        An instance of the Evaluate class to compute evaluation metrics and plots.

    Methods
    -------
    logistic_regression():
        Train and evaluate a Logistic Regression model.

    decision_tree_classifier():
        Train and evaluate a Decision Tree classifier.

    random_forest_classifier():
        Train and evaluate a Random Forest classifier.

    gradient_boosting_classifier(iterations_wout_change: float, variation_tolerance: float):
        Train and evaluate a Gradient Boosting classifier.

    svm():
        Train and evaluate a Support Vector Machine model.

    mlp_classifier(hidden_layer_sizes: tuple, solver: str, learning_rate: str):
        Train and evaluate a Multi-layer Perceptron model.
    """
    

    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, min_samples: int) -> None:
        """
        Initialize the Modeling class with training, validation, and test sets.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            The features of the training set.
        y_train : pd.DataFrame
            The target of the training set.
        X_valid : pd.DataFrame
            The features of the validation set.
        y_valid : pd.DataFrame
            The target of the validation set.
        X_test : pd.DataFrame
            The features of the test set.
        y_test : pd.DataFrame
            The target of the test set.
        min_samples : int
            Minimum samples required for certain methods in the Evaluate class.
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_valid = X_valid.copy()
        self.y_valid = y_valid.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        
        self.eval = Evaluate(X_valid=self.X_valid.copy(), y_valid=self.y_valid.copy(), X_test=self.X_test.copy(), y_test=self.y_test.copy(), min_samples=min_samples)

        
    def logistic_regression(self):
        """
        Train and evaluate a Logistic Regression model.
        
        Returns
        -------
        pd.DataFrame
            Dataframe containing class-wise accuracies for the Logistic Regression model based on percentiles.
        """
        logreg = LogisticRegression(class_weight='balanced', random_state=42)
        logreg.fit(self.X_train.to_numpy(), self.y_train.to_numpy().ravel())
        
        self.eval.evaluation_metrics(model=logreg, model_name='Logistic Regression')
        self.eval.roc_auc_metric(model=logreg, model_name='Logistic Regression')
        self.eval.confusion_matrix(model=logreg, model_name='Logistic Regression')
        df_classes_acc = self.eval.plot_accuracy_by_percentile(model=logreg)
        
        return df_classes_acc
    

    def decision_tree_classifier(self):
        """
        Train and evaluate a Decision Tree classifier.
        
        Returns
        -------
        pd.DataFrame
            Dataframe containing class-wise accuracies for the Decision Tree model based on percentiles.
        """
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(self.X_train.to_numpy(), self.y_train.to_numpy().ravel())
        
        self.eval.evaluation_metrics(model=dt, model_name='Decision Trees')
        self.eval.roc_auc_metric(model=dt, model_name='Decision Trees')
        self.eval.confusion_matrix(model=dt, model_name='Decision Trees')
        df_classes_acc = self.eval.plot_accuracy_by_percentile(model=dt)
        
        return df_classes_acc
    
        
    def random_forest_classifier(self):
        """
        Train and evaluate a Random Forest classifier.
        
        Returns
        -------
        pd.DataFrame
            Dataframe containing class-wise accuracies for the Random Forest model based on percentiles.
        """
        rf = RandomForestClassifier(random_state=42)
        rf.fit(self.X_train.to_numpy(), self.y_train.to_numpy().ravel())
        
        self.eval.evaluation_metrics(model=rf, model_name='Random Forest')
        self.eval.roc_auc_metric(model=rf, model_name='Random Forest')
        self.eval.confusion_matrix(model=rf, model_name='Random Forest')
        df_classes_acc = self.eval.plot_accuracy_by_percentile(model=rf)
        
        return df_classes_acc
    
        
    def gradient_boosting_classifier(self, iterations_wout_change: float, variation_tolerance: float):
        """
        Train and evaluate a Gradient Boosting classifier.
        
        Parameters
        ----------
        iterations_wout_change : float
            Number of iterations without change to stop the training early.
        variation_tolerance : float
            Tolerance for stopping the training early.
        
        Returns
        -------
        pd.DataFrame
            Dataframe containing class-wise accuracies for the Gradient Boosting model based on percentiles.
        """
        gb = GradientBoostingClassifier(random_state=42, n_iter_no_change=iterations_wout_change, tol=variation_tolerance)
        gb.fit(self.X_train.to_numpy(), self.y_train.to_numpy().ravel())
        
        self.eval.evaluation_metrics(model=gb, model_name='Gradient Boosting')
        self.eval.roc_auc_metric(model=gb, model_name='Gradient Boosting')
        self.eval.confusion_matrix(model=gb, model_name='Gradient Boosting')
        df_classes_acc = self.eval.plot_accuracy_by_percentile(model=gb)
        
        return df_classes_acc
    
        
    def svm(self):
        """
        Train and evaluate a Support Vector Machine (SVM) model.
        
        Returns
        -------
        pd.DataFrame
            Dataframe containing class-wise accuracies for the SVM model based on percentiles.
        """
        svm = SVC(kernel='rbf', random_state=42, probability=True)
        svm.fit(self.X_train.to_numpy(), self.y_train.to_numpy().ravel())
        
        self.eval.evaluation_metrics(model=svm, model_name='Support Vector Machines')
        self.eval.roc_auc_metric(model=svm, model_name='Support Vector Machines')
        self.eval.confusion_matrix(model=svm, model_name='Support Vector Machines')
        df_classes_acc = self.eval.plot_accuracy_by_percentile(model=svm)
        
        return df_classes_acc
    
        
    def mlp_classifier(self, hidden_layer_sizes: tuple, solver: str, learning_rate: str):
        """
        Train and evaluate a Multi-layer Perceptron (MLP) classifier.
        
        Parameters
        ----------
        hidden_layer_sizes : tuple
            Number of neurons in each hidden layer of the MLP.
        solver : str
            The solver to use for weight optimization.
        learning_rate : str
            The learning rate schedule for weight updates.
        
        Returns
        -------
        pd.DataFrame
            Dataframe containing class-wise accuracies for the MLP model based on percentiles.
        """
        if 'sgd' in solver:
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver=solver, learning_rate=learning_rate, early_stopping=True, random_state=42)
        else:
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver=solver, early_stopping=True, random_state=42)
            
        mlp.fit(self.X_train.to_numpy(), self.y_train.to_numpy().ravel())
        
        self.eval.evaluation_metrics(model=mlp, model_name='Multi-layer Perceptron')
        self.eval.roc_auc_metric(model=mlp, model_name='Multi-layer Perceptron')
        self.eval.confusion_matrix(model=mlp, model_name='Multi-layer Perceptron')
        df_classes_acc = self.eval.plot_accuracy_by_percentile(model=mlp)
        
        return df_classes_acc