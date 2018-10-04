from IPython.display import display, HTML, Image

from sklearn.base import BaseEstimator, ClassifierMixin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import random
import itertools
import collections

from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
import inspect
import sys
import copy

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

# Add more packages as required

# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes
class SuperLearnerClassifier(BaseEstimator, ClassifierMixin):
    
    """An ensemble classifier that uses heterogeneous models at the base layer and a aggregatnio model at the aggregation layer. A k-fold cross validation is used to gnerate training data for the stack layer model.

    Parameters
    ----------
    criteria : indicates whether the stacked layer classifier should be trained 
               on label outputs from the base classifiers 
               or probability outputs from the base classifiers.
    
    stacked_model : type of model to use at the stack layer
    
    base_models : list of base estimators to use
                'CART' -> Decision Tree
                'RF' -> Random Forest
                'NB' -> Naive Bayes 
                'KNN'-> K Nearest neighbour
                'LR' -> Logisitc regression
                'MLP' -> Multi-layer perceptron
    
    add_original_input : If original data is to be added to the input at the stack layer 
        
    Attributes
    ----------
    
    chosen_base_models : list of base estimator objects as chosen
    
    final_model : the final stack layer model
    
    criteria : the type of output used to train the stacked layer
    
    correlation_hash : contains predicted values of each base estimator to calculate diversity among them
    
    Notes
    -----
    

    See also
    --------
    
    ----------
    .. [1]  van der Laan, M., Polley, E. & Hubbard, A. (2007). 
            Super Learner. Statistical Applications in Genetics 
            and Molecular Biology, 6(1) 
            doi:10.2202/1544-6115.1309
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> clf = SuperLearnerClassifier()
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)

    """
    # Constructor for the classifier object
    global LIST_OF_MODELS
    # This global variable contains default set of base estimators if nothing is passed as parameter
    LIST_OF_MODELS = ('CART', 'RF', 'NB', 'KNN', 'LR', 'MLP')
    
    # SuperLearner constructor takes these arguments as parameters with default values if nothing is specified
    def __init__(self, criteria = 'label', stacked_model= 'CART', base_models = LIST_OF_MODELS, add_original_input = False):
        """Setup a SuperLearner classifier .
        Parameters
        ----------
        criteria : string like, among 'label' or 'probability'
    
        stacked_model : string like, indicates type of model - according to global values from LIST_OF_MODELS
    
        base_models : array-like, list of strings indicating type of model - according to global values from LIST_OF_MODELS
    
        add_original_input : Boolean
        
        Returns
        -------
        None
        
        """          
        # Dictionary of sensible set of hyper-parameters created from previously done grid search on base estimators
        best_hyperparams_found = {'CART': DecisionTreeClassifier(criterion= 'entropy', max_depth= 6, min_samples_split= 25),\
                        'RF': RandomForestClassifier(max_features= 4, min_samples_split= 25, n_estimators= 400),\
                        'LR': LogisticRegression(C= 0.2, max_iter= 1000, multi_class= 'ovr', solver= 'liblinear'),\
                        'NB': GaussianNB(),\
                        'KNN': KNeighborsClassifier(n_neighbors= 6),\
                        'MLP': MLPClassifier(alpha= 1e-05, hidden_layer_sizes= (400, 200), solver='lbfgs', activation='logistic')}
            
        # Use this set of sensible hyper-parameters
        __stacked_model_dict__ = {'CART': best_hyperparams_found['CART'],\
                                  'LR': best_hyperparams_found['LR'],\
                                  'RF' : best_hyperparams_found['RF'], \
                                  'NB': best_hyperparams_found['NB'], \
                                  'KNN': best_hyperparams_found['KNN'], \
                                  'MLP': best_hyperparams_found['MLP']}
        
        
#         __stacked_model_dict__ = {'CART': DecisionTreeClassifier(), 'LR': LogisticRegression(), 'RF' : RandomForestClassifier(), \
#                                   'NB': GaussianNB(), 'KNN': KNeighborsClassifier(), 'MLP': MLPClassifier(solver='lbfgs')}
        
        __base_model_dict__ = copy.deepcopy(__stacked_model_dict__ )
        
        self.base_models = base_models  #This is used in cross_val_score() which clones the object
        self.stacked_model = stacked_model #This is used in cross_val_score() which clones the object

        # Exception handling to ensure correct base estimators and stacked layer model is entered
        try:
            if len(stacked_model) < 0:
                raise ValueError
            final_model = __stacked_model_dict__[stacked_model]
        except ValueError:
            print("Please enter minimum 1 stacked model estimator")
            sys.exit(0)
        except KeyError:
            print("Please enter a valid stack model among {}".format(list(__stacked_model_dict__.keys())))
            sys.exit(0)
        
        # Final stack layer model
        self.final_model = final_model
       
        try:
            if len(base_models) < 3  or len(base_models) > 10:
                raise ValueError
            chosen_base_models = [__base_model_dict__[i] for i in self.base_models]
            self.chosen_base_models = chosen_base_models # Each base estimator
        except ValueError:
            print("Please enter minimum 3 base estimators and max 10 base estimators")
            sys.exit(0) 
        except KeyError:
            print("Please enter a valid base estimators among {}".format(list(__base_model_dict__.keys())))
            sys.exit(0) 

        # Criteria - label or probability
        self.criteria = criteria
        self.add_original_input = add_original_input

        # To check diversity among base estimators
        self.correlation_hash = {}
        
    # The fit function to train a classifier
    def fit(self, X, y):
        
        """Build a SuperLearner classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. 
        y : array-like, shape = [n_samples] 
            The target values (class labels) as integers or strings.
        Returns
        -------
        self : object
        """     
        
        # As described V-fold in the description of SuperLearner
        k_fold = 10

        #Split the original dataset into K folds
        split_X = np.array_split(X, k_fold)
        split_y = np.array_split(y, k_fold)
        
        stacked_list = list()
        
        for i in range(k_fold):
            # Segregating training and test set using splitted array for both X and y
            X_test = split_X[i].tolist()
            
            X_train = [ split_X[j].tolist() for j in range(k_fold) if i!= j]
            # Flattening the X_train to 1-D list
            X_train = list(itertools.chain.from_iterable(X_train))

            y_test = split_y[i].tolist()
            y_train = [ split_y[j].tolist() for j in range(k_fold) if i!= j]
            y_train = list(itertools.chain.from_iterable(y_train))

            y_pred_list = list()
            
            #Iterate for each base estimator
            for model in self.chosen_base_models:
                # Fit each base estimator
                model.fit(X_train , y_train)
                # If criteria is label use predict else predict_proba
                y_pred = model.predict(X_test) if self.criteria == "label" else model.predict_proba(X_test)
                # Append it to as list that can be used as input to the stacked layer
                y_pred_list.append(y_pred)
                
                
                # -----TASK 9 -----
                self.y = y
                if self.criteria == "label":
                    # To evaluate correlation between base estimators only in case of label outputs
                    if str(model.__class__.__name__) not in self.correlation_hash.keys():
                        self.correlation_hash[str(model.__class__.__name__)] = y_pred.tolist()
                    else:
                        self.correlation_hash[str(model.__class__.__name__)].extend(y_pred.tolist())

            
            # Using column stack on the appended list so that the input training set for stacked layer is of proper shape
            stacked_list.append(np.column_stack(y_pred_list).tolist())
        
        #Converting stacked_list to 1-D flattened list
        stacked_list = list(itertools.chain.from_iterable(stacked_list))

        if self.add_original_input:
            # Add the original input to the final list of training data for stacked layer
            stacked_list = np.column_stack((X,stacked_list)) 
        
        self.final_model.fit(stacked_list, y)
        
        for model in self.chosen_base_models:
            # Fitting each model with entire dataset
            model.fit(X,y)
            
        # Return the classifier
        return self

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        """Predict class labels of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples. 
        Returns
        -------
        p : array of shape = [n_samples, ].
            The predicted class labels of the input samples. 
        """
        stacked_list = list()
        y_pred_list = list()
        for model in self.chosen_base_models:
            # Predict for each base estimator if criteria is label and append it to a list
            y_pred = model.predict(X) if self.criteria == "label" else model.predict_proba(X)
            y_pred_list.append(y_pred)

        # Then append the entire list as a column stack to another list to be used at the stacked layer
        stacked_list.append(np.column_stack(y_pred_list).tolist())
        stacked_list = list(itertools.chain.from_iterable(stacked_list))

        if self.add_original_input:
            # Add the original input to the final list of training data for stacked layer
            stacked_list = np.column_stack((X,stacked_list))
        
        # Finally predict at the stacked layer
        y_pred = self.final_model.predict(stacked_list)
            
        return y_pred
    
    # The predict function to make a set of predictions for a set of query instances
    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples. 
        Returns
        -------
        p : array of shape = [n_samples, n_labels].
            The predicted class label probabilities of the input samples. 
        """
        stacked_list = list()
        y_pred_list = list()
        for model in self.chosen_base_models:
            # Predict_proba for each base estimator if criteria is probability and append it to a list
            y_pred = model.predict_proba(X) if self.criteria != "label" else model.predict(X)
            y_pred_list.append(y_pred)
        
        # Then append the entire list as a column stack to another list to be used at the stacked layer
        stacked_list.append(np.column_stack(y_pred_list).tolist())
        stacked_list = list(itertools.chain.from_iterable(stacked_list))
        
        if self.add_original_input:
            # Add the original input to the final list of training data for stacked layer
            stacked_list = np.column_stack((X,stacked_list))
            
        # Finally predict_proba at the stacked layer
        y_pred = self.final_model.predict_proba(stacked_list)
            
        return y_pred
        
    
    # ------------------------------ TASK 9 ---------------------------
    """
    This function calculates Accuracy among the base estimators using accuracy metric
    and the diversity among them using Disagreement measure.
    
    It also displays the Pearson-R correlation matrix among base estimators
    and plots the pandas dataframe using a scatter matrix plot
    
    
    Calculating disagreement measure among base estimators of SuperLearner using the formula for disagreement measure
    as stated in the Paper [2].
    
    See also: [2] https://link.springer.com/content/pdf/10.1023%2FA%3A1022859003006.pdf
    """
    
    def check_accuracy(self):
        if self.criteria != "label":
            raise ValueError("Please enter only label type output in the SuperLearner")
            return
        
        print("\n******************** Accuracy (Predictive power) of each base estimator ***********")
        acc_hash = {}
        # To evaluate accuracy between base estimators
        for model_i in self.chosen_base_models:
            y_pred = self.correlation_hash[model_i.__class__.__name__]
            accuracy = metrics.accuracy_score(self.y, y_pred)
            acc_hash[model_i.__class__.__name__] = accuracy
        display(pd.Series(acc_hash))

        
    def check_diversity(self):
        """
            Calculating Disagreement measure and Pearson-R correlation measure.
            N11 - number of Test Examples correctly classified by both classifiers
            N00 - number of Test Examples incorrectly classified by both classifiers
            N10 - number of Test Examples correctly classified by 1st classifier but incorrectly by 2nd classifier
            N01 - number of Test Examples incorrectly classified by 1st classifier but correctly by 2nd classifier
            
            Disagreement measure = (N01 + N10) / (N11 + N00 + N10 + N01)
            
            Pearson R measure is calculated by using corr() function in pandas dataframe.
            
        """
        # Dataframe used for plotting y_pred values and Pearson correlation
        plot_df = pd.DataFrame()
        diversity_hash = {}
        __dict_classfier_code__ = {'DecisionTreeClassifier': 'CART', 'RandomForestClassifier': 'RF',\
                                           'GaussianNB': 'NB', 'KNeighborsClassifier': 'KNN',\
                                           'LogisticRegression':'LR', 'MLPClassifier': 'MLP'}

        try:
            # Take every combination of base estimators and calculate disagreement measure among them
            for model_i in self.chosen_base_models:
                div_list = list()
                for model_j in self.chosen_base_models:
                    y_pred1 = self.correlation_hash[model_i.__class__.__name__]
                    y_pred2 = self.correlation_hash[model_j.__class__.__name__]
                    # Store y_preds in a pandas dataframe for visualisation
                    plot_df[__dict_classfier_code__[model_i.__class__.__name__]] = y_pred1
                    plot_df[__dict_classfier_code__[model_j.__class__.__name__]] = y_pred2

                    N11, N00, N10, N01 = 0,0,0,0

                    # Calculate the disagreement rate among the base estimators
                    for i in range(len(Y_train)):
                        if Y_train[i] == y_pred1[i] and Y_train[i] == y_pred2[i]:
                            N11 += 1
                        elif Y_train[i] != y_pred1[i] and Y_train[i] != y_pred2[i]:
                            N00 += 1
                        elif Y_train[i] == y_pred1[i] and Y_train[i] != y_pred2[i]:
                            N10 += 1
                        elif Y_train[i] != y_pred1[i] and Y_train[i] == y_pred2[i]:
                            N01 += 1

                    disagreement = (N01 + N10) / (N11 + N00 + N10 + N01)
#                     print("Disagreement between %s and %s is %.2f" % (__dict_classfier_code__[model_i.__class__.__name__],__dict_classfier_code__[model_j.__class__.__name__], disagreement))
                    div_list.append(disagreement)
                
                diversity_hash[__dict_classfier_code__[model_i.__class__.__name__]] = div_list

            # Display disagreement measure among base estimators
            print("\n******************** Disagreement (Heterogenity) measure  ***********")
            display(pd.DataFrame(collections.OrderedDict(diversity_hash), index=diversity_hash.keys()))
            # Display the Pearson R correlation among base estimators
            print("\n******************** Pearson-R correlation scores ***********")
            display(plot_df.corr())
            
            # Plotting the Pandas dataframe using a scatter matrix to visualise each base estimator's target label with others
            print("\n******************** Pandas scatter plot of y_pred values for each base estimator ***********")
            pd.plotting.scatter_matrix(df, figsize=(8, 8))
            plt.show()
            
    
        except KeyError:
            print("Please fit the model OR use label outputs in the model before checking for diversity")