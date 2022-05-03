"""
.. module:: results

results
******

:Description: results

    After the models are fitted, we take a deeper look into how some of the 
    models have decided to classify. We take a closer look checking the 
    feature importance of the "Logistic Regression" and "Random Forests".

:Authors:
    benjami parellada

:Version: 

:Date:  
"""

__author__ = 'benjami parellada'
import os.path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import modeling
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance


sns.set(rc={'figure.figsize':(10, 7.5)})
sns.set_color_codes("deep")
sns.set_style("whitegrid")
pal = 'rocket'

def plotNumerical(df):
    """
    Function that plots the Ejection Fraction and Serum Creatinine against the predicted class.
    Saves the plots in './figures/RBFboxplots'
    """
    df['True Value'].replace([1, 0], ['Yes', 'No'], inplace = True)
    fig, axes = plt.subplots(1,3, gridspec_kw = {'width_ratios': [1, 1,3]}, figsize = (12,5))
    fig.set_figheight(3.5)
    fig.set_figwidth(7.195)
    sns.boxplot(x = 'Prediction', y = 'Ejection Fraction', data = df, ax = axes[1], palette = pal)
    sns.boxplot(x = 'Prediction', y = 'Serum Creatinine', data = df, ax = axes[0], palette = pal)
    sns.scatterplot(x = df['Serum Creatinine'], y = df['Ejection Fraction'] + np.random.normal(scale = 0.1, size = len(df)), 
                    hue = df['Prediction'], style = df['True Value'], palette = pal, ax = axes[2])
    plt.legend(fontsize = '8', title_fontsize='10')
    plt.tight_layout()
    plt.savefig('./figures/RFboxplots')
    plt.clf()

def rfPlots(reg, x_train, y_train, x_test, y_test, column_names):
    """
    Function that plots the the confusion matrix, calculates the permutation 
    importance on the test data, plots the ejection fraction vs serum with grouped with
    the predicted class for the Random Forest.

    Also prints some of the more relavant values that we have shown in the report.
    """
    # Conf Matrix
    met = reg['metrics']
    norm = met['TN'] + met['FP'] + met['FN'] + met['TP']
    labels = [ [met['TN'], met['FP']], [met['FN'], met['TP']]]
    cm = [[met['TN'], met['FP']], [met['FN'], met['TP']]]/norm
    plt.figure(figsize=(7.195, 7.195), dpi = 96)
    sns.heatmap(cm, annot = labels, fmt = 'g', yticklabels=['No', 'Yes'], xticklabels=['No', 'Yes'], vmin=0, vmax=0.5, cbar=False)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('figures/RFconf')
    plt.clf()

    # Permutation Importance
    r = permutation_importance(reg['regressor'], x_test['None'], y_test, n_repeats=30, random_state=0)
    importances = pd.DataFrame({'name': column_names, 'mean': r.importances_mean, 'sd': r.importances_std})
    importances.drop(importances[importances['mean'] - 2.1 * importances['sd'] < 0].index, inplace = True)
    importances.drop(importances[importances['mean'] == 0].index, inplace = True)
    importances.sort_values(by = ['mean'], inplace = True, ascending = False)
    plt.figure(figsize=(7.195, 1.5), dpi = 96)
    sns.barplot(x = 'mean', y = 'name', data = importances, ci = None)
    plt.errorbar(x = 'mean', y = range(len(importances)), xerr = 'sd', data = importances, fmt = 'none', c = 'black', linewidth=1)
    plt.xlabel('Mean accuracy decrease')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('figures/RFfeatures', dpi = 96)

    # plot prediction vs serum and ejection
    x_aux = x_test['None']
    df = pd.DataFrame({'Prediction': reg['predictions'], 'Serum Creatinine': x_aux['serum_creatinine'], 'Ejection Fraction': x_aux['ejection_fraction'], 'True Value': y_test})
    df.Prediction.replace([1, 0], ['Yes', 'No'], inplace = True)
    plotNumerical(df)

    # Lets check the miss classified
    x_aux = x_aux.assign(True_val = y_test) 
    x_aux = x_aux.assign(Prediction =  reg['predictions'])

    print('Some examples of the RF')
    print(x_aux.loc[[4, 35, 229, 275] ][['True_val', 'Prediction', 'serum_creatinine', 'ejection_fraction']])

    x_aux = x_train['None']
    x_aux = x_aux.assign(Prediction = reg['regressor'].predict(x_aux))
    x_aux = x_aux.assign(True_val = y_train)
    print(x_aux.loc[[129] ][['True_val', 'Prediction', 'serum_creatinine', 'ejection_fraction']])

def lrPlots(reg, x_train, y_train, x_test, y_test, column_names):
    """
    Function that plots the the weight of the coefficients and prints
    some of the more relavant values that we have shown in the report.
    """
    df = pd.DataFrame({'Features': column_names, 'Coef': reg['regressor'].coef_[0]})
    df.sort_values(by = ['Coef'], inplace = True, ascending = False)

    plt.figure(figsize=(7.195, 2), dpi = 96)
    sns.barplot(x = 'Coef', y = 'Features', data = df, ci = None)
    plt.yticks(fontsize = 8)
    plt.xticks(fontsize = 8)
    plt.ylabel('')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig('figures/LRfeatures', dpi = 96)

    x_aux = x_test['StandardScaler']
    x_aux = x_aux.assign(True_val = y_test) 
    x_aux = x_aux.assign(Prediction =  reg['predictions'])

    print('Some examples of the LogReg')
    x_aux = x_aux.assign(Prob = reg['regressor'].predict_proba(x_test['StandardScaler'])[:,1])
    #x_aux.sort_values(by = ['serum_creatinine', 'creatinine_phosphokinase'], inplace = True, ascending = False)
    #print(x_aux[['True_val', 'Prediction', 'serum_creatinine', 'creatinine_phosphokinase', 'serum_sodium', 'ejection_fraction']])
    print(x_aux.loc[[4, 75, 88, 203] ][['True_val', 'Prediction', 'Prob', 'serum_creatinine', 'creatinine_phosphokinase', 'serum_sodium', 'ejection_fraction']])

def main():
    """
    Main function of the file, it executes the three previously described functions
    """
    x_train, y_train, x_test, y_test = modeling.read_data()
    column_names = ([i.replace('_', ' ').title() for i in x_train['StandardScaler'].columns])

    with open("./results/fitted_models.pickle","rb") as f:
        res = pickle.load(f)

    # print hyperparameters
    for model in res:
        m = res[model]
        print(model, m['best_params'])

    lrPlots(res['LogisticRegression'], x_train, y_train, x_test, y_test, column_names)
    rfPlots(res['RandomForestClassifier'], x_train, y_train, x_test, y_test, column_names)

if __name__ == '__main__':
    main()   