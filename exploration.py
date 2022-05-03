"""
.. module:: exploration

exploration
******

:Description: exploration

    In this file, the part of exploration of the project is realized.
    Will download from UCI and create a folder called 'data' and save the data at heart.csv if not available.
    Will create all the different plots comparing each feature with the target feature.
    Clustering algorithms are tested.
    Reduction Algorithms are tested and visualized.
    The splitting and scaling of data is realized here.

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
import gower
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sns.set(rc={'figure.figsize':(10, 7.5)})
sns.set_color_codes("deep")
sns.set_style("whitegrid")
pal = 'rocket'

seed = 1984

def read_data():
    """
    Creates a folder called data if it doesn't exist, and downloads the data if it doesn't exist.
    Returns the data in pandas Dataframe format
    """
    if not os.path.exists('./data/'): # create data fold if it doesnt exist
        os.makedirs('./data/')
    if not os.path.exists('./data/heart.csv'): # downloads data if it doesnt exist in ./data/ 
        import requests
        data_request = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv')
        open('./data/heart.csv', 'wb').write(data_request.content)
    return pd.read_csv('./data/heart.csv')

def plot_numerical(data, col):
    """
    Plots each numerical data as a boxplot and histogram, grouping by the target feature.
    Saves the plots in './figures/'
    """
    fig, axes = plt.subplots(1,2, gridspec_kw = {'width_ratios': [1, 4]}, figsize = (9,5))
    sns.boxplot(x = 'Death Event', y = col, data = data, ax = axes[0], palette = pal)
    sns.histplot(hue = 'Death Event', x = col, data = data, ax = axes[1], palette = pal, kde = True)
    plt.tight_layout()
    plt.savefig('./figures/' + col)
    plt.clf()

def plot_categoric(data, col):
    """
    Plots each categorical data as a countplot, grouping by the target feature.
    Saves the plots in './figures/'
    """
    sns.countplot(hue = 'Death Event', x = col, data = data, palette = pal)
    plt.tight_layout()
    plt.savefig('./figures/' + col)
    plt.clf()

def biplot(score, comp, coeff, var, y, labels):
    """
    Plots a biplot given the scores and components of a PCA. To plot the coefficients of the
    components we need to scale the data to [-1, 1].
    comp is a list of which components we want to plot
    Saves the plots in './figures/PC{x}_PC{y}.png', where x and y are comp[0] and comp[1].
    """
    xs = score[:,comp[0]]
    ys = score[:,comp[1]]
    n = coeff.shape[0]
    scalex = 2.0/(xs.max() - xs.min())
    scaley = 2.0/(ys.max() - ys.min())
    sns.scatterplot(x = xs * scalex, y = ys * scaley, hue = y, palette = pal)
    max_x = np.max(coeff[0,comp[0]])
    max_y = np.max(coeff[0,comp[0]])
    for i in range(n):
        if max_x < np.max(coeff[i,comp[0]]):
            max_x = np.max(coeff[i,comp[0]])
        if max_y < np.max(coeff[i,comp[0]]):
            max_y = np.max(coeff[i,comp[0]])
    for i in range(n):
        pc_x = coeff[i,comp[0]]/max_x
        pc_y = coeff[i,comp[1]]/max_y
        plt.arrow(0, 0, pc_x, pc_y, color = 'black', alpha = 0.9, width = 0.003)
        plt.text(pc_x, pc_y, labels[i], color = 'b', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel(f"PC{comp[0]}: explains {100*var[comp[0]]:.2f}%")
    plt.ylabel(f"PC{comp[1]}: explains {100*var[comp[1]]:.2f}%")
    plt.tight_layout()
    plt.savefig(f"./figures/PC{comp[0]}_PC{comp[1]}")
    plt.clf()

def biplot_lda(score, coeff, y, labels, lda, pred):
    """
    Plots a biplot given the scores and components of a LDA. To plot the coefficients of the
    components we need to scale the data to [-1, 1].
    Since we only have a binary target feature, the data is projected to 1D. We apply jitterting
    and plot the LDA coefficients on the side.
    Saves the plots in './figures/lda.png'
    """
    xs = np.random.uniform(-0.1, 0.1, len(y))
    order = np.argsort(score[:,0]) # order, and apply it to y
    ys = score[order[::-1],0]
    y = y[order[::-1]]
    pred = pred[order[::-1]]
    n = coeff.shape[0]
    scaley = 3.0/(ys.max() - ys.min())
    sns.scatterplot(x = xs, y = ys*scaley, style = pred, hue = y, palette = pal); scaley /= 22
    max_x = np.max(coeff[0,0])
    for i in range(n):
        if max_x < np.max(coeff[i,0]):
            max_x = np.max(coeff[i,0])
    for i in range(n):
        pc_x = coeff[i,0]/max_x
        pos_y = -0.5
        
        if abs(pc_x) > 0.15: # just plot the more important
            plt.text(pos_y, pc_x, labels[i], color = 'black', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.ylabel(f"LDA")
    plt.xticks([])
    plt.tight_layout()
    x_bound = np.array([-0.25, 0.25])
    y_bound = (-lda.coef_[0][0]/lda.coef_[0][1] * x_bound + lda.intercept_/lda.coef_[0][1])*scaley # scale it
    plt.plot(x_bound, y_bound)
    plt.savefig(f"./figures/lda")
    plt.clf()

def pca_explained(pca):
    """
    The amount of variance the PCA explains.
    Saves the plot in './figures/PCA_explained.png'
    """
    plt.plot(range(1,len(pca.explained_variance_ratio_ )+1),pca.explained_variance_ratio_ ,alpha=0.8,marker='.',label="Explained Variance");
    y_label = plt.ylabel('Explained variance')
    x_label = plt.xlabel('Principal Components')
    plt.plot(range(1,len(pca.explained_variance_ratio_ )+1),
         np.cumsum(pca.explained_variance_ratio_),
         c='red', marker='.',
         label="Cumulative Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figures/PCA_explained")
    plt.clf()

def pca_reduction(df, data):
    """
    Creates the PCA and applies it. We scale the data first with a Standard Scaler
    """
    x = data.drop('DEATH_EVENT', axis = 1)
    x = x.drop('time', axis = 1)
    y = df['Death Event']
    scale = StandardScaler()
    x = scale.fit_transform(x)
    pca = PCA()
    x_fit = pca.fit_transform(x)

    biplot(x_fit, [0, 1], np.transpose(pca.components_), pca.explained_variance_ratio_, y, df.columns)
    pca_explained(pca)

def reductions(df, data):
    """
    Various other dimensionality reduction algorithms are tested. The only relevant result is the LDA.
    """
    def scatter_reduction(name, y, tar):
        plt.tight_layout()
        sns.scatterplot(x = y[:, 0], y = y[:, 1], hue = tar, palette = pal)
        plt.savefig(f"./figures/{name}")
        plt.clf()
    
    x = data.drop('DEATH_EVENT', axis = 1)
    x = x.drop('time', axis = 1)
    tar = df['Death Event']

    lda = LinearDiscriminantAnalysis().fit(x, y = tar); y = lda.transform(x); biplot_lda(y, lda.scalings_, tar, df.columns, lda, lda.predict(x))
    y = TSNE(n_components = 2, init = 'pca').fit_transform(x); scatter_reduction('tsne', y, tar)
    y = LocallyLinearEmbedding(n_neighbors = 20, n_components = 2).fit_transform(x); scatter_reduction('lle', y, tar)
    y = MDS(n_components = 2, n_init = 15, metric = True).fit_transform(x); scatter_reduction('mds', y, tar)
    y = Isomap(n_components = 2, n_neighbors = 10).fit_transform(x); scatter_reduction('isomap', y, tar)

def clustering(df):
    """
    We apply Agglomerative Clustering with a Gower distance metric.
    """
    tar = df['Death Event']
    df = df.drop('Time', axis = 1)
    df = df.drop('Death Event', axis = 1)

    distance_matrix = np.sqrt(1 - gower.gower_matrix(df))
    clust = AgglomerativeClustering(n_clusters = 2, affinity = 'precomputed', linkage = 'complete').fit(distance_matrix)

    fig, axes = plt.subplots(1,2, figsize = (9,5))
    sns.scatterplot(x = df['Serum Creatinine'], y = df['Ejection Fraction'], style = tar, hue = clust.labels_, palette = pal, ax = axes[0])

    labels = pd.DataFrame(clust.labels_)
    labels.replace([1, 0], ['Yes', 'No'], inplace = True) # class 1 has less, so we assume it would be yes
    cm = confusion_matrix(tar, labels)
    sns.heatmap(cm, annot = True, fmt = 'g', yticklabels=['No', 'Yes'], ax = axes[1], cbar=False, vmin=0, vmax=0.5)
    plt.tight_layout()
    plt.savefig('figures/cluster')
    plt.clf()
    

def visualization(data):
    """
    Helper function that executes the description of the dataset, plots the numerical/categorical data
    with the functions above, and also calls the clustering and dimensionality reductions.
    """
    if not os.path.exists('./figures/'): # create data fold if it doesnt exist
        os.makedirs('./figures/')

    df = data.copy()
    column_names = ([i.replace('_', ' ').title() for i in df.columns])
    corr = data.corr()
    plt.subplots(figsize = (15,10))
    sns.heatmap(corr, annot = True, cbar=False, xticklabels = column_names, yticklabels = column_names)
    plt.tight_layout()
    plt.savefig('./figures/correlationPlot')

    df.age = df.age.astype('int')
    df.anaemia.replace([1, 0], ['Yes', 'No'], inplace = True)
    df.high_blood_pressure.replace([1, 0], ['Yes', 'No'], inplace = True)
    df.diabetes.replace([1, 0], ['Yes', 'No'], inplace = True)
    df.smoking.replace([1, 0], ['Yes', 'No'], inplace = True)
    df.DEATH_EVENT.replace([1, 0], ['Yes', 'No'], inplace = True)
    df.sex.replace([1, 0], ['Man', 'Woman'], inplace = True)

    # make columns prettier
    df.columns = column_names

    print('We first check the summary statistics of the numerical data.')
    print('We can see how there are no weird values, i.e. negative values "9999..." values or NA.')
    print(df.describe()) 
    print('We check summary statistics of the categorical data.')
    print('Again, no weird data is found')
    print(df.describe(include = ['object']))

    numerical = df.dtypes[df.dtypes != 'object'].index
    categoric = df.dtypes[df.dtypes == 'object'].index
    for col in numerical:
        plot_numerical(df, col)

    for col in categoric:
        if col != 'Death Event':
            plot_categoric(df, col)

    reductions(df, data)
    pca_reduction(df, data)
    clustering(df)

def resampling(data):
    """
    This function splits the data into train and test and does the only preprocessing we need
    Converts the age to integer
    """
    data.age = data.age.astype('int')
    x = data.drop(columns = ['DEATH_EVENT', 'time'])
    y = data.DEATH_EVENT

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = seed)
    return x_train, y_train, x_test, y_test

def scaling(x_tr, x_te):
    """
    Function that given x_train and x_test scales the data according to different scalers.
    We apply the scaler fitted on the train to the test data.
    It retruns two dictionaries that represent x_train and x_test with each scaler,
    such x_train['StandardScaler'] represents x_train scaled with the StandardScaler.
    """
    scalers = ['StandardScaler', 'MinMaxScaler']
    numeric = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

    x_train = dict()
    x_test  = dict()

    x_train['None'] = x_tr.copy()
    x_test['None'] = x_te.copy()
    for sc in scalers:
        scale = globals()[sc]()
        x_train[sc] = x_tr.copy()
        x_test[sc]  = x_te.copy()

        x_train[sc][numeric] = scale.fit_transform(x_tr[numeric])
        x_test[sc][numeric]  = scale.transform(x_te[numeric])

    return x_train, x_test

def main():
    """
    Main function of this script, it applies all the functions described above.
        - Downloads-Reads the data
        - Visualizes the data (plotting numerical data, dimensionality reduction, clustering...)
        - Splits into test-train
        - Applies scalers
        - Pickles it and saves it ino './data/preprocess.pickle'
    """
    data = read_data()
    visualization(data)
    x_train, y_train, x_test, y_test = resampling(data)
    x_train, x_test = scaling(x_train, x_test)

    with open("./data/preprocess.pickle","wb") as f:
        pickle.dump(x_train, f)
        pickle.dump(y_train, f)
        pickle.dump(x_test, f)
        pickle.dump(y_test, f)

if __name__ == '__main__':
    main()    
    

