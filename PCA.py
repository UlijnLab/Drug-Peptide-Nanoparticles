import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns
import pathlib

'''
Script to perform PCA on a csv where the last two columns are target labels

Version History
0.0.01 SM inception
0.0.02 NB updated plotting

'''


#-----------
#import CSV file as data dataframe
#-----------


csv_loc = str(pathlib.Path().absolute()) + "/all_drug_below2000_peptide_descriptors_clean_classified.csv"

data = pd.read_csv(csv_loc)
#print('Preview Data:')
#print(data)
#print(data.describe())

#-----------
#column headers to features (or variable names)
#-----------

features = data.columns.values
features2 = data.columns.values
#print('Column Headers:')
#print(features)


#-----------
#specify target column (if there is one)
#-----------

target = features[-1]
target2 = features[-2]

#-----------
#remove the target from list of features/variable names
#-----------

features = features[:-2]

#print('Features:')
#print(features)
#print('Target:')
#print(target)
#print(target2)

#-----------
#seperating and scaling data
#-----------

x = data.loc[:, features].values
y = data.loc[:, target].values

scaled_data = preprocessing.scale(x)
scaled_data = pd.DataFrame(scaled_data, columns=features)
#print(scaled_data)

#-----------
#export scaled data to CSV
#-----------

#scaled_data.to_csv(r'scaled.csv')

#-----------
#Perform PCA
#-----------

pca = PCA()
principalComponents = pca.fit_transform(scaled_data)
#print(principalComponents)
pca_data = pd.DataFrame(data = principalComponents, columns = ['PC' + str(x) for x in range(1, len(pca.components_)+1)])
#print(pca_data)

#-----------
#Construct final PCA data table for exporting/plotting
#-----------

finalcoords = pd.concat([pca_data, data[[target]]], axis = 1)
#print(finalcoords )

#-----------
#Extract additional data from the analysis
#-----------

singval = pd.DataFrame(pca.singular_values_)
explvar = pd.DataFrame(pca.explained_variance_)

#-----------
#Loadings
#Eigenvectors; how much (the weight) each original variable contributes to the corresponding principal component.
#https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
#-----------

loadings2 = pd.DataFrame(pca.components_.T, columns=['PC' + str(x) for x in range(1, len(pca.components_)+1)], index=features)
#print('Loadings:')
#print(loadings2)

#-----------
#Loading Matrix
#Here each entry of the matrix contains the correlation between the original variable and the principal component.
#http://strata.uga.edu/8370/lecturenotes/principalComponents.html
#These values are called the loadings, and they describe how much each variable contributes to a particular principal component.
#Large loadings (positive or negative) indicate that a particular variable has a strong relationship to a particular principal component.
#The sign of a loading indicates whether a variable and a principal component are positively or negatively correlated.
#-----------

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

loading_matrix = pd.DataFrame(loadings, columns=['PC' + str(x) for x in range(1, len(pca.components_)+1)], index=features)
#print('Loading Matrix:')
#print(loading_matrix)

#-----------
#Scree Plot
#-----------

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
explvar = pd.DataFrame(per_var, columns=[str('Percentage')], index=['PC' + str(x) for x in range(1, len(pca.components_)+1)])
 
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

#-----------
#Export All Data to an Excel File
#-----------

writer = pd.ExcelWriter('PCA_Fixed_sorted_Output.xlsx')
data.to_excel(writer, sheet_name='Input Data')
explvar.to_excel(writer, sheet_name='Scree Plot')
loading_matrix.to_excel(writer, sheet_name='Loading Matrix')
loadings2.to_excel(writer, sheet_name='Loadings')
finalcoords.to_excel(writer, sheet_name='PCA Plot Coordinates')
writer.save()

#-----------
#Plot final PCA Coordinate table
#-----------

graph = sns.relplot(x="PC1", y="PC2", hue=target, kind="scatter", data=finalcoords, s = 60)
sns.set(font_scale=1.5)
plt.xlabel('PC1 - {0}%'.format(per_var[0]), fontsize =20)
plt.ylabel('PC2 - {0}%'.format(per_var[1]), fontsize = 20)
plt.xticks(range(-7, 26, 5))  
plt.yticks(range(-7, 8, 2))
plt.tick_params(axis='both', labelsize=16)
#plt.legend(fontsize = 20)
plt.savefig('PCA_drugs_peptides.png', dpi=300)  # Save the plot at 300 DPI
#plt.show()
