
"""
Created on Sat Mar 23 13:03:07 2019

@author: Haythem
"""
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
from sklearn.decomposition import PCA

#KNN
X = np.load("C:\\Users\\TOSHIBA\\Desktop\\basedonnée1\\features1.npy")
y = np.load('C:\\Users\\TOSHIBA\\Desktop\\basedonnée1\\labels1.npy').ravel()
#ACP: 10 composantes 
pca = PCA(n_components=10)
pca.fit(X)
X= pca.fit_transform(X)
knn=KNeighborsClassifier(n_neighbors=8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
scaler = MinMaxScaler()#centrer et réduire les données
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn=knn.fit(X_train_scaled,y_train)
ypred=knn.predict(X_test_scaled)
conf=confusion_matrix(y_test,ypred)
df_cm = pd.DataFrame(conf, index = [i for i in ['Dog ','baby cry', 'Rooster']], columns = [i for i in  ['Dog',  'baby cry',  'Rooster']])

plt.figure(figsize=(8,6))
sns.heatmap(df_cm, annot=True)
plt.title('matrice de confusion du knn')
plt.ylabel('la vraie classe ')
plt.xlabel('classe prédite')
plt.show()

print('le score de k plus proche voisin est égal à=',knn.score(X_test_scaled,y_test))

#SVM
from sklearn.svm import SVC
svmcl = SVC(kernel='linear',C=28, gamma = 0.0001, decision_function_shape="ovr") 
svmcl.fit(X_train_scaled, y_train)
ypred=svmcl.predict(X_test_scaled)
conf=confusion_matrix(y_test,ypred)
df_cm = pd.DataFrame(conf, index = [i for i in ['Dog',   'baby cry','Rooster']], columns = [i for i in  ['Dog', 'baby cry',   'Rooster']])

plt.figure(figsize=(8,6))
sns.heatmap(df_cm, annot=True)
plt.title('matrice de confusion du SVM linéaire')
plt.ylabel('la vraie classe ')
plt.xlabel('classe prédite')
plt.show()
print('le score des SVM est égal à=',svmcl.score(X_test_scaled,y_test))

