# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 18:06:17 2023

@author: diosimarcardoza
"""
# se importa las ribrerias(modulos) necesarios para  el analisis exploratorio de  datos para determinar la calidad del dataset

import pandas as pd
from utils.functions import cambio_formato, smote_balanceo_clases
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# cargar  archivo de datos 
data = pd.read_csv("Data/IG Data Scientist Challenge.csv")

# se realiza tranformación sobre los campos Q, R y monto. los cuales inicialmente se encuentran en formato string y se cambia a numerico.
# con el objetivo de realizar calculos operacionales descriptivos.

data = cambio_formato(data)

##### Inicio de ingenieria de datos para depurar y limpiar dataset a utilizar para entrenamientos de modelos ML o AI
df = data.J 

data.drop(['K', 'J'], axis = 'columns', inplace=True) # Se elimina la columna K dado a que tiene un 76.20% de datos faltantes, basado en teoria estadistica de que tiene mas del 20% de informacion perdida


# se aplica el metodo de KNNimputer ( metodo para imputar valores faltantes considerando los vecinos mas cercanos - puntos proyectados en el hiperplano)

knn = KNNImputer(n_neighbors=5) # se define el modelo de imputacion concideranto las 5 proyecciones mas cercanas al valor faltante
neighbors = pd.DataFrame(knn.fit_transform(data)) # se entrena y transforma la data para aplicar la imputacion de datos faltantes
neighbors.columns = data.columns # se ajusta nombre de columnas
df_ = pd.merge(neighbors, df, left_index=True, right_index=True)
##________________________________________________________________________________________________________##
## eliminar los valores negativos existentes en el dataframe

df_ = df_.loc[( df_.S >=0) & (df_.B >= 0)]

# codificacion de  variables multiclass ( J - ubicacion de  la transaccion virtual) 
le = LabelEncoder() # pipeline para codificar VARIABLES
df_.J = le.fit_transform(df_.J) # Aplicacion de la tranformada de codificaion categorica de la variable J
le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_)) # Dicionario del lebelencoder

x, y = df_[df_ .columns.difference(['Fraude'])] , df_.Fraude # identificar caracteristicas (X) y variable objetivo (Y)



# división del conjutno de datos considerando el 20 % para testing de modelo ML
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y,
    random_state=20)

#____________________________________________ balanceo de clases _____________________________________________#
train_data = pd.concat([x_train, y_train], axis=1) # se concatena los features y el target de entrenamiento

oversampling = smote_balanceo_clases(train_data) # se realiza balanceo por medio de teccnicas de data aumentations con remuestreo (smote) para el grupo de transacciones fraudulentas 
No_fraud= train_data[train_data.Fraude ==0]

trainSet = pd.concat([oversampling, No_fraud])
testSet  = pd.concat([x_test, y_test], axis=1)


#_______ organizar formato de campos_________# 
 
int_num = ['A', 'B', 'C', 'D', 'E', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Fraude']
cols_float = ['S', 'Q']
for i in trainSet.columns :
    if i in int_num:
        trainSet[i] =   trainSet[i].round(0).astype(int)
        testSet[i] =   testSet[i].round(0).astype(int)
    elif  i in cols_float :
        trainSet[i] =  trainSet[i].round(2)
        testSet[i] =   testSet[i].round(2)
    else :
        trainSet[i] =  trainSet[i].round(3)
        testSet[i] =   testSet[i].round(3)
        
### almacenamiento local de archivos procesdos 
trainSet .to_csv('Data/trainSet .csv', encoding='utf-8',index = None)
testSet.to_csv('Data/testSet.csv', encoding='utf-8',index = None)

######################################################################################
#### metodo de balanceo de datos utilizando redes neuronales  bajo modelos GANG , opcional 
'''
from tabgan.sampler import GANGenerator
# from tabgan.sampler import OriginalGenerator, GANGenerator

# cargar datos originales
path = "data/card_featured.csv"
df = pd.read_csv(path)

# separar los datos de fraude para usarlos en el conjunto de test
fraud = df_[df_["Fraude"] == 1]
fraud.reset_index(drop=True, inplace=True)

# cargar datos oversampling con smote
path = "data/creditcard_smote_oversampling.csv"
over = oversampling.copy()

# ordenar para no tener problemas de data drift en el entrenamiento
df_concat = pd.concat([df_, over], axis=0)
df_concat = df_concat.sample(frac=1)

df_concat.drop(columns=['_merge'], inplace=True)

# # targets y features
targets = ["Fraude"]
features = list(df_concat.columns)

for tar in targets:
    features.remove(tar)
    
x = df_concat[features]
y = df_concat[targets]

# división del conjutno de datos
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y,
    random_state=20)
'''

