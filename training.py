# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:10:02 2023

@author: diosimarcardoza
"""
# cargue de  librerias y modulos requeridos para el analisis
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# carga de datos  a utilizar para el proceso de training de modelos ML               
#####___ training dataset___#####
data_train = pd.read_csv("Data/trainSet.csv")

xtrain = data_train[data_train.columns[:-1]]
ytrain = data_train.Fraude

####______testing dataset_______####
data_test= pd.read_csv("Data/testSet.csv")

xtest = data_test[data_test.columns[:-1]]
ytest = data_test.Fraude

### Se determinan 4 modelos de  clasificacion ML para ajustar a los datos y evaluar el rendimiento para detectar fraude transaccional en la plataforma de pagos
classifiers = [
    DecisionTreeClassifier(),
   RandomForestClassifier(),
    GradientBoostingClassifier(),
    SGDClassifier()
    ]

# definicion de pipeline para ajuste previo de modelos ML seleccionados para la clasificacion de transacciones fraudulentas Vs no fraudulentas
top_class = []
top_score = []
for classifier in classifiers:
    pipe = Pipeline(steps=[
                      ('classifier', classifier)])
    
    # training model
    pipe.fit(xtrain, ytrain)   
    print(classifier)
    
    acc_score = pipe.score(xtest, ytest) # calculo del score del acurracy por modelos 
    print("model score: %.3f" % acc_score)
    
    # using the model to predict
    y_pred = pipe.predict(xtest)
    
   #target_names = [le_name_mapping[x] for x in le_name_mapping]
    print(classification_report(ytest, y_pred))
    
    ## seleccion de modelos previos con un accurracy superior al 70 % en ajuste.
    if acc_score > 0.70:
        top_class.append(classifier)
        top_score.append(acc_score)
        
dict_params = dict(zip(top_class, top_score))
        
# mejor modelo ajustado
topModels =  pd.DataFrame([[str(key)[:-2],dict_params[key] ] for key in dict_params.keys() ] , columns=['Model', 'acc_score'])
topModels.to_csv('output/topModels_pretraining.csv', encoding='utf-8',index = None)
model  = topModels[topModels.acc_score ==  topModels.acc_score.max() ]


####_______ tunning de hiperparametros del mejor modelo pre-entrenado__________####

### Parametros para tunning  por modelo
parameters = {
    ## GradientBoostingClassifier 
    'GradientBoostingClassifier' : {
               "n_estimators" : [5,50,250,500],
               "max_depth":[1,3,5,7,9],
               "learning_rate":[0.01,0.1,1,10,100]
              },
    
    ## Decision Tree
    'DecisionTreeClassifier' : {'criterion':['gini', 'entropy'],
              'splitter':['best','random'],
              'max_depth':list(range(1,50)),
              }   ,
    
    ## RandomForestClassifier
    'RandomForestClassifier' : {
    "n_estimators":[5,10,50,100,250],
    "max_depth":[2,4,8,16,32,None]
    
             }
   
}


###### tunning  de los modelos  con mas accurracy previo 
name = []
accuracy = []
mod = []
for i in range(0,3):
    modelo = eval(list(topModels.Model)[i] + '(random_state=42)')
    locals()["grid_" + str(i)] = GridSearchCV(  modelo , parameters[list(topModels.Model)[i]],cv=5)
    locals()["grid_" + str(i)].fit(xtrain ,ytrain)
    y_pred_ = locals()["grid_" + str(i)].predict(xtest)
    Accuracy_score_ =  accuracy_score(ytest,y_pred_)
    accuracy.append(Accuracy_score_)
    #mod.append(locals()["grid_" + str(i)])
    mod.append("grid_" + str(i))
    name.append(list(topModels.Model)[i])
    
    
score_final = pd.DataFrame((zip(name, accuracy)), columns = ['Name_model', 'Accuracy'])

#Extraccion del modelo con mayor  accuracy 
index = accuracy.index(max(accuracy))
model =  eval(mod[index]) 

# guardar el mejor modelo entrenado
import pickle
filename = 'output/finalized_model.pkl'
pickle.dump(model, open(filename, 'wb'))