# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 21:46:00 2023

@author: diosimarcardoza
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
import os
import scipy.stats 


def cambio_formato(data_):
    '''    
    Parameters
    ----------
    data_ : Data frame de pandas de interes

    Returns
    -------
    data_ data frame con variables cambiadas en formato

    '''
    data_['Q'] = pd.to_numeric(data_['Q'],errors = 'coerce')
    data_['Monto'] = pd.to_numeric(data_['Q'],errors = 'coerce')
    data_['R'] = pd.to_numeric(data_['Q'],errors = 'coerce')
    return data_


def delete_outliers(df_, lim_lower, lim_upper):   
    '''
    Parameters
    ----------
    df_ : Data frame de pandas con variables continuas a analizar y detectar outlier
    lim_lower : percentil inferior  de la distribucion para detectar posibles outlier
    lim_upper : percentil superior  de la distribucion para detectar posibles outlier
    Returns
    -------
    df : Data frame de pandas con las variables depuradas de  datos faltantes( a partir del rango intercuantilico)
        DESCRIPTION.

    '''
    # encontrar Q1, Q3 y rango intercuartílico para cada columna
    Q1 = df_.quantile (q = lim_lower)
    Q3 = df_.quantile (q = lim_upper)
    IQR = df_.apply (scipy.stats.iqr)
    # solo mantenga las filas en el marco de datos que tengan valores dentro de 1.5 * IQR de Q1 y Q3
    df = df_[~ ((df_ <(Q1-1.5 * IQR)) | (df_> (Q3 + 1.5 * IQR))). any (axis = 1)]
    return df


def smote_balanceo_clases(df_):
    '''
    
    Parameters
    ----------
    df_ : Dataframe de pandas, 
        DESCRIPTION; conjunto de datos al cual se  realizada el proceso de smote (data aumentation)

    Returns
    -------
    data_smote : data frame de pandas
        DESCRIPTION; conjunto de datos  donde se almacena la informacion real de la muestra  + datos sinteticos generados aleatoriamente concideranto 
         los vecinos mas cercanos del df_ ingresado. Asi, permitiendo balancear la data suministrada.

    '''
    
    # # targets y features
    targets = ["Fraude"]
    features = list(df_.columns)

    for tar in targets:
        features.remove(tar)

    x = df_[features]
    y = df_[targets]

    # técnica  de oversampling para balanceo de  clases
    oversample = SMOTE(sampling_strategy="auto", random_state=20)
    x_oversampling, y_oversampling = oversample.fit_resample(x, y)

    # concatenar y eliminar duplicados para tener toda la data junta
    data_all = pd.concat([x_oversampling, y_oversampling], axis=1)
    data_all = pd.concat([data_all, df_], axis=0)
    data_all.drop_duplicates(inplace=True)

    # encontrar filas en data_all que no estan en el original, el oversampling
    oversampling = data_all.merge(df_, how='outer', indicator=True).loc[
        lambda x: x['_merge'] == 'left_only']

    oversampling.drop(columns=['_merge'], inplace=True)
    
    fraud = df_[df_["Fraude"] == 1]
    
    data_smote = pd.concat([oversampling, fraud ], axis=0)
    return data_smote