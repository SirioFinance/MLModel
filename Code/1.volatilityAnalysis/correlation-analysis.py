import pandas as pd
import os
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

#Return the minimum lenght of all assets 
def return_min_lenght(paths, files):
    min_lenght = 1000000
    for path in paths:
        for f in os.listdir(path):
            if f != '.DS_Store':
                files.append(path+'/'+f)

    for f in files:
        df = pd.read_csv(f)

        if min_lenght > df.size/4:
            min_lenght = df.size/4
        
    return int(min_lenght)

def build_matrix(paths, files, min_lenght):

    df = {}

    for path in paths:
        for f in os.listdir(path):
            if f != '.DS_Store':
                df.update({f.replace('.csv',''): list(pd.read_csv(os.path.abspath(os.getcwd())+'/'+path+'/'+f)['price'][-min_lenght:-1])})

    return pd.DataFrame.from_dict(df)

def build_corr_matrix(df):
    corr = df.corr()
    sn.heatmap(corr, annot=True)
    plt.show()
    
paths = ['Sirio','Compound']
files = []

min_lenght = return_min_lenght(paths, files)
df = build_matrix(paths, files, min_lenght)
build_corr_matrix(df)
