import numpy as np
import pandas as pd
from sklearn.preprocessing  import LabelEncoder


def fun_remove_object_cols(df):
    df = df.select_dtypes(exclude= 'object')
    return df

def fillna(df):
    df = df.fillna(0)
    return df 

def preproces(df):
    
    ##droping negatively correalted colns with target variablt
    df.corr()['SalePrice'].sort_values(ascending = False)
    x =pd.DataFrame(df.corr()['SalePrice'].sort_values(ascending = False))
    to_drop  = x[x['SalePrice'] <  0].index
    to_drop.tolist()    
    df_1 = df.drop(to_drop, axis =1)
        
    ###remove colms having more than 75% Null values 
    X=  pd.DataFrame(df_1.isnull().sum()/df_1.shape[0] * 100, columns=['Missing_count'])
    to_drop_1  = X[X['Missing_count'] > 75 ].index
    to_drop_1.tolist()
    df_2 = df_1.drop(to_drop_1, axis =1)
    
    ###lets do label encoding
    le=LabelEncoder()
    df_processed = df_2.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
    
    return df_processed


#print('code working')
