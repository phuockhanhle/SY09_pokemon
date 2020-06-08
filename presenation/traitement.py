import pandas as pd
import numpy as np    

def onehotencoding(df):
    X = df.iloc[:,:-1]
    X = X.drop(columns=['p1_id','p2_id','p1_name','p2_name'])
    X["p1_type1"] = pd.Categorical(X["p1_type1"])
    X["p1_type2"] = pd.Categorical(X["p1_type2"])
    X["p2_type1"] = pd.Categorical(X["p2_type1"])
    X["p2_type2"] = pd.Categorical(X["p2_type2"])
    X["p1_gen"] = pd.Categorical(X["p1_gen"])
    X["p2_gen"] = pd.Categorical(X["p2_gen"])
    Xp1_type = X[["p1_type1","p1_type2"]]
    Xp1_type = pd.get_dummies(Xp1_type, prefix = 'category')
    cats = np.unique(Xp1_type.columns)
    for i in range(0,18):
        Xp1_type['p1'+str(cats[i])[8:]] = Xp1_type.iloc[:,i] + Xp1_type.iloc[:,i+18]
    Xp1_type = Xp1_type.iloc[:,36:]

    Xp2_type = X[["p2_type1","p2_type2"]]
    Xp2_type = pd.get_dummies(Xp2_type, prefix = 'category')
    cats = np.unique(Xp2_type.columns)
    for i in range(0,18):
        Xp2_type['p2'+str(cats[i])[8:]] = Xp2_type.iloc[:,i] + Xp2_type.iloc[:,i+18]
    Xp2_type = Xp2_type.iloc[:,36:]

    Xp1_gen = pd.get_dummies(X['p1_gen'], prefix = 'category')
    Xp2_gen = pd.get_dummies(X['p2_gen'], prefix = 'category')

    X = pd.merge(X,Xp1_type,on=X.index)
    X= X.drop(columns="key_0")
    X = pd.merge(X,Xp2_type,on=X.index)
    X= X.drop(columns="key_0")
    X = pd.merge(X,Xp1_gen,on=X.index)
    X= X.drop(columns="key_0")
    X = pd.merge(X,Xp2_gen,on=X.index)
    X= X.drop(columns="key_0")
    X = X.drop(columns=["p1_type1","p1_type2","p2_type1","p2_type2","p1_gen","p2_gen"])
    X.loc[X["p1_legen"]=='t', "p1_legen"] = 1
    X.loc[X["p1_legen"]=='f', "p1_legen"] = 0
    X.loc[X["p2_legen"]=='t', "p2_legen"] = 1
    X.loc[X["p2_legen"]=='f', "p2_legen"] = 0
    X['winner'] = df['winner']
    df = X
    return df

def Differ(df):
    X = df[['p1_hp','p1_atk','p1_def','p1_spatk','p1_spdef','p1_sp'
            ,'p2_hp','p2_atk','p2_def','p2_spatk','p2_spdef','p2_sp']]
    for i in range(6):
        X['dif'+str(X.columns[i])[2:]] = X.iloc[:,i] - X.iloc[:,i+6]
    X = X.iloc[:,12:]
    df = pd.merge(df,X,on=X.index)
    df= df.drop(columns="key_0")
    df = df.drop(columns=['p1_hp','p1_atk','p1_def','p1_spatk','p1_spdef','p1_sp'
                          ,'p2_hp','p2_atk','p2_def','p2_spatk','p2_spdef','p2_sp'])
    return df

def TraitQuali(df):
    X_type1 = df[['p1_type1','p2_type1']]
    y = df[["winner"]]
    y.loc[y["winner"]=="f", "winner"] = 0
    y.loc[y["winner"]=="t", "winner"] = 1
    df1 = pd.concat([X_type1,y],axis =1,sort=False)
    type = np.unique(X_type1.p1_type1)
    tAvantage = np.zeros([18,18])
    for i in range(len(type)):
        for j in range(len(type)):
            tAvantage[i,j] = df1.loc[((df1.p1_type1 == type[i]) & (df1.p2_type1 == type[j]) & (df1.winner == 1)) |
                                    ((df1.p1_type1 == type[j]) & (df1.p2_type1 == type[i]) & (df1.winner == 0))].shape[0]
    nbrContre = np.zeros([18,18])
    for i in range(18):
        for j in range(18):
            nbrContre[i,j] = df1.loc[((df1.p1_type1 == type[i]) & (df1.p2_type1 == type[j]))|
                                    ((df1.p1_type1 == type[j]) & (df1.p2_type1 == type[i]))].shape[0]
    tmp = np.zeros([18,18])
    for i in range(18):
        for j in range(18):
            tmp[i,j] = tAvantage[i,j]/nbrContre[i,j]
    
    df1 = df1.assign(probWinParType=0)
    df1 = df1.assign(nbrContre=0)

    for i in range(len(type)):
        for j in range(len(type)):
            df1.loc[(df1.p1_type1 == type[i]) & (df1.p2_type1==type[j]),'nbrContre']= nbrContre[i,j]
            df1.loc[(df1.p1_type1 == type[i]) & (df1.p2_type1==type[j]),'probWinParType']= tmp[i,j]

    tTraitDataQuali = df1[['probWinParType','nbrContre']]
    df = pd.merge(df,tTraitDataQuali,on=df.index)
    df.drop(columns=['p1_type1','p1_type2','p2_type1','p2_type2','key_0'])
    return df

