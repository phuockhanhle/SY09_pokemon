import pandas as pd
import numpy as np


combats = pd.read_csv('Combats.csv')
pokemon = pd.read_csv('pokemonPCA.csv')
qualiTrait = pd.read_csv('BattleVariablesQualitativesTraite.csv')
dist_aftd = pd.read_csv('pokemon_dist.csv')


"""df = pd.merge(combats,qualiTrait,on=combats.index)
pokemon = pokemon[['#','PCA1','PCA2','PCA3','Legendary']]
pokemon = pokemon.rename(columns={'#':'First_pokemon'})

df = pd.merge(df,pokemon,on='First_pokemon')
df = df.rename(columns={'PCA1':'P1_PCA1','PCA2':'P1_PCA2','PCA3':'P1_PCA3','Legendary':'p1_legen'})
pokemon = pokemon.rename(columns={'First_pokemon':'Second_pokemon'})
df = pd.merge(df,pokemon,on='Second_pokemon')
df = df.rename(columns={'PCA1':'P2_PCA1','PCA2':'P2_PCA2','PCA3':'P2_PCA3','Legendary':'p2_legen'})

df = df.drop(columns=['key_0','First_pokemon','Second_pokemon','Unnamed: 0','p1_type1','p2_type1','Winner'])"""
#df.to_csv('traiteQuali_PCA.csv')

"""df = pd.merge(combats,qualiTrait,on=combats.index)
dist_aftd = dist_aftd.drop(columns=['Name'])
dist_aftd = dist_aftd.rename(columns={'#':'First_pokemon'})
df = pd.merge(df,dist_aftd,on='First_pokemon')
dist_aftd = dist_aftd.rename(columns={'First_pokemon':'Second_pokemon'})
df = pd.merge(df,dist_aftd,on='Second_pokemon')
df = df.drop(columns=['First_pokemon','Second_pokemon',
                      'Winner','Unnamed: 0_x','key_0','p1_type1','p2_type1',
                      'Unnamed: 0_y','Unnamed: 0'])"""
#df.to_csv('traitQuali_distAftd.csv')

"""df = pd.read_csv('Original.csv')
X = df.iloc[:,:-1]
X = X.drop(columns=['p1_id','p2_id','p1_name','p2_name'])
print(X.head())
X["p1_type1"] = pd.Categorical(X["p1_type1"])
X["p1_type2"] = pd.Categorical(X["p1_type2"])
X["p2_type1"] = pd.Categorical(X["p2_type1"])
X["p2_type2"] = pd.Categorical(X["p2_type2"])
X["p1_gen"] = pd.Categorical(X["p1_gen"])
X["p2_gen"] = pd.Categorical(X["p2_gen"])
Xp1_type1 = pd.get_dummies(X['p1_type1'], prefix = 'category')
Xp1_type2 = pd.get_dummies(X['p1_type2'], prefix = 'category')
Xp2_type1 = pd.get_dummies(X['p2_type1'], prefix = 'category')
Xp2_type2 = pd.get_dummies(X['p2_type2'], prefix = 'category')
Xp1_gen = pd.get_dummies(X['p1_gen'], prefix = 'category')
Xp2_gen = pd.get_dummies(X['p2_gen'], prefix = 'category')
X = pd.merge(X,Xp1_type1,on=X.index)
X= X.drop(columns="key_0")
X = pd.merge(X,Xp1_type2,on=X.index)
X= X.drop(columns="key_0")
X = pd.merge(X,Xp2_type1,on=X.index)
X= X.drop(columns="key_0")
X = pd.merge(X,Xp2_type2,on=X.index)
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
df = X"""
#df.to_csv('onehotencoding_origin.csv')

"""df = pd.read_csv('BattleDiffVariableQuant.csv')
df = df.drop(columns='Unnamed: 0')
df = pd.merge(df,qualiTrait,on=df.index)
df = df.drop(columns=['Unnamed: 0','key_0','p1_type1','p2_type1'])"""
#df.to_csv("traiteQuali_diff.csv")

