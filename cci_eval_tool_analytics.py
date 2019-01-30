import os
import pandas as pd
import numpy as np
import random

os.chdir(input("Specify the path where the file is located: "))

df = pd.read_csv(input("Specify the name of the file: "))

df.drop(["name"],1,inplace= True)

#First we need to store all variables in 2 separate lists

cols = list(df)

variables = np.arange(len(cols))

#The next step is to import the cci reference table file 

cci = pd.read_csv(input("Specify the name of the file: "))

cci_list = np.arange(len(cci))

names = cci["CCI Name"]

#Constructing a new dataframe

#Now that we have both the rows and the columns of the new df, lets construct it

df = pd.DataFrame(index=names, columns= cols)

#Now lets populate it with dummy data for the sake of this run

for x in cci_list:
    for y in variables:
        df.iloc[x,y]= random.choice("ABCDE")     


#Now we need to construct a dictionary for the score mapping

score_mapping = {"A": 5, "B":4, "C":3, "D": 2, "E":1}


#A function that calculates the total score for a CCI

def total_scores(col_list,cci):
    for x in col_list:
        df[x].replace(score_mapping, inplace=True)
    new_df= df.apply(pd.to_numeric)
    score = new_df.loc[cci].sum()
    return score

#Testing the function

total_scores(cols,'Jeevan Jyothi Rao ')

#A function to calculate the total scores per category

#First let us set up the columns assigned to each category

health= [x for x in df.columns if "health_" in x][1:]
hygiene= [x for x in df.columns if "hygiene_" in x]
nutrition= [x for x in df.columns if "nutrition_" in x]
infra = [x for x in df.columns if "infra_" in x]
protection= [x for x in df.columns if "protection_" in x]
education = [x for x in df.columns if "education_" in x][:-1]
aftercare = [x for x in df.columns if "aftercare_" in x]
psycare_cm= [x for x in df.columns if "psycare_" in x]
governance= [x for x in df.columns if "governance_" in x]
legal = [x for x in df.columns if "legal_" in x][1:]
management = [x for x in df.columns if "management_" in x]
ecrec= [x for x in df.columns if "ecrec_" in x]


def score_calculator(cci):
    """Split the df into different categories"""
    
    health_df = df[health]
    hygiene_df =df[hygiene]
    nutrition_df= df[nutrition]
    infra_df= df[infra]
    protection_df= df[protection]
    psycare_cm_df= df[psycare_cm]
    aftercare_df= df[aftercare]
    governance_df= df[governance]
    legal_df= df[legal]
    management_df= df[management]
    ecrec_df= df[ecrec]
    
    """Initialize a dictionary where the scores will be added"""

    scores_dict= {}

    """ Compute the scores of each category for the CCI"""

    health_score = health_df.loc[cci].sum()
    hygiene_score = hygiene_df.loc[cci].sum()
    nutrition_score = nutrition_df.loc[cci].sum()
    infra_score = infra_df.loc[cci].sum()
    protection_score = protection_df.loc[cci].sum()
    psycare_cm_score = psycare_cm_df.loc[cci].sum()
    aftercare_score = aftercare_df.loc[cci].sum()
    governance_score = governance_df.loc[cci].sum()
    legal_score = legal_df.loc[cci].sum()
    management_score = management_df.loc[cci].sum()
    ecrec_score = ecrec_df.loc[cci].sum()
        
    scores_dict[cci]= [health_score, hygiene_score, nutrition_score, infra_score, protection_score,psycare_cm_score,
                  aftercare_score, governance_score, legal_score, management_score, ecrec_score] 
    
    scores_df= pd.DataFrame.from_dict(scores_dict)
    scores_df.index= ["health","hygiene","nutrition","infra","protection","psycare","aftercare","governance",
                     "legal","management","ecrec"]
    scores_df= scores_df.T
    return scores_df

    #Testing the function

    score_calculator('Bethesda Life Centre Small Boys Home ')

    #Now that we have the scores per category we want to create our final df for all CCI's and their scores

list_df= []

for x in df.index:
    y= score_calculator(x)
    list_df.append(y)

df = pd.concat(list_df)

df["total_score"]= df["health"]+ df["hygiene"]+df["nutrition"]+df["infra"]+df["protection"]+df["psycare"]+df["aftercare"]+df["governance"]+ df["legal"]+df["management"]+df["ecrec"]

df.to_csv("cci_scores.csv")

#A function to calculate the highest score 

def highest_score():
    return df.loc[df["total_score"]== df["total_score"].max()]

 print (highest_score().index[0])

 # A function to calculate the highest scores per category

 def category_highest(category_list):
    category_highest= {}
    for x in category_list:
        category_highest[x]= df.loc[df[x]== df[x].max()].index[0]
    return pd.DataFrame.from_dict(category_highest, orient='index')

#Applying the function to all columns 

category_highest(df.columns)

#A function to calculate lowest scores for each category

def category_lowest(category_list):
    category_highest= {}
    for x in category_list:
        category_highest[x]= df.loc[df[x]== df[x].min()].index[0]
    return pd.DataFrame.from_dict(category_highest, orient='index')

#Applying the function

category_lowest(df.columns)

#Clustering the orphanages using K- Means

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

# First we need to determine the optimal number of clusters "k"

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df)
    kmeanModel.fit(df)
    distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method for finding the optimal k')
plt.show()

#Applying the classifier to our data

kmeans= KMeans(n_clusters=2)

kmeans.fit(df)

clusters= kmeans.labels_

df["cluster"]= cluster_labels 


