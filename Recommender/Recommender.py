from sqlalchemy import false
from Recommender.ItemBasedCollaborativeFiltering import recommend as IBCFrecc
from Recommender.WeightRating import recommend as WRrecc
from Recommender.SVDUserBasedCollaborativeFiltering import recommend as SVDUBCF
from Recommender.NeuralCollaborative import recommend as NCF
from Recommender.GenreRecc import recommend as G
from Recommender.KeyDesc import recommend as KD
import heapq
import pandas as pd


title_id = pd.read_csv('Dataset/all_movies.csv')



def isEnglish(s):
    result = title_id[
                title_id['movie_title'] == s
             ]
    try:
        pd.Series(result['imdb_id']).to_list()[0]
        return True
    except:
        return False

def english_only(list_of_titles):
    for title in list_of_titles:
        if(not isEnglish(title)):
            list_of_titles.remove(title)
    return list_of_titles

class Recommender:
    def __init__(self):
        pass
    
    # collaborative filtering
    def IBCF(self, movit_title, topN):
        try:
            recc = IBCFrecc(movit_title, topN+20)
            recc = english_only(recc)
            if(len(recc)<= topN):
                return recc
            else:
                return recc[:topN]
        except:
            return[]
    
    def SVDUCF(self, user_id, topN):
        try:
            view, recc = SVDUBCF(user_id, topN+20)
            recc = english_only(recc)
            view = english_only(view)
            if(len(recc)<= topN):
                pass
            else:
                recc = recc[:topN]
            return heapq.nlargest(20, view), recc
        
        except:
            return[]
        
    def NCF(self, user_id, topN):
        try:
            view, recc = NCF(user_id, topN+20)
            recc = english_only(recc)
            view = english_only(view)
            if(len(recc)<= topN):
                pass
            else:
                recc = recc[:topN]
            return heapq.nlargest(20, view), recc
        except:
            return[]
    
    # content based       
    def WR(self, topN):
        try:
            recc = WRrecc(topN+20)
            recc = english_only(recc)
            if(len(recc)<= topN):
                return recc
            else:
                return recc[:topN]
        except:
            return[]
    
    def G(self,genre, topN):
        try:
            recc = G(genre, topN+20)
            recc = english_only(recc)
            if(len(recc)<= topN):
                return recc
            else:
                return recc[:topN]
        except:
            return[]
    
    def KD(self, title, topN):
        try:
            recc =  KD(title, topN+20)
            recc = english_only(recc)
            if(len(recc)<= topN):
                return recc
            else:
                return recc[:topN]
        except:
            return[]
