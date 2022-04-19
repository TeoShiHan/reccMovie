import pandas as pd
from surprise import dump



n, model = dump.load('Model/reccSvd.pickle')
recommendable = pd.read_csv('Dataset/recommendable.csv')
ratings = pd.read_csv('Dataset/ratings.csv')
movies = pd.read_csv('Dataset/movies.csv')


def recommend(user_id, topN):
    movie = pd.Series(recommendable['movie_id'])
    movie_set = set(movie)
    condition = ratings['userId'] == user_id
    viewed_id = set(ratings.loc[condition]['movie_id'])
    recc = movie_set - viewed_id
    map = {
        'movie_id':[],
        'predict_rating':[]}
    for movie_id in recc:
        map['movie_id'].append(movie_id)
        map['predict_rating'].append(model.predict(user_id,movie_id)[3])
    recommendation =\
        pd.Series(
            pd.DataFrame(map)\
                .sort_values(by='predict_rating', ascending=False)\
                .head(topN)['movie_id']
        ).tolist()
    titles = []
    for movie_id in recommendation:
        q_title = movies['movie_id'] == movie_id
        title = movies.loc[q_title]['movie_title'].iloc[0]
        titles.append(title)
    
    viewed_movies = []    
    
    for id in viewed_id:
        nv_title = movies['movie_id'] == id
        title = movies.loc[nv_title]['movie_title'].iloc[0]
        viewed_movies.append(title)
        
    return viewed_movies, titles
