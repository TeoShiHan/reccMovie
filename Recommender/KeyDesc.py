from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np

metadata = pd.read_csv('Dataset\metadata_mergeWLinks_small.csv')
keywords = pd.read_csv('Dataset\keywords.csv')
qualified = pd.read_csv('Dataset\qualified.csv')

def get_recommendations_desc(title, topN):
    global metadata
    metadata['tagline'] = metadata['tagline'].fillna('')
    metadata['description'] = metadata['overview'] + metadata['tagline']
    metadata['description'] = metadata['description'].fillna('')
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(metadata['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    metadata = metadata.reset_index()
    titles = metadata['title']
    indices = pd.Series(metadata.index, index=metadata['title'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return list(titles.iloc[movie_indices])[:topN]

def recommendations_key_word(title, topN):
    import pickle
    global metadata
    
    
    with open('Dataset\count_matrix.pkl', 'rb') as f:
        count_matrix = pickle.load(f)
        
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    
    metadata = metadata.reset_index()
    titles = metadata['title']
    indices = pd.Series(metadata.index, index=metadata['title'])
    titles = metadata['title']
    
    indices = pd.Series(metadata.index, index=metadata['title'])
    idx = indices[title]    
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return list(titles.iloc[movie_indices])[:topN]


def recommend_key_word(title, topN):
    import pickle
    with open('Dataset\count_matrix.pkl', 'rb') as f:
        count_matrix = pickle.load(f)
        
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(metadata.index, index=metadata['title'])

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    
    
    movie_indices = [i[0] for i in sim_scores]
    movies = metadata.iloc[movie_indices][['title', 'vote_count', 'vote_average']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)

    
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False)['title'].to_list()
    return qualified[0:topN]


def recommend_desc(title, topN):
    global metadata
    metadata['tagline'] = metadata['tagline'].fillna('')
    metadata['description'] = metadata['overview'] + metadata['tagline']
    metadata['description'] = metadata['description'].fillna('')
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(metadata['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    metadata = metadata.reset_index()
    
    indices = pd.Series(metadata.index, index=metadata['title'])
    idx = indices[title]
    

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = metadata.iloc[movie_indices][['title', 'vote_count', 'vote_average']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)

    
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False)['title'].to_list()
    return qualified[0:topN]


def recommend(title, topN):
    rec1 = set(recommend_key_word(title, topN))
    rec2 = set(recommend_desc(title, topN))
    return list(rec1.union(rec2))
