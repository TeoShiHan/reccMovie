import pickle
import pandas as pd
from scipy import sparse

sparse_matrix = sparse.load_npz('Dataset/sparse_matrix.npz')
movies = pd.read_csv('Dataset/movies.csv')
recommendable = pd.read_csv('Dataset/recommendable.csv')
knn = pickle.load(open( "Model/knn_cfr_item.pkl", "rb" ))
rating_matrix = pd.read_csv('Dataset/matrix_row_index.csv')

def recommend(movie_title, topN):
    
    knn = pickle.load(open( "model/knn_cfr_item.pkl", "rb" ))
    
    q_getMovie = recommendable['movie_title'] == movie_title
    result = recommendable.loc[q_getMovie]
	
    if result.empty:
        return []
    else:
        movie_id = result['movie_id'].iloc[0]  
    
    
    # query position in matrix
    q_get_movie_id = rating_matrix['movie_id'] == movie_id
    matrix_pos = rating_matrix[q_get_movie_id].index[0]
    
    
    # get distance and indices of the movies
    distances, indices = knn.kneighbors(sparse_matrix[matrix_pos],n_neighbors=topN+1)
    
    
    # array of tuple (index, distance)
    oneD_indices   = indices.squeeze().tolist()
    oneD_distances = distances.squeeze().tolist()
    index_distance = list(zip(oneD_indices, oneD_distances))
    
    
    # sort tuple by distance, descendingly
    sort_by_tuple = lambda tuples: tuples[1] # second element in tupel
    rec_movie_indices = sorted(index_distance, key= sort_by_tuple)
    
    
    # recommend
    rec_movie = []
    for recommendation in rec_movie_indices:
        index_in_sparse = recommendation[0]
    
        # get column of based on sparse
        ratingCol = rating_matrix.iloc[index_in_sparse]
        movie_id = ratingCol['movie_id']
        query = movies['movie_id'] == movie_id
        rec_movie.append(movies.loc[query]['movie_title'].iloc[0])
    
    return rec_movie