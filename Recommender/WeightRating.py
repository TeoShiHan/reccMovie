import pandas as pd
import numpy as np
metadata = pd.read_csv('Dataset\metadata_mergeWLinks_small.csv')
 
def recommend(topN):
    import pandas as pd
    vote_counts = metadata[metadata['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = metadata[metadata['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.90)
    metadata['year'] = pd.to_datetime(metadata['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    qualified = metadata[(metadata['vote_count'] >= m) & (metadata['vote_count'].notnull()) & (metadata['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)
    
    qualified['weightedRating'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('weightedRating', ascending=False)['title'].values.tolist()
    
    return qualified[:topN]