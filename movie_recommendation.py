import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from fuzzywuzzy import process
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

movies = pd.read_csv('Data\\movies.csv')
ratings = pd.read_csv('Data\\ratings.csv')
#cold_start_experiment_ratings = pd.read_csv('Data\\experiment_data.xlsx')
movie_titles = dict(zip(movies['movieId'], movies['title']))

def main():

    """ Baseline method """

    movie_id = 114074

    movie_recommendation = collaborative_filter("The Skeleton Twins (2014)", k=5)
    movie_title = movie_titles[movie_id]

    print(f"Because you watched {movie_title}")
    for i in movie_recommendation:
        print(movie_titles[i])

    """ Refined design """

    data_cleanup(movies)
    get_movie_features(movies)
    title = 'The Skeleton Twins'
    movie_recommendation = content_based_filter(title, recommendations=5)

    print(f"Recommendations for {title}:")
    print(movies['title'].iloc[movie_recommendation])


def create_user_item(df):
    """
    Generates a sparse matrix.

    Args:
        df: pandas dataframe

    Returns:
        user_item: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    user_item = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return user_item, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper


def collaborative_filter(movie_name, k=5, metric='cosine', show_distance=False):

    """
    Finds k-nearest neighbours for a given movie id.

    Args:
        ratings: pandas dataframe
        movie_id: id of the movie of interest
        k: number of similar movies to retrieve
        metric: distance metric for kNN calculations

    Returns:
        list of k similar movie ID's
    """

    user_item, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_user_item(ratings)
    neighbour_ids = []
    movie_id = get_movie_id(movie_name)
    movie_ind = movie_mapper[movie_id]
    movie_vec = user_item[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(user_item)
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

def get_year(movie):
    year = 2000
    if "(" in movie:
        year = movie.replace(")", "").split("(")[-1]
        if "–" in year:
            year = year.split("–")[0]
    return int(year)

def data_cleanup(movie):
    movie['genres'] = movie['genres'].apply(lambda x: x.split("|"))
    movie['year'] = movie['title'].apply(lambda x: get_year(x))
    return movie

def round_down(year):
    return year - (year % 10)

def get_movie_features(movie):
    df_genres_count = Counter(genre for genres in movie['genres'] for genre in genres)
    # print(f"There are {len(df_genres_count)} genre labels.")
    """ deleting no genres listed """
    del df_genres_count['(no genres listed)']
    genres = list(df_genres_count.keys())
    for genre in genres:
        movie[genre] = movie['genres'].transform(lambda x: int(genre in x))
    movie['decade'] = movie['year'].apply(round_down)
    movie_decades = pd.get_dummies(movie['decade'])
    df_movie_features = pd.concat([movie[genres], movie_decades], axis=1)
    #print(df_movie_features.head())
    return df_movie_features


def cosine_similiarity(df_movie_features):
    cosine_score = cosine_similarity(df_movie_features, df_movie_features)
    return cosine_score

def get_movie_title(movie,movie_title):
    all_movie_titles = movie['title'].tolist()
    movie_match = process.extractOne(movie_title, all_movie_titles)
    return movie_match[0]

def similar_movies_content(movie):
    title = get_movie_title(movie,'juminji')
    movie_index = dict(zip(movie['title'],list(movie.index)))
    movie_features = get_movie_features(movie)
    cosine_score = cosine_similiarity(movie_features)
    index = movie_index[title]
    recommendations = 10
    similarity_scores = list(enumerate(cosine_score[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:(recommendations + 1)]
    movie_recommendation = [i[0] for i in similarity_scores]
    return movie_recommendation

def content_based_filter(title_string, recommendations=5):
    title = get_movie_title(movies, title_string)
    movie_index = dict(zip(movies['title'], list(movies.index)))
    index = movie_index[title]
    movie_features = get_movie_features(movies)
    cosine_score = cosine_similiarity(movie_features)
    similarity_scores = list(enumerate(cosine_score[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:(recommendations+1)]
    movie_recommendation = [i[0] for i in similarity_scores]
    return movie_recommendation

def evaluate_contentbased(ratings):
    mean = ratings.groupby(['movieId']).mean()
    print(mean)


def evaluation_data(ratings):
    experiment_data = ratings.sample(frac=.2)
    return None

def hybrid_model(user_id, liked_movie_name):
    num_movies = len(ratings.loc[ratings['userId'] == user_id].index)
    if num_movies >= 50:
        movie_recommendations = collaborative_filter(liked_movie_name)
        print(f"Because user_id : {user_id} liked {liked_movie_name}")
        print("Recommending below movies through collaborative filtering")
        for i in movie_recommendations:
            print(f"{i}            {movie_titles[i]}")
    else:
        data_cleanup(movies)
        get_movie_features(movies)
        movie_recommendations = content_based_filter(liked_movie_name)
        print(f"Because user_id : {user_id} liked {liked_movie_name}")
        print("Recommending below movies through content based filtering")
        print(movies['title'].iloc[movie_recommendations])

def get_movie_id(movie_name):
    movie_id = movies.loc[movies["title"] == movie_name]["movieId"]
    return movie_id.values[0]


if __name__ == '__main__':
    # hybrid_model(1, "Toy Story (1995)")
    print()
    hybrid_model(2, "Toy Story (1995)")
    # main()



