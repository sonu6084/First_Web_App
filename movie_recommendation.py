
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Building%20a%20Movie%20Recommendation%20Engine/movie_dataset.csv")

def title_index(title):
    return movies[movies['original_title'] == title] ['index'].values[0]

def index_title(index):
    return movies[movies['index'] == index] ['original_title'].values[0]

def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
    except:
        pass


def predict_movies(user_liked_movie):
    
    features = ['keywords','cast','genres','director']
    for feature in features:
        movies[feature].fillna(" ", inplace = True)

    movies["combined_features"] = movies.apply(combine_features,axis=1)

    cv = CountVectorizer()
    vector = cv.fit_transform(movies['combined_features'])
    vector = vector.toarray()

    cosine_sim = cosine_similarity(vector)

    movie_index = title_index(user_liked_movie)

    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key = lambda x:x[1],reverse = True)

    recommended_movies = list()
    i = 0
    for movie in sorted_similar_movies:
        recommended_movies.append(index_title(movie[0]))
        i=i+1
        if i>5:
            break

    recommended_movies.pop(0)
    return recommended_movies







