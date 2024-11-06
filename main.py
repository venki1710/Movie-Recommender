import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tmdbv3api import TMDb, Movie
import streamlit as st

tmdb = TMDb()
tmdb.api_key = '9e0b7f57d3b5f874fea0e35a82b5f54e'
movie = Movie()

# similarity_matrix = pickle.load(open('similarity.pkl', 'rb'))
df = pd.read_csv('highrated1.csv')

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['final'])
similarity_matrix = cosine_similarity(count_matrix)

def getposters(title):
    search = movie.search(title)
    path = search[0].poster_path
    # print('https://image.tmdb.org/t/p/original{}'.format(path))
    return ('https://image.tmdb.org/t/p/original{}'.format(path))

def recommend(movie):
    movie_idx = df[df['title'] == movie].index[0]
    distances = similarity_matrix[movie_idx]
    #to get indices of movies 
    distances_with_idx = enumerate(distances)
    #sorting based on similarities which is 1st index of tuple
    similar_list = sorted(list(distances_with_idx), reverse = True, key = lambda x : x[1])
    top_list = similar_list[1:11] 
    
    recommended_movies = []
    recommended_posters = []
    for i in top_list:
        title = df.loc[i[0]].title
        recommended_movies.append(title)
        
        #fetching posters
        recommended_posters.append(getposters(title))
        
    return recommended_movies, recommended_posters 

st.title('Movie Recommender')

selected = st.selectbox(
    'Select a Movie',
    df['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected)
    col1, col2, col3 = st.columns(3, gap = 'large')
    with col1:
        st.image(posters[0], width = 200)
        st.text(names[0])
    with col2:
        st.image(posters[1], width = 200)
        st.text(names[1])

    with col3:
        st.image(posters[2], width = 200)
        st.text(names[2])
    st.text("")
    st.text("")
    col4, col5, col6 = st.columns(3, gap = 'large')
    with col4:
        st.image(posters[3])
        st.text(names[3])
    with col5:
        st.image(posters[4])
        st.text(names[4])
    with col6:
        st.image(posters[5])
        st.text(names[5])
    st.text("")
    st.text("")
    col7, col8, col9 = st.columns(3, gap = 'large')
    with col7:
        st.image(posters[6])
        st.text(names[6])
    with col8:
        st.image(posters[7])
        st.text(names[7])
    with col9:
        st.image(posters[8])
        st.text(names[8])
    # with col10:
    #     st.text(names[9])
    #     st.image(posters[9])

