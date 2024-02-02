import requests
import streamlit as st
import pandas as pd
import pickle


# movies = pd.read_pickle("movies.pkl")
# similarity = pickle.load(open("similarities.pkl", "rb"))

from main import new_df, similarity

movies = new_df
similarity = similarity


def fetch_poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=2c6d217abe8d4a0fcd545cba7d582a67".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]


def recommender(movie):
    try:
        movie_index = movies[movies.title == movie].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        recommended_poster_path = []
        for i in movie_list:
            movie_id = movies.iloc[i[0]].movie_id
            recommended_movies.append(movies.iloc[i[0]].title)
            recommended_poster_path.append(fetch_poster(movie_id))
        return recommended_movies, recommended_poster_path

    except IndexError:
        print("Error: Movie not found or index out of range.")


st.title("Movie Recommender System")
selected_movie = st.selectbox("Choose a movie from the dropdown?", movies.title.values)

if st.button("Recommend"):
    movie, poster = recommender(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.header(movie[0])
        st.image(poster[0])

    with col2:
        st.header(movie[1])
        st.image(poster[1])

    with col3:
        st.header(movie[2])
        st.image(poster[2])

    with col4:
        st.header(movie[3])
        st.image(poster[3])

    with col5:
        st.header(movie[4])
        st.image(poster[4])
