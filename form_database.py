import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests

from form import movies_credits, similarity_matrix

# Load the trained data or use the existing similarity_matrix and movies_credits
# Example data loading:
# movies_credits = pd.read_csv('movies_credits.csv')
# similarity_matrix = pd.read_csv('similarity_matrix.csv')

# Initialize session_state
if 'name' not in st.session_state:
    st.session_state.name = None
if 'age_group' not in st.session_state:
    st.session_state.age_group = None
if 'email' not in st.session_state:
    st.session_state.email = None

# Function to recommend movies based on age group
def recommend_movies(age_group):
    # Your logic to recommend movies based on age group
    # For example, get movies similar to a seed movie for the specified age group
    seed_movie_index = movies_credits.sample().index[0]
    similar_movies_indices = similarity_matrix[seed_movie_index].argsort()[::-1][1:6]
    recommended_movies = movies_credits.iloc[similar_movies_indices].to_dict(orient='records')
    return recommended_movies

# Function to fetch movie poster URL using TMDB API
def fetch_poster(movie_id):
    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=4ec5bbe91fe6b4481e6a1933f24ac372&language=en-US'
    )
    data = response.json()
    poster_path = data['poster_path']
    return f'https://image.tmdb.org/t/p/w500/{poster_path}' if poster_path else None

# Streamlit app
st.title("Movie Recommendation Form")

# Input form for user information
name = st.text_input("Enter your name:")
age_group = st.selectbox("Select your age group:", ["Under 18", "18-50", "Above 50"])
email = st.text_input("Enter your email:")

# Submit button
if st.button("Submit"):
    # Store user information in session_state
    st.session_state.name = name
    st.session_state.age_group = age_group
    st.session_state.email = email

    # Display user information
    st.title("Movie Recommendations for {}".format(st.session_state.name))
    st.write("Age Group: {}".format(st.session_state.age_group))
    st.write("Email: {}".format(st.session_state.email))

    # Get movie recommendations based on age group
    recommended_movies = recommend_movies(st.session_state.age_group)

    # Display recommended movies and posters
    for movie in recommended_movies:
        st.write(f"**{movie['title']}**")
        st.image(fetch_poster(movie['id']), caption=movie['title'], use_column_width=True)
        st.write(f"Overview: {movie['overview']}")
        st.write("---")
