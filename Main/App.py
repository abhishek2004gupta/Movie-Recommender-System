import streamlit as st
import pickle
import pandas as pd
import requests
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))
top_ratings = pickle.load(open('top_ratings.pkl', 'rb'))
movies_dict = pickle.load(open('movie_dict2.pkl', 'rb'))
movies2 = pd.DataFrame(movies_dict)
similarity2 = pickle.load(open('similarity2.pkl', 'rb'))
vectors = pickle.load(open('vector2.pkl', 'rb'))
search = pickle.load(open('search2.pkl', 'rb'))
search = sorted(search)


st.title("Movie Recomendation System")

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=f5da9b0396c6540468e572000662f2f1&append_to_response=videos,images'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
def fetch_trailer(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=f5da9b0396c6540468e572000662f2f1&append_to_response=videos,images'.format(
            movie_id))
    data = response.json()
    try:
        return "https://www.youtube.com/watch?v=" + data['videos']['results'][0]['key']
    except:
        return -1


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_movies_posters = []
    trailers =[]
    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        # fetching movie poster from API
        recommended_movies_posters.append(fetch_poster(movie_id))
        trailers.append(fetch_trailer(movie_id))
    return recommended_movies,recommended_movies_posters,trailers


# Fit the CountVectorizer on the movie titles (not in the recommend function)
cv = CountVectorizer(max_features=5000, stop_words='english')
cv.fit(movies2['title'])  # Fit on the movie titles
# Create the vectorized representation of movie titles to match dimensions
vectorized_titles = cv.transform(movies2['tags']).toarray()


def recommender(query):
    recommended_movies2 = []
    recommended_movies2_posters = []
    trailers = []

    # If the query matches a movie title, use its index. Otherwise, treat it as a tag search.
    if query in movies2['title'].values:
        movie_index = movies2[movies2['title'] == query].index[0]
        distances = similarity2[movie_index]
    else:
        # Vectorize the query and compute its similarity2 with the movie vectors
        query_vector = cv.transform([query]).toarray()  # Use transform, not fit_transform
        # Use the vectorized titles created earlier to ensure consistent dimensions
        distances = cosine_similarity(query_vector, vectorized_titles)  # Compute similarity2

    # Get the top 5 most similar movies2 (excluding the movie itself if it's in the query)
    movies2_list = sorted(list(enumerate(distances[0])), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies2_list:
        movie_id = movies2.iloc[i[0]].id
        recommended_movies2.append(movies2.iloc[i[0]].title)
        # Fetch movie poster and trailer using movie_id
    for j in recommended_movies2:
        movie_id = movies2.loc[movies2['title'] == j,'id'].values[0]
        recommended_movies2_posters.append(fetch_poster(movie_id))
        trailers.append(fetch_trailer(movie_id))

    return recommended_movies2, recommended_movies2_posters, trailers





def top_movies():
    L = random.sample(top_ratings,5)
    top_movies_posters = []
    top_movies_names = []
    trailers = []
    for i in L:
        movie_id = movies.iloc[i[0]].id
        top_movies_names.append(movies.iloc[i[0]].title)
        top_movies_posters.append(fetch_poster(movie_id))
        trailers.append(fetch_trailer(movie_id))
    return top_movies_names,top_movies_posters,trailers




if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = None

if 'top_movies' not in st.session_state:
    st.session_state['top_movies'] = None

if 'recommendations2' not in st.session_state:
    st.session_state['recommendations2'] = None

# Dropdown to select a movie
selected_movie_name = st.selectbox('Which Movie would you like to recommend?', movies['title'].values)

# Recommend Button
if st.button('Recommend'):
    names, posters, trailers = recommend(selected_movie_name)
    st.session_state['recommendations'] = (names, posters, trailers)  # Store results in session state


# Display recommendations if available
if st.session_state['recommendations']:
    st.subheader("Recommended Movies:")
    names, posters, trailers = st.session_state['recommendations']
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.text(names[idx])
            st.image(posters[idx])
            if trailers[idx] != -1:
                st.video(trailers[idx])
            else:
                st.text("Trailer not available")
# st.text()
st.divider()
# Dropdown to select a movie
selected_movie_name = st.selectbox('Recommend movies based on tags?', search)

# Recommend Button
if st.button('Recommend_movies'):
    names, posters, trailers = recommender(selected_movie_name)
    st.session_state['recommendations2'] = (names, posters, trailers)  # Store results in session state

# Display recommendations if available
if st.session_state['recommendations2']:
    st.subheader("Recommended movies2:")
    names, posters, trailers = st.session_state['recommendations2']
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.text(names[idx])
            st.image(posters[idx])
            if trailers[idx] != -1:
                st.video(trailers[idx])
            else:
                st.text("Trailer not available")
st.divider()
# Top Movies Button
if st.button('Top-Movies'):
    name, poster, trailer = top_movies()
    st.session_state['top_movies'] = (name, poster, trailer)  # Store results in session state
# Display top movies if available
if st.session_state['top_movies']:
    st.subheader("Top Movies:")
    name, poster, trailer = st.session_state['top_movies']
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.text(name[idx])
            st.image(poster[idx])
            if trailer[idx] != -1:
                st.video(trailer[idx])
            else:
                st.text("Trailer not available")