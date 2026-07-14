import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv(r"C:\Users\velku\Downloads\New folder\DS\project\project_5\imdb_movies_2024.csv")
movies = movies[['Movie Name', 'Cleaned_Storyline',"Storyline"]]

# Train vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
movie_vectors = vectorizer.fit_transform(movies['Cleaned_Storyline'])

st.title("Movie Recommendation System")

user_story = st.text_area("Enter a movie storyline")

if st.button("Recommend"):
    user_vector = vectorizer.transform([user_story])

    similarity = cosine_similarity(user_vector, movie_vectors)

    top_movies = np.argsort(similarity[0])[::-1][:5]

    st.subheader("Top 5 Recommended Movies")

    for index in top_movies:
        st.write("🎬", movies.iloc[index]['Movie Name'])
        st.write("📖", movies.iloc[index]['Storyline'])
