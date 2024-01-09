import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Assuming you have CSV files named 'movies.csv' and 'credits.csv'
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies.head()

movies_credits = pd.merge(movies, credits, on='title')
movies_credits['genres'] = movies_credits['genres'].fillna('[]').apply(eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movies_credits['keywords'] = movies_credits['keywords'].fillna('[]').apply(eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movies_credits['content'] = movies_credits.apply(lambda row: row['genres'] + row['keywords'] + [row['overview']] if pd.notna(row['overview']) else row['genres'] + row['keywords'], axis=1)

if 'content' not in movies_credits.columns or movies_credits['content'].isnull().all():
    print("Error: 'content' column is missing or empty.")
else:
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Combine 'genres', 'keywords', and 'overview' into 'content'
    movies_credits['content'] = movies_credits.apply(
        lambda row: ' '.join(row['genres']) + ' ' + ' '.join(row['keywords']) + ' ' + str(row['overview']),
        axis=1
    )

    # Fit and transform the TF-IDF vectorizer
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_credits['content'])

    # Compute the similarity matrix
    similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


    # Function to recommend movies based on age group
    def recommend_movies(age_group):
        # Define age group categories
        age_categories = {
            "Under 18": ["Animation", "Family","Cartoon","Anime"],
            "18-50": ["Action", "Adventure","Science","Comedy","Drama","Horror","Thriller"],
            "Above 50": ["Documentary","Old","1980","Politics"]
        }

        # Get movie genres for the specified age group
        target_genres = age_categories.get(age_group, [])

        # Filter movies that match the specified genres
        recommended_movies = []
        for genre in target_genres:
            genre_movies = movies_credits[movies_credits['genres'].apply(lambda x: genre in x)]
            recommended_movies.extend(genre_movies.sample(min(5, len(genre_movies))))

        return recommended_movies
