import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import re

# Load Data
movies = pd.read_csv("/Users/processed_movie_data.csv")
users = pd.read_csv("/Users/data.csv")

# Process user data
users["liked_movies"] = users["liked_movies"].apply(lambda x: eval(x) if isinstance(x, str) else x)
users["disliked_movies"] = users["disliked_movies"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Create user-movie interaction matrix
user_movie_interaction = pd.DataFrame(0, index=users['user_id'], columns=movies['movie_id'])
for i, row in users.iterrows():
    for movie_id in row['liked_movies']:
        user_movie_interaction.loc[row['user_id'], movie_id] = 1

# Perform SVD
svd = TruncatedSVD(n_components=50, random_state=42)
svd_matrix = svd.fit_transform(user_movie_interaction.fillna(0))
svd_reconstructed = np.dot(svd_matrix, svd.components_)

# Content-Based Recommendation
scaler = StandardScaler()
movie_features_scaled = scaler.fit_transform(movies.iloc[:, 13:33])
cosine_sim = cosine_similarity(movie_features_scaled)

def get_content_based_recommendations(movie_id, top_n=10):
    if movie_id not in movies["movie_id"].values:
        return []
    idx = movies[movies['movie_id'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['movie_id', 'title', 'genres']].to_dict(orient="records")

# Hybrid Recommendation
def hybrid_recommendation(user_id, top_n=10, top_liked_n=10):
    if user_id not in users["user_id"].values:
        return []

    user = users[users["user_id"] == user_id].iloc[0]
    liked_movies = user["liked_movies"]

    # If the user has no liked movies, recommend top 10 movies from the current year
    if not liked_movies:
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Filter movies from the current year
        top_movies = movies[movies["release_date"].str.startswith(str(current_year))]
        
        # If less than top_n movies exist, include movies from the remaining months of last year
        if len(top_movies) < top_n:
            previous_year = current_year - 1
            additional_movies = movies[
                (movies["release_date"].str.startswith(str(previous_year))) & 
                (movies["release_date"].str[5:7].astype(int) > current_month)
            ]
            top_movies = pd.concat([top_movies, additional_movies])
        
        # Sort by popularity and vote_average, and return top_n movies
        top_movies = top_movies.sort_values(by=["popularity", "vote_average"], ascending=False).head(top_n)
        return top_movies[["movie_id", "title", "genres"]].to_dict(orient='records')

    # Proceed with normal recommendation logic if the user has liked movies
    liked_movies_df = movies[movies["movie_id"].isin(liked_movies)]
    liked_movies_df = liked_movies_df.sort_values(by=["popularity", "vote_average"], ascending=False)
    top_liked_movies = liked_movies_df.head(top_liked_n)

    recommended_movies = []
    for movie_id in top_liked_movies["movie_id"]:
        recommended_movies.extend(get_content_based_recommendations(movie_id, top_n))

    recommended_movie_ids = [movie['movie_id'] for movie in recommended_movies]
    recommended_movies_df = movies[movies["movie_id"].isin(recommended_movie_ids)]
    recommendations_with_metadata = recommended_movies_df[["movie_id", "title", "popularity", "vote_average", "genres"]]

    recommendations_with_metadata['genres'] = recommendations_with_metadata['genres'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else ('No Genre' if pd.isnull(x) else x)
    )

    recommendations_with_metadata = recommendations_with_metadata.sort_values(by=["popularity", "vote_average"], ascending=False)
    return recommendations_with_metadata.head(top_n).to_dict(orient='records')

# Function to normalize movie titles by removing extra spaces and converting to lowercase
def normalize_title(title):
    # Remove extra spaces and convert to lowercase
    return re.sub(r'\s+', ' ', title.strip().lower())

# Create a dictionary of canonical movie titles and their variations
def create_movie_variation_mapping(movies):
    movie_variation_mapping = {}
    
    for _, movie in movies.iterrows():
        canonical_title = normalize_title(movie['title'])
        if canonical_title not in movie_variation_mapping:
            movie_variation_mapping[canonical_title] = movie['title']
    
    return movie_variation_mapping

# Create movie variation mapping from the movie dataset
movie_variation_mapping = create_movie_variation_mapping(movies)

# Function to get the canonical title for a movie variation
def get_canonical_movie_title(movie_input_title):
    normalized_input_title = normalize_title(movie_input_title)
    
    # Iterate over all movies and check for substring matches
    for canonical_title in movie_variation_mapping:
        if normalized_input_title in canonical_title:
            return movie_variation_mapping[canonical_title]
    
    return None  # If no match is found, return None


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/existing_user", methods=["GET", "POST"])
def existing_user():
    if request.method == "POST":
        user_id = int(request.form["user_id"])

        # Fetch user details
        user = users[users["user_id"] == user_id]
        if user.empty:
            return render_template("existing_user.html", error="Invalid User ID")
        
        user = user.iloc[0]

        # Ensure liked_movies is a list
        liked_movies = user["liked_movies"] if isinstance(user["liked_movies"], list) else eval(user["liked_movies"])

        # Ensure disliked_movies is handled properly
        disliked_movies = user.get("disliked_movies", [])
        if isinstance(disliked_movies, str):
            disliked_movies = eval(disliked_movies)

        # Extract liked genres
        liked_genres = set()
        for movie_id in liked_movies:
            movie = movies[movies["movie_id"] == movie_id]
            if not movie.empty:
                genres = movie.iloc[0]["genres"]
                if isinstance(genres, str):
                    liked_genres.update(genres.split(", "))  # Ensure genres are properly split into a list

        user_info = {
            "user_id": user_id,
            "liked_movies": len(liked_movies),
            "disliked_movies": len(disliked_movies),
            "liked_genres": ", ".join(liked_genres) if liked_genres else "No Liked Genres Found"
        }

        # Get recommendations
        recommendations = hybrid_recommendation(user_id, top_n=10)

        return render_template("existing_user.html", recommendations=recommendations, user_info=user_info)
    
    return render_template("existing_user.html")

@app.route("/new_user", methods=["GET", "POST"])
def new_user():
    if request.method == "POST":
        movie_title = request.form["movie_title"]
        language = request.form["language"]

        # Get the canonical movie title based on the user's input
        canonical_movie_title = get_canonical_movie_title(movie_title)
        
        if not canonical_movie_title:
            return render_template("new_user.html", error="No matching movie found with the specified title.")
        
        # Find the movie by the canonical title and filter by language
        filtered_movies = movies[movies['title'].str.lower() == canonical_movie_title.lower()]
        filtered_movies = filtered_movies[filtered_movies['original_language'] == language]

        if filtered_movies.empty:
            return render_template("new_user.html", error="No matching movie found with the specified title and language.")
        
        # Select the first matching movie
        selected_movie = filtered_movies.iloc[0]

        # Get recommendations for the selected movie
        recommendations = get_content_based_recommendations(selected_movie['movie_id'], top_n=10) 
        if recommendations:
            return render_template("new_user.html", recommendations=recommendations)
        else:
            return render_template("new_user.html", error="No recommendations found.")
    return render_template("new_user.html")

@app.route("/get_movie_suggestions", methods=["GET"])
def get_movie_suggestions():
    query = request.args.get('query', '').lower()  # Get the query from the request
    # Find movies that match the query (case-insensitive substring match)
    matching_movies = movies[movies['title'].str.lower().str.contains(query)]
    suggestions = matching_movies[['movie_id', 'title']].to_dict(orient='records')
    return jsonify({'suggestions': suggestions})


@app.route("/movie-details", methods=["GET"])
def movie_details():
    try:
        # Get the movie_id from the query parameter
        movie_id = int(request.args.get("movie_id"))
        # Find the movie in the movies DataFrame
        movie = movies[movies["movie_id"] == movie_id]
        if movie.empty:
            return jsonify({"message": f"No movie found with movie_id {movie_id}"}), 404
        
        # Extract the movie details
        movie_details = movie.iloc[0]
        runtime_minutes = round(movie_details["runtime"] * 600)  # Assuming the runtime is in hours, convert to minutes
        scaled_popularity = movie_details["popularity"]*1000
        popularity = round(scaled_popularity, 3)
        
        # Prepare the movie details response
        response = {
            "title": movie_details["title"],
            "overview": movie_details["overview"],
            "genres": movie_details["genres"],
            "release_date": movie_details["release_date"],
            "original_language": movie_details["original_language"],
            "runtime": runtime_minutes,
            "popularity": popularity
        }
        return render_template("movie_details.html", movie=response)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred while fetching movie details: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)



## User_id = 123 Doesn't have any liked movies.
