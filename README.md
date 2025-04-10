# 🎬 Movie Recommendation System

This project is a content-based and user-based movie recommendation system built using TMDB's dataset. It intelligently suggests top movie recommendations based on user preferences or a selected movie using a combination of data preprocessing, user simulation, and API integration.

## Project Flow

** API Integration: **
Api_key.env is used to securely store and access your TMDB API key.
Data Collection & Cleaning:
Data from TMDB is collected, cleaned, and preprocessed to enhance recommendation accuracy.
User Simulation:
Randomly generated 5,000 users to simulate a user base and their movie preferences.
Recommendation System Logic:
test.py handles the core logic:
Existing User: Input a user_id, and the system recommends 10 movies based on their past liked movies.
New User: Input one liked movie, and the system recommends 10 similar movies.
Frontend:
HTML pages:
index.html: Landing page
newuser.html: New user recommendation input
existinguser.html: Existing user recommendation input
style.css: Custom styles for the interface.
