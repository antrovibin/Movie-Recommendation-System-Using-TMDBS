# 🎬 Movie Recommendation System

This project is a content-based and user-based movie recommendation system built using TMDB's dataset. It intelligently suggests top movie recommendations based on user preferences or a selected movie using a combination of data preprocessing, user simulation, and API integration.

## Project Flow

** API Integration: **
The Movie Recommendation System uses Api_key.env to securely store and access your TMDB API key. The system collects, cleans, and preprocesses data from TMDB to enhance the accuracy of recommendations. It simulates a user base by generating 5,000 random users and their movie preferences. The core logic for the recommendation system is handled in test.py: for existing users, the system recommends 10 movies based on their past liked movies when a user_id is input, while for new users, it recommends 10 similar movies based on one liked movie. The frontend is built with HTML pages, including index.html for the landing page, newuser.html for new user recommendation input, and existinguser.html for existing user recommendation input, with custom styles provided in style.css to enhance the interface's appearance.

## Tech Stack
* Python
* Pandas / NumPy / Scikit-learn
* HTML / CSS
* TMDB API

## Usage

Run the main script:
```bash
python testb.py
```
Navigate through the HTML pages to test with new or existing users.
* For existing users, input a user_id.
* For new users, input a movie title you like.
The system will return top 10 recommendations accordingly.

## Project Structure
```bash
├── Api_key.env
├── test.py
├── datasets   # Processed datasets
├── templates/
│   ├── index.html
│   ├── newuser.html
│   └── existinguser.html
├── static/
│   └── style.css
```

## License

This project is open-source and available under the MIT License.
