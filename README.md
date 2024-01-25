# Movie Recommendation System

## Overview

This project is a movie recommendation system built on content-based filtering. It suggests 5 movies to the user based on their preferences and the content features of the movies.

## Features

- **Content-Based Filtering:** The recommendation system analyzes the content features of movies, such as genre, actors, and keywords, to provide personalized suggestions.

- **Streamlit Interface:** The user interacts with the recommendation system through a Streamlit web interface, making it user-friendly and accessible.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- Streamlit

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/movie-recommendation-system.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd movie-recommendation-system
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

2. **Open the provided URL in your web browser.**

3. **Input your preferences or select a movie to get personalized recommendations.**

## Project Structure

- `app.py`: The main Streamlit application script.
- `movie-recommendation-system.ipynb`: The Jupyter Notebook provides the dataset's analysis.
- `movies.pkl`: The pickle file for a dataframe used in streamlit_app.py.
  
## Data

The recommendation system uses a tmdb 5000 movies dataset. Ensure that the dataset is up-to-date and relevant to achieve accurate recommendations. If you need sample data, consider using [tmdb dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

## Model Training

If you want to retrain the recommendation model or update the features, check the `movie-recommendation-system.ipynb` file for any model training scripts or feature extraction processes.

## Customization

Feel free to customize the recommendation algorithm or interface based on your specific requirements. Explore the codebase and modify as needed.

## Future Enhancements

- **Collaborative Filtering:** Consider implementing collaborative filtering techniques to enhance recommendation accuracy.

- **User Profiles:** Allow users to create profiles to improve personalization.

- **Deployment:** Explore options for deploying the recommendation system, such as cloud platforms or containerization.
