from data_cleaning import clean_data, get_user_watched
import pandas as pd

from finalProject.weighted_prediction import FinalModel

if __name__ == "__main__":
    print("Running Predictions")
    SAVE_INTERMEDIATES = False

    print("Importing data")
    watchings = pd.read_csv('../data/watchings.csv')
    movies = pd.read_csv('../data/movies.csv')
    movie_watchings = pd.read_csv('../data/movie_watchings.csv')

    print("Cleaning Data, and generating intermediate cleaned data")
    df = clean_data(watchings,movies , movie_watchings)
    user_watched = get_user_watched(df)

    if SAVE_INTERMEDIATES:
        print("Saving data to intermediates folder")
        df.to_csv("intermediates/df.csv", encoding = "utf-8-sig")
        user_watched.to_csv("intermediates/user_watched.csv", encoding = "utf-8-sig")

    uid = 2976
    weights = (80,10,10)

    final = FinalModel(movies, user_watched, df)
    print("Training model")
    final.train()
    print(f"Getting prediction for user: {uid}, with weights: {weights} ")
    print(final.predict(2976, (80, 10, 10)))

