from data_cleaning import clean_data, get_user_booked
import pandas as pd

from finalProject.weighted_prediction import FinalModel


##### DEFINING DATA DIRECTORIES
BOOKINGS_CSV = '../data/bookings.csv'
HOTELS_CSV = '../data/hotels.csv'
HOTEL_BOOKINGS_CSV = '../data/hotel_bookings.csv'

if __name__ == "__main__":
    print("Running Predictions")
    SAVE_INTERMEDIATES = False

    print("Importing data")
    bookings = pd.read_csv(BOOKINGS_CSV)
    hotels = pd.read_csv(HOTELS_CSV)
    hotel_bookings = pd.read_csv(HOTEL_BOOKINGS_CSV)

    print("Cleaning Data, and generating intermediate cleaned data")
    df = clean_data(bookings,hotels , hotel_bookings)
    user_watched = get_user_booked(df)

    if SAVE_INTERMEDIATES:
        print("Saving data to intermediates folder")
        df.to_csv("intermediates/df.csv", encoding = "utf-8-sig")
        user_watched.to_csv("intermediates/user_watched.csv", encoding = "utf-8-sig")

    uid = 2976
    weights = (80,10,10)

    final = FinalModel(hotels, user_watched, df)
    print("Training model")
    final.train()
    print(f"Getting prediction for user: {uid}, with weights: {weights} ")
    print(final.predict(2976, (80, 10, 10)))

