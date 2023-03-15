import pandas as pd
import numpy as np
from datetime import datetime
import os

from util import __convert_int__


def clean_data(bookings: pd.DataFrame, hotels: pd.DataFrame, hotel_bookings: pd.DataFrame) -> pd.DataFrame:
    """
        Cleans and processes raw data from the client.

        bookings: Bookings events in terms of the provider and user
        hotels: Metadata regarding hotels
        hotel_bookings: Booking events in terms of the user and Hotel

        Returns: A analysis ready datarame with some added convinience columns
    """
    assert isinstance(bookings, pd.DataFrame), "Bookings_path must be a pd.DataFrame"
    assert isinstance(hotels, pd.DataFrame), "Hotels_path must be a pd.DataFrame"
    assert isinstance(hotel_bookings, pd.DataFrame), "Hotel_Bookings_path must be a pd.DataFrame"

    bookings = bookings.filter(['BookingID', 'UserID', 'BookDate', 'ProviderID'], axis=1)
    hotels = hotels.filter(['HotelID', 'HotelType', 'HotelRank'], axis=1)
    hotel_bookings = hotel_bookings.filter(['BookingID', 'ProviderID', 'HotelID', 'BookDate'], axis=1)

    #Merge: hotel_bookings and booking both have information regarding a single booking event
    df1 = pd.merge(bookings, hotel_bookings, on='BookingID').rename(columns={"ProviderID_x": "ProviderID"}).rename(columns={"BookDate_x": "BookDate"}).filter(['BookingID', 'UserID', 'ProviderID', 'HotelID', 'BookDate'], axis=1)
    #Add in data about the hotel being booked to each book event
    df2 = pd.merge(df1, hotels, on='HotelID')

    # Convert the 'time' column to a datetime object
    df2['BookDate'] = pd.to_datetime(df2['BookDate'])

    # Calculate the number of days between the 'time' column and the current date
    df2['Days_Since_Booked'] = (datetime.now() - df2['BookDate']).apply(lambda x: x.days)

    # Extract the month from the 'time' column
    df2['Month'] = df2['BookDate'].apply(lambda x: x.month)

    df = df2

    for col in set(df.columns) - {'BookDate'}:
        df[col] = df[col].apply(__convert_int__)

    df.drop_duplicates(subset=df.columns.difference(['BookDate']), inplace=True)
    df["BookDate"] = df["BookDate"].astype(str)
    return df


def get_user_booked(df: pd.DataFrame) -> pd.DataFrame:
    user_booked = df.filter(['BookingID', 'UserID', 'HotelID'], axis=1)

    # drop the ID column, then count how many times a user has booked each hotel
    user_booked = user_booked.drop(columns=['BookingID']).groupby(['UserID', 'HotelID']).size().reset_index(name='Number_Booked')

    for col in user_booked:
        user_booked[col] = user_booked[col].apply(__convert_int__)

    #Add 1, so if one time Book is then one
    user_booked['Number_Booked_log'] = user_booked['Number_Booked'].apply(np.log) + 1

    users_counts = user_booked['UserID'].value_counts().rename('users_counts')
    users_data = user_booked.merge(users_counts.to_frame(),
                                    left_on='UserID',
                                    right_index=True)

    subset_df = users_data[
        (users_data.users_counts >= 3) & (users_data.Number_Booked >= 5)]

    df2 = subset_df['HotelID'].value_counts().rename('df2')
    df2_data = subset_df.merge(df2.to_frame(),
                               left_on='HotelID',
                               right_index=True)

    df2_data = df2_data[df2_data.df2 >= 5]

    user_booked = df2_data.drop(['df2', 'users_counts'], axis=1)

    return user_booked
