from typing import Union

import pandas as pd
import math
import datetime
from scipy.stats import norm


DATA_DIR = '../data/combine_df.csv'

def get_history_orders(df: pd.DataFrame, hotelID: int, timespan: int = 3, max_days: Union[int, float] = math.inf):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(hotelID, int)
    assert isinstance(timespan, int)
    """
    Query history orders of some hotel.
    count the orders that fall into each timespan days
    (e.g., timespan = 7, we count the orders in each week)
    """
    data = df[df['HotelID'] == hotelID]
    assert data.size != 0
    data = data['BookDate'].values.tolist()
    data = [date_type_converter(i) for i in data]
    deltaTime = datetime.timedelta(days=timespan)
    start_day = data[0]
    end_day = start_day + deltaTime
    num_orders_history = []

    num_orders = 0
    for d in data:
        if len(num_orders_history) * timespan > max_days:
            break
        if d > end_day:
            num_orders_history.append(num_orders)
            num_orders = 0
            start_day = end_day
            end_day = start_day + deltaTime
        num_orders += 1
    return num_orders_history


def guassian_estimatation(data):
    """ Fit a gaussian model for given data """
    mu, std = norm.fit_loc_scale(data)
    return mu, std


def date_type_converter(mydatetime: str):
    """
        Convert datatype of strings to date
    """
    assert isinstance(mydatetime, str), "Inputted datetime is not a string"
    return datetime.datetime.strptime(mydatetime, "%Y-%m-%d %H:%M:%S").date()


def output_score(df, timespan = 30, max_days=math.inf):
    hotelIDs = list(set(df['HotelID'].values.tolist()))
    hotelIDs.sort()
    scores = []
    for hotel in hotelIDs:
        data = get_history_orders(df, hotel, timespan, max_days)
        if len(data) == 0:
            scores.append(0)
            continue
        mu, std = norm.fit_loc_scale(data)
        scores.append((data[-1] - mu) / std)

    maxScore = max(scores)
    minScore = min(scores)
    assert maxScore > minScore
    scores = [(i - minScore) / (maxScore - minScore) * 100 for i in scores]
    return pd.DataFrame({'HotelID': hotelIDs, 'score': scores})

if __name__ == '__main__':
    # read data
    df = pd.read_csv(DATA_DIR)
    print("The size of the data is:" + str(df.size))

    hotelID = 88879
    timespan = 7
    std_coef = 1.5
    print(output_score(df))
