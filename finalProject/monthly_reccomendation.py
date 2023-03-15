import pandas as pd

def movie_gt_booking(df, booking=250):
    '''return a list of movieID with at least
    booking + 1 in the df'''
    assert isinstance(df, pd.DataFrame)
    assert isinstance(booking, int) and booking > 3
    month_map = dict()
    for row in df[['MovieID', 'WatchDate']].itertuples():
        ID = row[1]
        date = row[2]
        assert isinstance(ID, int)
        assert isinstance(date, str) and len(date) == 19
        dateInfo = date.split(' ')[0].split('-')
        month = int(dateInfo[1])

        if ID not in month_map:
            month_map[ID] = [0] * 12
        month_map[ID][month-1] += 1

    movies = []
    for key in month_map:
        total_booking = sum(month_map[key])
        if total_booking > booking:
            movies.append(key)
    return movies

def movie_le_booking(df, booking=250):
    '''return a list of movieID with at most
    booking in the df'''
    assert isinstance(df, pd.DataFrame)
    assert isinstance(booking, int) and booking > 3
    month_map = dict()
    for row in df[['MovieID', 'WatchDate']].itertuples():
        ID = row[1]
        date = row[2]
        assert isinstance(ID, int)
        assert isinstance(date, str) and len(date) == 19
        dateInfo = date.split(' ')[0].split('-')
        month = int(dateInfo[1])

        if ID not in month_map:
            month_map[ID] = [0] * 12
        month_map[ID][month-1] += 1

    movies = []
    for key in month_map:
        total_booking = sum(month_map[key])
        if total_booking <= booking:
            movies.append(key)
    return movies
def booking_maps(df):
    '''Input: df with column movieId and watchData
    return a map from movieID to a list of booking sum
    of each month.'''
    assert isinstance(df, pd.DataFrame)
    month_map = dict()

    for row in df[['MovieID', 'WatchDate']].itertuples():
        ID = row[1]
        date = row[2]
        assert isinstance(ID, int)
        assert isinstance(date, str) and len(date) == 19
        dateInfo = date.split(' ')[0].split('-')
        yr = int(dateInfo[0])
        month = int(dateInfo[1])

        if ID not in month_map:
            month_map[ID] = [0] * 12
        month_map[ID][month-1] += 1
    return month_map


def movie_sd_mapping(month_map, movie_list, month, weightEqual=True, sd_diff=2.0):
    '''given month_map, movie_list, return
    mapping from movieID to standard deviation based weights '''
    assert isinstance(month_map, dict)
    assert isinstance(movie_list, list)
    assert isinstance(month, int) and month >= 1 and month <= 12
    assert isinstance(sd_diff, float)
    for s in movie_list:
        assert isinstance(s, int)
    monthIndex = month - 1

    mapping = {}
    maxSdDiff = 0
    for movie in movie_list:
        book_history = month_map[movie]
        mean = sum(book_history) / len(book_history)
        variance = sum([((x - mean) ** 2) for x in book_history]) / len(book_history)
        sd = variance ** 0.5
        n_sd_diff = abs(book_history[monthIndex] - mean) / sd
        if n_sd_diff > sd_diff:
            mapping[movie] = n_sd_diff
            if maxSdDiff < n_sd_diff:
                maxSdDiff = n_sd_diff

    multiplicand = 100.0 / maxSdDiff if maxSdDiff != 0 else 100.0
    for movie in mapping:
        if weightEqual:
            mapping[movie] = 100
        else:
            mapping[movie] = int(mapping[movie] * multiplicand)
    return mapping
def monthlyTrendyWrapper(df, month,  bookings=250, sd_diff=2.0, weightEqual=False):
    '''Input: a dataframe with booking data,
    bookings: to filter out hotel with at least this number of bookings,
    month: current month
    sd_diff: to filter out hotel with at least sd_diff number of standard deviation from mean,
    weightEqual: output ranking all 100 if True. Otherwise, ranking is weighted upon sd_diff
    Return a dataframe that has columns movieID and ranking.
    MovieID: unique id for that movie
    Ranking: weighted 1-100, always 100 if weightEqual is True.'''
    df2 = df[['MovieID', 'WatchDate']]
    # assert statements are included in the following functions
    month_map = booking_maps(df2)
    movie_list = movie_gt_booking(df2, bookings)
    mapping = movie_sd_mapping(month_map, movie_list, month, weightEqual, sd_diff)
    lst = sorted(mapping.items(), key=lambda x:x[1], reverse=True)
    movieID = []
    movieRanking = []
    for tup in lst:
        movieID.append(tup[0])
        movieRanking.append(tup[1])
    data = {'movieID': movieID,
            'ranking': movieRanking}
    ret = pd.DataFrame(data, columns = ["MovieID", "ranking"])
    return ret

if __name__ == "__main__":
    from datetime import datetime as dt
    df = pd.read_csv("intermediates/df.csv")
    ret = monthlyTrendyWrapper(df, dt.now().month, bookings=250, sd_diff=1.5, weightEqual=False)
    print(ret)
    print(type(ret))