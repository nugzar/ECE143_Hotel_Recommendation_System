from typing import Callable, Any

import pandas as pd
from surprise import SVD, SVDpp, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV


def trainSVDPP(user_booked: pd.DataFrame) -> Callable[[Any], pd.DataFrame]:
    """
        Trains a svd++ model, given the user_booked dataframe.
        Returns a prediction function that takes in a user function and yields prediction outputs.
    """
    assert isinstance(user_booked, pd.DataFrame), "Inputted data is not a pd.DataFrame"
    COLS = ['UserID', 'HotelID', 'Number_Booked_log']
    assert all(
        (col in set(user_booked.columns) for col in COLS)), f"Necesary columns not found. Necessary columns: {COLS}"

    reader = Reader(rating_scale=(0, 10))
    surprise_data = Dataset.load_from_df(user_booked[['UserID', 'HotelID', 'Number_Booked_log']], reader)
    trainset, testset = train_test_split(surprise_data, test_size=.30, random_state=7)
    svd_param_grid = {'n_epochs': [10, 15, 20, 25, 30], 'lr_all': [0.003, 0.005, 0.007],
                      'reg_all': [0.0025, 0.005, 0.01]}

    svdpp_gs = GridSearchCV(SVDpp, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
    svdpp_gs.fit(surprise_data)

    svd = SVD(n_epochs=20, lr_all=0.005, reg_all=0.005)
    reader = Reader()
    data = user_booked
    data = data.rename(columns={"Number_Booked_log": "rating"})

    data = Dataset.load_from_df(data[['UserID', 'HotelID', 'rating']], reader)
    # Run 5-fold cross-validation and print the results
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)

    # sample full trainset
    trainset = data.build_full_trainset()

    # Train the algorithm on the trainset
    svd.fit(trainset)

    # Get all the possible Hotels that were used
    Hotels = user_booked['HotelID'].unique()

    def get_svd_predictions(uid):
        # Convert the Hotels array to a list
        HotelList = Hotels.tolist()

        predictions = [svd.predict(uid=uid, iid=Hotel, r_ui=None)[3] for Hotel in Hotels]

        # Adjust prediction values
        min_pred = min(predictions)
        max_pred = max(predictions)
        if min_pred == max_pred:
            # if all predictions are the same, return 50 for each prediction
            predictions = [50] * len(predictions)
        else:
            predictions = [round(1 + 99 * (p - min_pred) / (max_pred - min_pred)) for p in predictions]

        # Create a new DataFrame with the Hotel IDs and predictions
        df = pd.DataFrame({'HotelID': list(Hotels), 'Prediction': predictions})
        # Sort the DataFrame by the Prediction column in descending order
        df = df.sort_values(by='Prediction', ascending=False).reset_index(drop=True)

        return df

    return get_svd_predictions


if __name__ == "__main__":
    # Example usage, using a intermediate user-booked dataframe and the user 2976
    user_id = 2976
    print(trainSVDPP(pd.read_csv("intermediates/user_booked.csv"))(user_id))
