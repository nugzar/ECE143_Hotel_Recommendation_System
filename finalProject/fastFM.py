# # Importing libraries


import numpy as np
import pandas as pd
import scipy

import sklearn 

import os

from fastFM import als



# # Data dictionary

# WatchID - Instance of someone booking a hotel

# UserID - User ID

# ProviderID - Which website/provider they used when booking (Maybe matters, maybe not?)

# MovieID - HotelID

# WatchDate - When the hotel was booked

# MovieType - Hotel Type (By the beach, downtown, idk)

# MovieRank - Hotel Stars (Not by the user, but the actual stars)

data_dir = './data/'



def RMSE(predictions, labels):

    """Calculates the Root Mean Square Value

    Returns:
        RMSE(Float): Root mean square value of the prediction
    """
    assert len(predictions)  == len(labels), "Prediction and labels len are not same"

    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


def get_data(data_dir):

    ''' Gets the training data for the Fast FM model
    '''

    assert isinstance(data_dir,str)

    assert os.path.exists(data_dir) , 'Path is not present'


    csv_files = os.listdir(data_dir)


    df_di = {}

    for file in csv_files:
        df_di[file] = pd.read_csv(os.path.join(data_dir,file))


    main_df = df_di['df.csv']

    main_df = main_df.sort_values('Days_Since_Watched').reset_index(drop = True)

    main_df = main_df.drop('WatchingID',axis= 1)


    main_df = main_df.drop_duplicates(['UserID','MovieID'])

    ## adding additional columns for filtering the data

    users_counts = main_df['UserID'].value_counts().rename('users_counts')
    users_data   = main_df.merge(users_counts.to_frame(),
                                    left_on='UserID',
                                    right_index=True)

    ### users with less than 3 hotel bookings are dropped
    subset_df = users_data[users_data.users_counts >= 3]

    df2 = subset_df['MovieID'].value_counts().rename('df2')
    df2_data   = subset_df.merge(df2.to_frame(),
                                    left_on='MovieID',
                                    right_index=True)

    ### hotels with less than 5 hotel bookings are dropped
    df2_data = df2_data[df2_data.df2 >= 5]


    u_watch_df = df_di['User_Watched.csv']


    ## combining the datset (user watched and main df TO include recency in terms of Days_Since_Watched column)
    club_df = pd.merge(u_watch_df,main_df,on=['UserID','MovieID'],how = 'left')

    club_df = pd.merge(u_watch_df,main_df,on=['UserID','MovieID'],how = 'left')


    return prepared_data(club_df)

def prepared_data(processed_data_frame):
    """Uses the processed data set to prepare the data for modelling

    Args:
        processed_data_frame (DataFrame): Preprocessed assembled dataset for preparation
    """

    assert isinstance(processed_data_frame,pd.core.frame.DataFrame)


    data = []

    mi = processed_data_frame['Days_Since_Watched'].min()
    ma = processed_data_frame['Days_Since_Watched'].max()

    for u_id,h_id,d_sw in zip(processed_data_frame['UserID'],processed_data_frame['MovieID'],processed_data_frame['Days_Since_Watched']):
        scaled_d_sw = int(((d_sw-mi) /(ma-mi))*10)
        
        di = {'u_id':u_id,'h_id':h_id,'day_bin': scaled_d_sw}
        
        data.append(di)


    u_ids,h_ids = {},{}

    for d in data:
        u,h,da = d['u_id'],d['h_id'],d['day_bin']
        
        if not u in u_ids: u_ids[u] = len(u_ids)
        if not h in h_ids: h_ids[h] = len(h_ids)
            
            
    nUsers,nHotels = len(u_ids),len(h_ids)

    nDays = 10 - 0 + 1 ## days for offsetting


    X = scipy.sparse.lil_matrix((len(data), nUsers + nHotels + nDays))



    for i in range(len(data)):
        user = u_ids[data[i]['u_id']]
        hotel = h_ids[data[i]['h_id']]
        days = data[i]['day_bin'] 
        
        X[i,user] = 1 # One-hot encoding of user
        X[i,nUsers + hotel] = 1 # One-hot encoding of hotels
        X[i,nUsers + nHotels + days] = 1


    y = np.array([d for d in processed_data_frame['Number_Watched_log']])


    # X.shape[0],X.shape[0]*0.7

    train_size = int(X.shape[0]*0.7)

    X_train,y_train = X[:train_size],y[:train_size]
    X_test,y_test = X[train_size:],y[train_size:]



    return X_train,y_train,X_test,y_test


def model_training(X_train, y_train):
    '''
    Function for model training

    '''

    assert len(X_train) == len(y_train), 'Input x and y are of not same len along axis 0'

    fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=5, l2_reg_w=0.1, l2_reg_V=0.5)


    fm.fit(X_train, y_train)

    return fm



def model_test_set_result(fm,X_test,y_test):
    ''' Prints and returns the rmse value on the test set
    '''

    assert len(X_test) == len(y_test),'Ensure the dimension of the X and y are same along axis 0'
    y_pred_with_features = fm.predict(X_test)



    rmse_val = RMSE(y_pred_with_features, y_test)


    print('rmse is : ',rmse_val)

    ### 20% test set
    ## 0.47192768855603473


    ### 30% test set
    # 0.4605

    return rmse_val

