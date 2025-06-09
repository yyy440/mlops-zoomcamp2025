import pandas as pd 
import os 

import pickle
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# load the data
def read_dataframe(filename):
    df = pd.read_parquet(filename)
    print(f"There are {len(df)} records.")
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df, categorical

def main():
    
    file_path = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    df, categorical = read_dataframe(file_path)
    print(f"There are {len(df)} records after removing outliers.")
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    target = 'duration'
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(f"The model intercept is {lr.intercept_}")

    p = pickle.dumps(lr)
    print(sys.getsizeof(p))

if __name__ == "__main__":

    main()