#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    # df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    # df['duration'] = df.duration.dt.total_seconds() / 60

    # df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

## splitting read_data() into two parts (I/O and data transform) for easier testing
def prepare_data(df: pd.DataFrame):
    """
    Intakes a df and applies transformations to it
    """
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

## needs to be put into a fn called main() that takes year, and month as params
def main(year: int, month: int):

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    # loading data
    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # loading model and dictionary vectorizer
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

if __name__ == "__main__":
    
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    
    main(year=year, month=month)
