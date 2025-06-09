import pandas as pd
import json

import pendulum

from airflow.sdk import dag, task

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
# @dag(
#     schedule=None,
#     start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
#     catchup=False,
#     tags=["example"],
# )
# def tutorial_taskflow_api():
#     """
#     ### TaskFlow API Tutorial Documentation
#     This is a simple data pipeline example which demonstrates the use of
#     the TaskFlow API using three simple tasks for Extract, Transform, and Load.
#     Documentation that goes along with the Airflow TaskFlow API tutorial is
#     located
#     [here](https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html)
#     """
#     @task()
#     def extract():
#         """
#         #### Extract task
#         A simple Extract task to get data ready for the rest of the data
#         pipeline. In this case, getting data is simulated by reading from a
#         hardcoded JSON string.
#         """
#         data_string = '{"1001": 301.27, "1002": 433.21, "1003": 502.22}'

#         order_data_dict = json.loads(data_string)
#         return order_data_dict
#     @task(multiple_outputs=True)
#     def transform(order_data_dict: dict):
#         """
#         #### Transform task
#         A simple Transform task which takes in the collection of order data and
#         computes the total order value.
#         """
#         total_order_value = 0

#         for value in order_data_dict.values():
#             total_order_value += value

#         return {"total_order_value": total_order_value}
#     @task()
#     def load(total_order_value: float):
#         """
#         #### Load task
#         A simple Load task which takes in the result of the Transform task and
#         instead of saving it to end user review, just prints it out.
#         """

#         print(f"Total order value is: {total_order_value:.2f}")
#     order_data = extract()
#     order_summary = transform(order_data)
#     load(order_summary["total_order_value"])
# tutorial_taskflow_api()
