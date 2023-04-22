import argparse
import pandas as pd
from etna.datasets.tsdataset import TSDataset
from etna.models import SeasonalMovingAverageModel
from etna.pipeline import Pipeline
from etna.transforms import (ChangePointsTrendTransform, DateFlagsTransform,
                             HolidayTransform, LagTransform,
                             LinearTrendTransform, MeanTransform)


def train_model(path_to_data: str):
    df = pd.read_csv(path_to_data, index_col=0)
    df = (df.assign(day=pd.to_datetime(df['day']))
          .rename(columns={'day': 'timestamp', 'item_name': 'segment', 'sales_quantity': 'target'})
          .drop(columns=['item_number'])
          .drop_duplicates()
          .fillna(0))

    tsdf = TSDataset.to_dataset(df)
    ts = TSDataset(tsdf, freq="D")

    lags_list = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 49]

    transforms = [
        ChangePointsTrendTransform(in_column="target"),
        LinearTrendTransform(in_column="target", poly_degree=2),
        LagTransform(in_column="target", lags=lags_list),
        DateFlagsTransform(
            day_number_in_week=True,
            special_days_in_week=[4, 5, 6],
        ),
        HolidayTransform(iso_code='DEU'),
        MeanTransform(in_column="target", window=30),
    ]

    HORIZON = 1

    model = SeasonalMovingAverageModel(window=14, seasonality=7)
    pipeline = Pipeline(model=model, transforms=transforms, horizon=HORIZON)
    pipeline.fit(ts)

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with given data path.")
    parser.add_argument("path_to_data", type=str, help="Path to the data file.")
    args = parser.parse_args()

    path_to_data = args.path_to_data
    trained_model = train_model(path_to_data)
    print("Model training complete.")
