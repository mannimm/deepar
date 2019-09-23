import pandas as pd
air = pd.read_csv("AirPassengers.csv")['#Passengers'].values
source_df = pd.DataFrame({'feature_1': air[:-1], 'target': air[1:]})
source_df['category'] = ['1' for i in range(source_df.shape[0])]


from deepar.dataset.time_series import TimeSeries
from deepar.model.lstm import DeepAR
from sklearn.preprocessing import MinMaxScaler
ts = TimeSeries(source_df, scaler=MinMaxScaler)
dp_model = DeepAR(ts, epochs=100)
dp_model.instantiate_and_fit()