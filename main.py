import pandas as pd
air = pd.read_csv("AirPassengers.csv")['#Passengers'].values
source_df = pd.DataFrame({'feature_1': air[:-1], 'target': air[1:]})
source_df['category'] = ['1' for i in range(source_df.shape[0])]



hrv = pd.read_csv("RR_train.csv")

dataset_df = pd.DataFrame()




from deepar.dataset.time_series import TimeSeries
from deepar.model.lstm import DeepAR
from sklearn.preprocessing import MinMaxScaler
ts = TimeSeries(source_df, scaler=MinMaxScaler)
dp_model = DeepAR(ts, epochs=100)
dp_model.instantiate_and_fit()





%matplotlib inline
from numpy.random import normal
import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
batch = ts.next_batch(1, 20)
def get_sample_prediction(sample, prediction_fn):
    sample = np.array(sample).reshape(1, 20, 1)
    output = prediction_fn([sample])
    samples = []
    for mu,sigma in zip(output[0].reshape(20), output[1].reshape(20)):
        samples.append(normal(loc=mu, scale=np.sqrt(sigma), size=1)[0])
    return np.array(samples)
ress = []
for i in tqdm.tqdm(range(300)):
    pred = get_sample_prediction(batch[0], dp_model.predict_theta_from_input)
    ress.append(pred)
def plot_uncertainty(ress, ground_truth, n_steps=20, figsize=(9, 6), 
                     prediction_dots=True, title='Prediction on training set'):
    
    res_df = pd.DataFrame(ress).T
    tot_res = res_df
    plt.figure(figsize=figsize)
    plt.plot(ground_truth.reshape(n_steps), linewidth=6, label='Original data')
    tot_res['mu'] = tot_res.apply(lambda x: np.mean(x), axis=1)
    tot_res['upper'] = tot_res.apply(lambda x: np.mean(x) + np.std(x), axis=1)
    tot_res['lower'] = tot_res.apply(lambda x: np.mean(x) - np.std(x), axis=1)
    tot_res['two_upper'] = tot_res.apply(lambda x: np.mean(x) + 2*np.std(x), axis=1)
    tot_res['two_lower'] = tot_res.apply(lambda x: np.mean(x) - 2*np.std(x), axis=1)
    plt.plot(tot_res.mu, linewidth=4)
    if prediction_dots:
        plt.plot(tot_res.mu, 'bo', label='Likelihood mean')
    plt.fill_between(x = tot_res.index, y1=tot_res.lower, y2=tot_res.upper, alpha=0.5)
    plt.fill_between(x = tot_res.index, y1=tot_res.two_lower, y2=tot_res.two_upper, alpha=0.5)
    plt.title(title)
    plt.legend()
    
plot_uncertainty(ress, batch[1])



from tqdm import tqdm
for i in tqdm(range(1000000000000000000)):
    l =9