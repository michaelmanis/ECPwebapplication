import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_option('deprecation.showPyplotGlobalUse', False)

#page title
st.set_page_config(page_title='Energy Consumption Prediction Application',
                   layout='wide',
                   page_icon="⚡")
def model(df):
    df = pd.read_csv('energydatanew.csv', parse_dates=['date'])
    df.index = pd.to_datetime(df['date'])

    #Create temporal train-test split based on Appliances
    def temporal_train_test_split(test_start_ind):
        train_set = df[:test_start_ind]['Appliances']
        test_set = df[test_start_ind:]['Appliances']
        return train_set, test_set


    test_start_ind = 1100

    train_set, test_set = temporal_train_test_split(test_start_ind)


    def convert_to_series(forecasts, ind):
        return pd.Series(forecasts, index=test_set.index[ind:ind + 120])


    def plot_forecasts(forecasts):
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        train_set.plot(ax=ax, label='training set')
        test_set.plot(ax=ax, label='test set')
        forecasts.plot(ax=ax, label='forecast')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('year')
        ax.set_ylabel('appliances')
        ax.legend()


    def naive1(dset):
        return [dset[-1]] * 120


    def plot_multiple_forecasts(preds, ax=None):
        if not ax:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()

        train_set.plot(ax=ax)
        test_set.plot(ax=ax)

        for i, f in enumerate(preds):
            f.plot(ax=ax, c='C2', alpha=(i + 1) / len(preds) * 4)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('year')
        ax.set_ylabel('appliances')
        return ax


    naive1_forecasts = []

    for i, ind in enumerate(range(test_start_ind, len(df) - 120)):
        train, _ = temporal_train_test_split(ind)
        naive1_forecasts += [convert_to_series(naive1(train), i)]


    def naive2(dset, lookback=120):
        return [dset[-lookback:].mean()] * 120


    naive_forecasts = {'naive1': naive1_forecasts}

    for lb in (30, 60, 120, 240):

        k = 'naive2_' + str(lb)
        naive_forecasts[k] = []

        for i, ind in enumerate(range(test_start_ind, len(df) - 120)):
            train, _ = temporal_train_test_split(ind)
            naive_forecasts[k] += [convert_to_series(naive2(train, lb), i)]


    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))


    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


    train, test = temporal_train_test_split(test_start_ind)

    for k, v in naive_forecasts.items():
        print('{:<100}  MAE = {:.1f}  MAPE = {:.3f}'.format(
            k, mae(test[:120], v[0]), mape(test[:120], v[0])))


    def unfold_test_set(test_set):
        return np.array([test_set[i:i + 120] for i in range(len(test_set) - 120)])


    unfolded_test_set = unfold_test_set(test_set)

    for k, v in naive_forecasts.items():
        print('{:<10}  MAE = {:.1f}  MAPE = {:.1f}%'.format(
            k, mae(unfolded_test_set, np.array(v)), mape(unfolded_test_set, np.array(v))))


    def forecast_bias(y_true, y_pred):
        return np.mean(y_true - y_pred)


    forecast_bias(unfolded_test_set, np.array(naive_forecasts['naive2_120']))


    def plot_preds_vs_targets(y_true, y_pred):

        plt.figure(figsize=(8, 8))

        line = np.linspace(y_true.min(), y_true.max(), 100)


    def build_dset(dset, lookback, horizon):
        data = pd.concat([dset.shift(-i)
                          for i in range(lookback + horizon)], axis=1).dropna()
        data.columns = range(-lookback, horizon)
        data.index = dset.index[-len(data):]
        return data.iloc[:, :lookback], data.iloc[:, lookback:]


    x, y = build_dset(df['Appliances'], 240, 120)

    train_end_date = '2018-01-01'
    test_start_date = '2019-01-01'

    x_train = x[:train_end_date]
    x_test = x[test_start_date:]

    y_train = y[:train_end_date]
    y_test = y[test_start_date:]

    lr = LinearRegression()

    lr.fit(x_train, y_train)

    preds = lr.predict(x_test)


    def convert_to_series_2(predictions, ind):
        df_ind = np.where(df.index == test_start_date)[0][0] + ind
        return pd.Series(predictions, index=df.index[df_ind - 110:df_ind + 10])


    i = 0
    plot_forecasts(convert_to_series_2(preds[i], i))

#---------------------------------#
# Page layout
# Page expands to full width
#---------------------------------#
st.write("""
# Energy Consumption Prediction Application
In this implementation, the *LinearRegression()* function is used in this app for build a regression model using the **Linear Regression** algorithm.
"""
         )

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file", type=["csv"])


# Sidebar - Specify parameter settings
with st.sidebar.header('Set Parameters'):
    split_size = st.sidebar.slider(
        'Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider(
        'Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider(
        'Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider(
        'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider(
        'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)


#---------------------------------#
# Main panel
# Displays the dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    model(df)


else:
    st.info('Awaiting for CSV file to be uploaded.')


st.header('Prediction Table')
plt.title('Comparison of Prediction with Actual Energy Consumption')





st.pyplot()
st.markdown('** Glimpse of dataset**')
st.write(df)
