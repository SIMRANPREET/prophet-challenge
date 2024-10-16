# Module 8 Challenge

## Mercado Libre Analysis

This notebook analyses the Mercado Libre search trend data and the stock price data.

### Requirements

#### Find unusual patterns in hourly Google search traffic

* Read the search data into a DataFrame

``` python
df_mercado_trends = pd.read_csv(
    "https://static.bc-edx.com/ai/ail-v-1-0/m8/lms/datasets/google_hourly_search_trends.csv",
    index_col='Date',
    parse_dates=True
).dropna()
```

* Calculate the total search traffic for the month

``` python
df_mercado_trends_may = df_mercado_trends.loc["2020-05"]
```

* Compare the value to the monthly median across all months

``` python
traffic_may_2020 = df_mercado_trends_may.sum()
median_monthly_traffic = df_mercado_trends.groupby([df_mercado_trends.index.year,df_mercado_trends.index.month]).agg({"Search Trends":["sum","median"]})
median_monthly_traffic.rename_axis(["Year","Month"],inplace=True)
median_monthly_traffic.rename(columns={"sum":"Sum","median":"Median"},inplace=True)
median_monthly_traffic_value = median_monthly_traffic["Search Trends"]["Median"].median()
traffic_may_2020/median_monthly_traffic_value
```

* Did the Google search traffic increase during the month that MercadoLibre released its financial results? Write your answer in the space provided in the starter file

``` python
# Compared to the median search traffic of all the months median search traffic, the total searches in May 2020 was 749 times greater than the median search traffic across all months.
```

#### Mine the search traffic data for seasonality

* Group the hourly search data to plot the average traffic by the hour of day 

``` python
hourly = df_mercado_trends.groupby(df_mercado_trends.index.hour).agg(TotalSearchTrends = ("Search Trends", "mean"))
```

* Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday)

``` python
dayofweek = df_mercado_trends.groupby(df_mercado_trends.index.isocalendar().day).agg(TotalSearchTrends = ("Search Trends", "mean"))
```

* Group the hourly search data to plot the average traffic by the week of the year

``` python
weekofyear = df_mercado_trends.groupby(df_mercado_trends.index.isocalendar().week).agg(TotalSearchTrends = ("Search Trends", "mean"))
```

* Are there any time based trends that you can see in the data? Write your answer in the space provided in the starter file

``` python
# Based on the hour of day, it starts off high and then declines to near zero during the start of the work day and then goes back up when work hours are ending and peaks at night. 
# Based on the day of week, there is more traffic earlier in the week and then it dies down when it gets to the weekend. 
# Based on the week of the year, there is more traffic at the beginning of the year than at the end of the year.
```

#### Relate the search traffic to stock price patterns

* Read in and plot the stock price data

``` python
df_mercado_stock = pd.read_csv(
    "https://static.bc-edx.com/ai/ail-v-1-0/m8/lms/datasets/mercado_stock_price.csv",
    index_col="date",
    parse_dates=True
).dropna()
df_mercado_stock.plot()
```

* Concatenate the stock price data to the search data in a single DataFrame

``` python
mercado_stock_trends_df = df_mercado_trends.join(df_mercado_stock).dropna()
```

* Slice the data to just the first half of 2020 (2020-01 to 2020-06 in the DataFrame), and then plot the data

``` python
first_half_2020 = mercado_stock_trends_df.loc["2020-01":"2020-06"]
first_half_2020.plot(subplots=True)
```

* Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour

``` python
mercado_stock_trends_df["Lagged Search Trends"] = mercado_stock_trends_df["Search Trends"].shift(1)
```

* Create two additional columns
    * “Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility

    ``` python
    mercado_stock_trends_df["Stock Volatility"] = mercado_stock_trends_df["close"].rolling(4).std()
    ```

    * “Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis

    ``` python
    mercado_stock_trends_df["Hourly Stock Return"] = mercado_stock_trends_df["close"].pct_change(1)
    ```

* Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns? Write your answer in the space provided in the starter file

``` python
# The correlation table shows that there is a weak negative correlation between the lagged search traffic and the stock volatility, I would say it is not significant enough to reliably predict a correlation between those two. There is an even weaker correlation between lagged search traffic and the stock price.
```

#### Create a time series model with Prophet

* Set up the Google search data for a Prophet forecasting model

``` python
mercado_prophet_df = df_mercado_trends.copy()
mercado_prophet_df.reset_index(inplace=True)
mercado_prophet_df.rename(columns={"Date":"ds","Search Trends":"y"},inplace = True)
mercado_prophet_df.dropna()
```

* After estimating the model, plot the forecast

``` python
m = Prophet()
m.fit(mercado_prophet_df)
future_mercado_trends = m.make_future_dataframe(periods=2000, freq="H")
forecast_mercado_trends = m.predict(future_mercado_trends)
m.plot(forecast_mercado_trends)
```

* Plot the individual time series components of the model

``` python
forecast_mercado_trends.reset_index(inplace=True)
m.plot_components(forecast_mercado_trends)
```

* Answer the following questions in the space provided in the starter file

``` python
# Based on the "Hour of Day" plot, the most popularity is exhibited near midnight.
# Based on the "Day of Week" plot, Tuesday exhibits the most search traffic.
# Based on the "Day of Year" plot, the lowest point is exhibited mid October.
```
