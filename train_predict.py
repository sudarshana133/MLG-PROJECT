import yfinance as yf
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import time

# Cache to store previously fetched data
data_cache = {}

def fetch_and_prepare_data(company_name, max_retries=3):
    if company_name in data_cache:
        return data_cache[company_name]

    start_date = "2023-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    for attempt in range(max_retries):
        try:
            stock_data = yf.download(company_name, start=start_date, end=end_date)
            if stock_data.empty:
                return None, None, None, None, None

            stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
            stock_data['5-Day MA'] = stock_data['Adj Close'].rolling(window=5).mean()
            stock_data['10-Day MA'] = stock_data['Adj Close'].rolling(window=10).mean()
            stock_data.dropna(inplace=True)

            X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Return', '5-Day MA', '10-Day MA']].values
            y = stock_data['Adj Close'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            imputer = SimpleImputer(strategy='mean')
            X_train_imputed = imputer.fit_transform(X_train_scaled)
            X_test_imputed = imputer.transform(X_test_scaled)

            data_cache[company_name] = (X_train_imputed, X_test_imputed, y_train, y_test, stock_data)
            return X_train_imputed, X_test_imputed, y_train, y_test, stock_data

        except Exception as e:
            print(f"Error fetching data for {company_name} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                return None, None, None, None, None

def train_and_predict(X_train_imputed, X_test_imputed, y_train):
    model = RandomForestRegressor()
    model.fit(X_train_imputed, y_train)
    y_pred = model.predict(X_test_imputed)
    return y_pred
