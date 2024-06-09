import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64

def plot_summary_statistics(stock_data, title):
    summary_stats = stock_data.describe()
    counts = summary_stats.loc['count']
    fig, ax = plt.subplots()
    counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_correlation_matrix(stock_data):
    numeric_columns = stock_data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_columns.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_histograms(stock_data):
    fig, ax = plt.subplots()
    stock_data.hist(bins=50, ax=ax)
    plt.suptitle('Histograms of Stock Data')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_predictions(y_test, final_predictions):
    fig, ax = plt.subplots()
    ax.plot(y_test, label='Actual', color='blue')
    ax.plot(final_predictions, label='Predictions', color='red')
    ax.legend()
    ax.set_title('Stock Price Predictions')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_boxplots(stock_data):
    stock_data_for_boxplot = stock_data.drop(columns=['Date'], errors='ignore')
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # Decreased the width by changing figsize from (15, 10) to (10, 10)
    stock_data_for_boxplot.plot(kind='box', subplots=True, layout=(3, 3), ax=axes, patch_artist=True)
    fig.suptitle('Boxplots of Features')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_bump_chart(y_test, final_predictions):
    sorted_indices = np.argsort(y_test)
    sorted_actual = y_test[sorted_indices]
    sorted_predicted = final_predictions[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_actual, color='blue', label='Actual', marker='o')
    ax.plot(sorted_predicted, color='red', label='Predicted', marker='o')
    ax.set_xlabel('Observations')
    ax.set_ylabel('Values')
    ax.set_title('Bump Chart of Actual vs. Predicted Values')
    ax.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')