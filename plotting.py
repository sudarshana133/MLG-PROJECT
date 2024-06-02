import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

def create_plot(y_test, y_pred, stock_data):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()

    numeric_columns = stock_data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_columns.corr()

    plt.subplot(1, 2, 2)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64
