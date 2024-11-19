import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
from darts import TimeSeries
import plotly.graph_objs as go
from darts.models import (
    ExponentialSmoothing,
    ARIMA,
    RandomForest,
    XGBModel,
    NHiTSModel
)
import matplotlib.pyplot as plt
import io
import base64
import boto3
from io import StringIO

# AWS S3 details
S3_BUCKET_NAME = "my-dash-app-data"  # Replace with your S3 bucket name
S3_FILE_KEY = "powerconsumption.csv"  # Replace with the path to your file in the S3 bucket

# Function to load data from S3
def load_data_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(data))

# Load the data from S3
df = load_data_from_s3(S3_BUCKET_NAME, S3_FILE_KEY)
df = df.tail(1000)
df['Datetime'] = pd.to_datetime(df['Datetime'])

series = TimeSeries.from_dataframe(df, time_col='Datetime', value_cols='PowerConsumption_Zone2')
df.set_index('Datetime', inplace=True)

# Initialize the Dash app
app = dash.Dash('Cloud project app')

# Global variables for the models
model_arima = None
model_hw = None
model_rf = None
model_xgb = None
model_nhits = None

# Train and initialize the models
def train_models():
    global model_arima, model_hw, model_rf, model_xgb, model_nhits

    # Split data into train and test
    train_size = int(0.8 * len(series))
    train, test = series[:train_size], series[train_size:]

    # Train ARIMA model
    model_arima = ARIMA()
    model_arima.fit(train)

    # Train Holt-Winters model
    seasonal_periods = 144
    model_hw = ExponentialSmoothing(seasonal_periods=seasonal_periods)
    model_hw.fit(train)

    # Train RandomForest model
    model_rf = RandomForest(lags=12)
    model_rf.fit(train)

    # Train XGBoost model
    model_xgb = XGBModel(lags=12)
    model_xgb.fit(train)

    # Train N-HiTS model
    model_nhits = NHiTSModel(input_chunk_length=288, output_chunk_length=144)
    model_nhits.fit(train)

# Train all models when the app starts
train_models()

# Define app layout
app.layout = html.Div([
    html.H1("Power Consumption Dashboard"),

    dcc.Tabs([
        # Tab for visualization
        dcc.Tab(label='Data Visualization', children=[
            html.Div([
                dcc.Graph(id='line-chart'),
                dcc.Dropdown(
                    id='zone-select',
                    options=[
                        {'label': 'Zone 1', 'value': 'PowerConsumption_Zone1'},
                        {'label': 'Zone 2', 'value': 'PowerConsumption_Zone2'},
                        {'label': 'Zone 3', 'value': 'PowerConsumption_Zone3'}
                    ],
                    value='PowerConsumption_Zone2',
                    clearable=False
                ),
                html.P("Select Zone"),
            ]),
        ]),

        # The second tab for Forecasting
        dcc.Tab(label='Forecasting', children=[
            html.Div([
                dcc.Dropdown(
                    id='model-select',
                    options=[
                        {'label': 'ARIMA', 'value': 'ARIMA'},
                        {'label': 'Holt-Winters', 'value': 'Holt_Winters'},
                        {'label': 'Random Forest', 'value': 'RandomForest'},
                        {'label': 'XGBoost', 'value': 'XGBoost'},
                        {'label': 'N-HiTS', 'value': 'NHiTS'}
                    ],
                    value='ARIMA',
                    clearable=False
                ),
                html.Img(id='forecast-chart')  # Use an HTML <img> for Matplotlib plots
            ])
        ]),
    ])
])

# Callbacks
@app.callback(
    Output('line-chart', 'figure'),
    Input('zone-select', 'value')
)
def update_line_chart(zone):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[zone], mode='lines', name=zone))
    fig.update_layout(title="Power Consumption Over Time", xaxis_title="Datetime", yaxis_title="Power Consumption")
    return fig

@app.callback(
    Output('forecast-chart', 'src'),
    Input('model-select', 'value')
)
def forecast_model(model):
    # Split data into train and test
    train_size = int(0.8 * len(series))
    train, test = series[:train_size], series[train_size:]
    predictions = None

    # Select the model based on user input
    if model == 'ARIMA':
        model_instance = model_arima
    elif model == 'Holt_Winters':
        model_instance = model_hw
    elif model == 'RandomForest':
        model_instance = model_rf
    elif model == 'XGBoost':
        model_instance = model_xgb
    elif model == 'NHiTS':
        model_instance = model_nhits

    # Predict using the selected model
    predictions = model_instance.predict(len(test))

    # Extract predicted values and dates
    predicted_values = np.asarray(predictions.values())
    predicted_dates = np.asarray(predictions.time_index)

    # Actual values
    actual_values = np.asarray(test.values())
    actual_dates = np.asarray(test.time_index)

    # Plot with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actual_dates, actual_values, label='Actual', color='blue', linewidth=2)
    ax.plot(predicted_dates, predicted_values, label=f'{model} Prediction', color='orange', linestyle='--')
    ax.set_title(f'{model} Forecast vs Actual', fontsize=16)
    ax.set_xlabel('Datetime', fontsize=14)
    ax.set_ylabel('Power Consumption', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.xticks(rotation=45)

    # Convert Matplotlib plot to image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return f"data:image/png;base64,{encoded_image}"

# Run the app
app.server.run(debug=False, port=8031, host='0.0.0.0')
