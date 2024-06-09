import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
from flask import Flask, send_from_directory
import os
import base64
import requests
import train_predict
import plotting  # Ensure this import matches the location of your plotting.py
from flask_mail import Mail, Message
from dotenv import load_dotenv
import logging

load_dotenv()
server = Flask(__name__)

server.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
server.config['MAIL_PORT'] = 587
server.config['MAIL_USE_TLS'] = True
server.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
server.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
mail = Mail(server)

app = dash.Dash(__name__, server=server, external_stylesheets=['/static/styles.css'])

@server.route('/static/<path:path>')
def static_file(path):
    return send_from_directory('static', path)

app.layout = html.Div([
    html.Div([
        html.H1("Stock Price Predictor"),
        html.Button('Logout', id='logout-button', n_clicks=0)
    ], className='navbar'),
    html.Div([
        dcc.Input(id='company-name-input', type='text', placeholder='Enter company ticker...'),
        html.Div(id='input-error', style={'color': 'red'}),
        html.Button('Submit', id='submit-button', n_clicks=0)
    ]),
    html.Div(id='output-div'),
    html.Button('Send results', id='send-button', n_clicks=0),
    dcc.Store(id='local-storage', storage_type='local'),
    html.Div(id='auth-output'),

    # Modal for Login and Signup
    html.Div([
        html.Div([
            html.Span('×', id='modal-close', className='close'),
            html.H2('Login / Signup'),
            dcc.Input(id='email-input', type='email', placeholder='Enter your email...'),
            dcc.Input(id='password-input', type='password', placeholder='Enter your password...'),
            html.Button('Login', id='login-button', n_clicks=0),
            html.Button('Signup', id='signup-button', n_clicks=0),
            html.Div(id='modal-output', style={'color': 'red'})
        ], className='modal-content'),
    ], id='modal', className='modal', style={'display': 'none'})
])

@app.callback(
    Output('output-div', 'children'),
    Output('input-error', 'children'),
    Input('submit-button', 'n_clicks'),
    State('company-name-input', 'value')
)
def update_output(n_clicks, company_name):
    if n_clicks > 0:
        if not company_name:
            return html.Div(), "Please enter the ticker name."
        
        X_train_imputed, X_test_imputed, y_train, y_test, stock_data = train_predict.fetch_and_prepare_data(company_name)
        if X_train_imputed is None:
            return html.Div("Company doesn't exist or no data available or there is a network issue."), ""
        
        y_pred = train_predict.train_and_predict(X_train_imputed, X_test_imputed, y_train)
        
        summary_after_base64 = plotting.plot_summary_statistics(stock_data, 'Distribution of Non-null Values After Feature Engineering')
        corr_matrix_base64 = plotting.plot_correlation_matrix(stock_data)
        histograms_base64 = plotting.plot_histograms(stock_data)
        predictions_base64 = plotting.plot_predictions(y_test, y_pred)
        boxplots_base64 = plotting.plot_boxplots(stock_data)
        bump_chart64 = plotting.plot_bump_chart(y_test,y_pred)
        return (
            html.Div([
                html.Div([html.Img(src=f'data:image/png;base64,{summary_after_base64}')]),
                html.Div([html.Img(src=f'data:image/png;base64,{corr_matrix_base64}')]),
                html.Div([html.Img(src=f'data:image/png;base64,{histograms_base64}')]),
                html.Div([html.Img(src=f'data:image/png;base64,{predictions_base64}')]),
                html.Div([html.Img(src=f'data:image/png;base64,{boxplots_base64}')]),
                html.Div([html.Img(src=f'data:image/png;base64,{bump_chart64}')])
            ]),
            ""
        )
    
    return html.Div(), ""

@app.callback(
    Output('local-storage', 'data'),
    Output('auth-output', 'children'),
    Output('modal', 'style'),
    Output('modal-output', 'children'),
    Input('login-button', 'n_clicks'),
    Input('signup-button', 'n_clicks'),
    Input('logout-button', 'n_clicks'),
    Input('send-button', 'n_clicks'),
    Input('modal-close', 'n_clicks'),
    State('local-storage', 'data'),
    State('email-input', 'value'),
    State('password-input', 'value'),
    State('company-name-input', 'value')
)
def handle_auth_buttons(login_clicks, signup_clicks, logout_clicks, send_clicks, close_clicks, data, email, password, company_name):
    ctx = callback_context

    if not ctx.triggered:
        return data, "", {'display': 'none'}, ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'modal-close' and close_clicks > 0:
        return data, "", {'display': 'none'}, ""

    if button_id == 'login-button' and login_clicks > 0:
        try:
            response = requests.post('http://localhost:5000/login', json={'email': email, 'password': password})
            response.raise_for_status()
            message = response.json().get('message', 'Login failed')
            if response.status_code == 200:
                return {'email': email}, message, {'display': 'none'}, ""
            else:
                return data, message, {'display': 'block'}, message
        except requests.exceptions.RequestException as e:
            return data, f"Login request failed: {e}", {'display': 'block'}, f"Login request failed: {e}"

    if button_id == 'signup-button' and signup_clicks > 0:
        try:
            response = requests.post('http://localhost:5000/signup', json={'email': email, 'password': password})
            response.raise_for_status()
            message = response.json().get('message', 'Signup failed')
            return data, message, {'display': 'block'}, message
        except requests.exceptions.RequestException as e:
            return data, f"Signup request failed: {e}", {'display': 'block'}, f"Signup request failed: {e}"

    if button_id == 'logout-button' and logout_clicks > 0:
        return {}, "Logged out successfully.", {'display': 'block'}, ""

    if button_id == 'send-button' and send_clicks > 0:
        if not data or 'email' not in data:
            return data, "Please log in to send the results.", {'display': 'block'}, "Please log in to send the results."
        
        if not company_name:
            return data, "Please enter the ticker name before sending the results.", {'display': 'block'}, "Please enter the ticker name before sending the results."
        
        X_train_imputed, X_test_imputed, y_train, y_test, stock_data = train_predict.fetch_and_prepare_data(company_name)
        y_pred = train_predict.train_and_predict(X_train_imputed, X_test_imputed, y_train)

        summary_after_base64 = plotting.plot_summary_statistics(stock_data, 'Distribution of Non-null Values After Feature Engineering')
        corr_matrix_base64 = plotting.plot_correlation_matrix(stock_data)
        histograms_base64 = plotting.plot_histograms(stock_data)
        predictions_base64 = plotting.plot_predictions(y_test, y_pred)
        boxplots_base64 = plotting.plot_boxplots(stock_data)
        bump_chart64 = plotting.plot_bump_chart(y_test,y_pred)

        msg = Message("Stock Data Results", sender=os.getenv('MAIL_USERNAME'), recipients=[data['email']])
        msg.body = "Please find the attached stock data results."
        msg.attach("summary_after.png", "image/png", base64.b64decode(summary_after_base64))
        msg.attach("correlation_matrix.png", "image/png", base64.b64decode(corr_matrix_base64))
        msg.attach("histograms.png", "image/png", base64.b64decode(histograms_base64))
        msg.attach("predictions.png", "image/png", base64.b64decode(predictions_base64))
        msg.attach("boxplots.png", "image/png", base64.b64decode(boxplots_base64))
        msg.attach("bumpchat.png","image/png",base64.b64decode(bump_chart64))

        try:
            mail.send(msg)
            logging.info('Email sent successfully.')
            return data, "Email sent successfully.", {'display': 'block'}, ""
        except Exception as e:
            logging.error(f"Failed to send email: {e}")
            return data, f"Failed to send email: {e}", {'display': 'block'}, f"Failed to send email: {e}"

    return data, "", {'display': 'none'}, ""

if __name__ == '__main__':
    app.run_server(debug=True)