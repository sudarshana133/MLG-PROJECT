import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from flask import Flask, jsonify, request
from flask_mail import Mail, Message
import os
from dotenv import load_dotenv
import train_predict
import plotting
import requests
import base64
load_dotenv()
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

server.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
server.config['MAIL_PORT'] = 587
server.config['MAIL_USE_TLS'] = True
server.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
server.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
mail = Mail(server)

app.layout = html.Div([
    html.H1("Stock Data Plotter"),
    html.Div([
        dcc.Input(id='company-name-input', type='text', placeholder='Enter company ticker...'),
        html.Button('Submit', id='submit-button', n_clicks=0)
    ]),
    html.Div(id='output-div'),
    html.Button('Send results', id='send-button', n_clicks=0),
    dcc.Store(id='local-storage', storage_type='local'),
    dcc.Input(id='email-input', type='email', placeholder='Enter your email...'),
    dcc.Input(id='password-input', type='password', placeholder='Enter your password...'),
    html.Button('Signup', id='signup-button', n_clicks=0),
    html.Button('Login', id='login-button', n_clicks=0),
    html.Div(id='auth-output')
])

@app.callback(
    Output('output-div', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('company-name-input', 'value')]
)
def update_output(n_clicks, company_name):
    if n_clicks > 0:
        X_train_imputed, X_test_imputed, y_train, y_test, stock_data = train_predict.fetch_and_prepare_data(company_name)
        if X_train_imputed is None:
            return html.Div("Company doesn't exist or no data available.")
        y_pred = train_predict.train_and_predict(X_train_imputed, X_test_imputed, y_train)
        image_base64 = plotting.create_plot(y_test, y_pred, stock_data)
        return html.Img(src='data:image/png;base64,{}'.format(image_base64))
    return html.Div()

@app.callback(
    [Output('local-storage', 'data'),
     Output('auth-output', 'children')],
    [Input('send-button', 'n_clicks'),
     Input('login-button', 'n_clicks'),
     Input('signup-button', 'n_clicks')],
    [State('local-storage', 'data'),
     State('company-name-input', 'value'),
     State('email-input', 'value'),
     State('password-input', 'value')]
)
def handle_buttons(send_clicks, login_clicks, signup_clicks, data, company_name, email, password):
    ctx = dash.callback_context

    if not ctx.triggered:
        return data, ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'login-button' and login_clicks > 0:
        try:
            response = requests.post('http://localhost:5000/login', json={'email': email, 'password': password})
            response.raise_for_status()  # Raise HTTPError for bad responses
            message = response.json().get('message', 'Login failed')
            if response.status_code == 200:
                return {'email': email}, message
            else:
                return None, message
        except requests.exceptions.RequestException as e:
            return None, f"Login request failed: {e}"

    if button_id == 'signup-button' and signup_clicks > 0:
        try:
            response = requests.post('http://localhost:5000/signup', json={'email': email, 'password': password})
            response.raise_for_status()  # Raise HTTPError for bad responses
            return data, response.json().get('message', 'Signup failed')
        except requests.exceptions.RequestException as e:
            return data, f"Signup request failed: {e}"

    if button_id == 'send-button' and send_clicks > 0:
        if not data or 'email' not in data:
            return data, "Please log in to send the results."

        X_train_imputed, X_test_imputed, y_train, y_test, stock_data = train_predict.fetch_and_prepare_data(company_name)
        y_pred = train_predict.train_and_predict(X_train_imputed, X_test_imputed, y_train)
        image_base64 = plotting.create_plot(y_test, y_pred, stock_data)

        msg = Message("Stock Data Results", sender=os.getenv('MAIL_USERNAME'), recipients=[data['email']])
        msg.body = "Please find the attached stock data results."
        msg.attach("plot.png", "image/png", base64.b64decode(image_base64))
        mail.send(msg)
        return data, "Email sent successfully."

    return data, ""

if __name__ == '__main__':
    app.run_server(debug=True)
