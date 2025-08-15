import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__)

# Define available components
components = [
    "Linear", "Dropout", "Batch Normalization", "Convolutional", "Pool",
    "Flatten", "ReLU Activation", "Leaky ReLU", "ELU", "Sigmoid Activation"
]

app.layout = html.Div(
    style={'fontFamily': 'Arial', 'textAlign': 'center', 'background': 'linear-gradient(120deg, #2c3e50, #4ca1af)', 'color': 'white', 'minHeight': '100vh', 'padding': '20px'},
    children=[
        html.H1("Neural Network Trainer"),

        html.Div([
            html.H3("Select Components"),
            html.Div(
                [html.Button(c, id={'type': 'add-comp', 'index': c}, n_clicks=0,
                             style={'margin': '5px', 'padding': '10px', 'background': '#f1c40f', 'color': 'black', 'borderRadius': '5px', 'fontWeight': 'bold'})
                 for c in components],
                style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}
            )
        ]),

        html.H3("Network Configuration"),
        html.Div(id="network-config", style={'margin': '20px', 'padding': '10px', 'border': '2px dashed #ccc', 'borderRadius': '10px'}),

        html.Div([
            html.Button("Train Model", id="train-btn", style={'margin': '10px'}),
            html.Button("Reset", id="reset-btn", style={'margin': '10px'})
        ]),

        dcc.Graph(id="plot")
    ]
)

@app.callback(
    Output("network-config", "children"),
    Input({'type': 'add-comp', 'index': dash.ALL}, 'n_clicks'),
    Input("reset-btn", "n_clicks"),
    State("network-config", "children")
)
def update_network(add_clicks, reset_click, current_config):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_config

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if "reset-btn" in trigger_id:
        return []

    if current_config is None:
        current_config = []

    # Identify which button was clicked
    trigger = eval(trigger_id)
    comp_name = trigger['index']

    current_config.append(html.Div(comp_name, style={'background': '#3498db', 'color': 'white', 'padding': '8px',
                                                     'margin': '5px auto', 'width': '160px', 'borderRadius': '5px'}))
    return current_config

@app.callback(
    Output("plot", "figure"),
    Input("train-btn", "n_clicks"),
    State("network-config", "children")
)
def train_model(n_clicks, config):
    if not n_clicks or not config:
        return go.Figure()

    x = np.linspace(0, 10, 100)
    y_true = np.sin(x)
    y_pred = np.sin(x) * 0.9 + np.random.rand(100) * 0.2 - 0.1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_true, mode='lines', name='Target Function', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='Predicted Function', line=dict(color='red', dash='dot')))
    fig.update_layout(title="Training Results", template='plotly_dark')
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
