import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import dash_bootstrap_components as dbc
from data import merge_demand_and_temp_data
from conf import merged_data_demand_path, raw_temperature_data_path

demand_data_file_path = os.path.join(os.curdir, merged_data_demand_path)
uk_temp_file_path = os.path.join(os.curdir, raw_temperature_data_path)
demand_df = pd.read_csv(demand_data_file_path)
uk_temp_df = pd.read_csv(uk_temp_file_path)
merged_df = merge_demand_and_temp_data(demand_df, uk_temp_df)

# Convert 'SETTLEMENT_DATE' to datetime format (if it is not already)
merged_df['SETTLEMENT_DATE'] = pd.to_datetime(merged_df['SETTLEMENT_DATE'])

# Set 'SETTLEMENT_DATE' as the DataFrame index
merged_df.set_index('SETTLEMENT_DATE', inplace=True)

# De-seasonalize TSD
result = seasonal_decompose(merged_df['TSD'], model='additive', period=48)  # added freq=48
merged_df['TSD_de-seasonalized'] = result.trend

# Drop any rows with missing values
merged_df.dropna(inplace=True)

# Create a linear regression model
X = merged_df['temp_c'].values.reshape(-1, 1)  # Feature: temperature
y = merged_df['TSD_de-seasonalized']  # Target: de-seasonalized TSD

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Create a plotly graph object
fig = go.Figure()
fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Actual'))
fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred, mode='lines', name='Fitted line'))

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("UK's energy Demand based on Temperature", className="display-4"),
        html.Hr(),
        html.P(
            "UK's energy Demand based on Temperature for 2017", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Demand Data", href="/demand-data", active="exact"),
                dbc.NavLink("UK Temperature Data", href="/temp-data", active="exact"),
                dbc.NavLink("LinearRegression-TSD Predict", href="/predict-tsd", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("This is the content of the home page!")
    elif pathname == "/demand-data":
        return dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in demand_df.columns],
            data=demand_df.to_dict('records'),
            page_size=10,  # we have pagination enabled and choose a size
            style_table={'overflowX': 'auto'}  # for responsiveness
        )
    elif pathname == "/temp-data":
        return dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in uk_temp_df.columns],
            data=uk_temp_df.to_dict('records'),
            page_size=10,  # we have pagination enabled and choose a size
            style_table={'overflowX': 'auto'}  # for responsiveness
        )
    elif pathname == "/predict-tsd":
        return dcc.Graph(
        id='scatter-plot',
        figure=fig
    )
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    app.run_server(port=8888)
