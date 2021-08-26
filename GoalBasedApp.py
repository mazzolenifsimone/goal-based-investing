import ALMPlanner as ALM
import ALMChart as  ALMc
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import time
import base64
import pickle
import pandas
import io

buyandhold_portfolios = ALM.load_scenario("scenario","eff_front")

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.4/css/bulma.min.css']

def generate_example(EX):
    user_portfolio_code = {
    0:23,
    1:37,
    2:56,
    3:74,
    4:84
    }
    user_risk_profile = 2
    if EX == 1:
        problem = ALM.ALMplanner(start = "2021", end = "2041", user_risk_profile = user_portfolio_code[user_risk_profile])
        problem.liabilities.insert("car", "2026", 25000, 25000*0.65)
        problem.liabilities.insert("university", "2029", 50000, 50000*0.95)
        problem.liabilities.insert("hawaii", "2037",25000, 25000*0.85) 
        problem.assets.insert("init","Jan 2021",30000)
        ALM.add_recurrent(problem, start = "Jan 2022", end = "Jan 2027", type = "asset", value = 10000, label = "ass")
    elif EX == 2:
        problem = ALM.ALMplanner(start = "Jan 2021", end = "Jan 2061", user_risk_profile = user_portfolio_code[user_risk_profile])
        ALM.add_recurrent(problem, start = "Jan 2021", end = "Jan 2040", type = "asset", value = 1000, label = "ass")
        ALM.add_recurrent(problem, start = "Jan 2041", end = "Jan 2060", type = "goal", value_tg = 1500, value_lb = 1100, label = "ret")
    elif EX == 3:
        problem = ALM.ALMplanner(start = "2021", end = "2041", user_risk_profile = user_portfolio_code[user_risk_profile])
        problem.liabilities.insert("car", "2036", 45000, 45000*0.65) 
        problem.assets.insert("init","Jan 2021",40000)
    elif EX == 4:
        problem = ALM.ALMplanner(start = "2021", end = "2041", user_risk_profile = user_portfolio_code[user_risk_profile])
        problem.liabilities.insert("car", "2026", 30000, 30000*0.65)
        problem.liabilities.insert("university", "2029", 50000, 50000*0.95)
        problem.liabilities.insert("hawaii", "2037",30000, 30000*0.85) 
        problem.assets.insert("init","Jan 2021",30000)
        ALM.add_recurrent(problem, start = "Jan 2022", end = "Jan 2027", type = "asset", value = 10000, label = "ass")
    return problem


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

select_example = dcc.Dropdown(
        id='select_example',
        options=[
            {'label': 'Base', 'value': 1},
            {'label': 'UnFeas Base', 'value': 4},
            {'label': 'Long', 'value': 2}
        ]
    )

solve_button = html.Button("Solve", id = "solve-button")

planner_chart = dcc.Graph(id = "planner_chart")
solution_chart = dcc.Graph(id = "solution_chart")

solution = html.Div([
                html.H3('Tab content 2', style={'textAlign': 'center'}),
                solution_chart
            ])

app.layout = html.Div([
    dcc.Tabs(id='tab_app', value='tab_app', children=[
        dcc.Tab(label='Wealth Planner', 
                value='planner', 
                children = html.Div([
                                html.H3('Select an example:', style={'textAlign': 'center'}),
                                select_example,
                                planner_chart,
                                solve_button
                            ])
                ),
        dcc.Tab(label='Investment Solution', value='solution',children = solution),
    ])
])

@app.callback(Output('planner_chart', 'figure'),
              Input('select_example', 'value'))
def load_example(n_ex):
    if n_ex is None:
        plt = go.Figure()
        ALMc.standardized_chart(plt)    
    else:
        problem = generate_example(n_ex)
        plt = ALMc.display(problem, bar_width=6)
    return plt

@app.callback(Output('solution', 'figure'),
              Input('solve-button', 'click'))
def solve(n_ex):
    if n_ex is None:
        plt = go.Figure()
        ALMc.standardized_chart(plt)    
    else:
        problem = generate_example(n_ex)
        plt = ALMc.display(problem, bar_width=6)
    return plt

if __name__ == '__main__':
    app.run_server(debug=True)


