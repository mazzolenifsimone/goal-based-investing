
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import ALMplanner as ALM
import ALMChart as ALMc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    #'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}

TITLE_STYLE = {
    'textAlign':'center',
    'font-weight':'bold',
    'font-size':'150%',
    'margin-top':'5%',
    'margin-left':'5%',
}

TEXT_STYLE = {
    'textAlign': 'center',
    #'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    #'color': '#0074D9'
}

BUTTON_STYLE = {
    'textAlign':'center',
    'width':'100%',
    'border':'none',
}

example_options = [ {"label": "20y-feasible-3-goals", "value": 1},
                    #{"label": "20y-unfeasible-3-goals", "value":2},
                    {"label": "40y-feasible-pensionfund", "value":3},
                    {"label":"20y-feasible-singlegoal", "value":4},
]

urp_marks = { 23:{"label":"low", 'style':{"transform": "rotate(45deg)", 'color':'limegreen'}}, 
              37:{"label":"low-mid",'style':{"transform": "rotate(45deg)",'color':'greenyellow'}},
              54:{"label": "mid",'style':{"transform": "rotate(45deg)",'color':'gold'}},
              70:{"label":"mid-high",'style':{"transform": "rotate(45deg)",'color':'orange'}},
              84:{"label":"high",'style':{"transform": "rotate(45deg)",'color':'red'}},
}

def generate_example(example_ID, user_risk_profile):
    if example_ID == 1:
        problem = ALM.ALMplanner(start = "2021", end = "2041", user_risk_profile = user_risk_profile)
        problem.liabilities.insert("car", "2026", 25000, 25000*0.65)
        problem.liabilities.insert("university", "2029", 50000, 50000*0.95)
        problem.liabilities.insert("hawaii", "2037",25000, 25000*0.85) 
        problem.assets.insert("init","Jan 2021",30000)
        ALM.add_recurrent(problem, start = "Jan 2022", end = "Jan 2027", type = "asset", value = 10000, label = "ass")
    elif example_ID == 3:
        problem = ALM.ALMplanner(start = "Jan 2021", end = "Jan 2061", user_risk_profile = user_risk_profile)
        ALM.add_recurrent(problem, start = "Jan 2021", end = "Jan 2040", type = "asset", value = 1000, label = "ass")
        ALM.add_recurrent(problem, start = "Jan 2041", end = "Jan 2060", type = "goal", value_tg = 1500, value_lb = 1100, label = "ret")
    elif example_ID == 4:
        problem = ALM.ALMplanner(start = "2021", end = "2041", user_risk_profile = user_risk_profile)
        problem.liabilities.insert("car", "2036", 45000, 45000*0.65) 
        problem.assets.insert("init","Jan 2021",40000)
    elif example_ID == 2:
        problem = ALM.ALMplanner(start = "2021", end = "2041", user_risk_profile = user_risk_profile)
        problem.liabilities.insert("car", "2026", 30000, 30000*0.65)
        problem.liabilities.insert("university", "2029", 50000, 50000*0.95)
        problem.liabilities.insert("hawaii", "2037",30000, 30000*0.85) 
        problem.assets.insert("init","Jan 2021",30000)
        ALM.add_recurrent(problem, start = "Jan 2022", end = "Jan 2027", type = "asset", value = 10000, label = "ass")
    return problem

def legend_top_position(plt):
    plt.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="center",
        x=0.5)
    )

first_example_ID = 1
first_urp = 23

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY],)

controls = dbc.FormGroup(
    [html.P('Select example', style = {'textAlign':'center'}),
    dcc.Dropdown(
        id = 'example_id',
        options = example_options, 
        value = first_example_ID,
    ),
    html.Hr(),
    html.P('Select user risk attitude', style = {'textAlign':'center'}),
    dcc.Slider(
        id = 'user_risk_profile',
        min=min(urp_marks.keys()),
        max=max(urp_marks.keys()),
        value = min(urp_marks.keys()),
        marks = urp_marks,
    ),
    html.Hr(),
    html.Button('Update chart', id='update_chart', style = BUTTON_STYLE, className='btn'),
    html.Hr(),
    html.Button('Submit problem', id='submit_problem', style = BUTTON_STYLE, className='btn'),
    ]
)

sidebar = html.Div([
    html.H2("Controls", style = TEXT_STYLE),
    html.Hr(),
    controls
    ],
    style=SIDEBAR_STYLE
)

def get_planner_tab(planner_chart):
    return html.Div(
        [
            dbc.Row(dbc.Col(dcc.Graph(figure = planner_chart), width = True)),
        ]
    )

def get_asset_allocation_tab(asset_allocation_chart):
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.Div('Goal-Based strategy',style=TITLE_STYLE), width = 6),
                    dbc.Col(html.Div('Buy-and-hold strategy',style=TITLE_STYLE), width = 6),
                ],
            ),
            dbc.Row(dbc.Col(dcc.Graph(figure = asset_allocation_chart), width = True)),
        ]
    )

def get_shortfall_prob_tab(shortfall_prob_chart):
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.Div('Goal-Based strategy',style=TITLE_STYLE), width = 6),
                    dbc.Col(html.Div('Buy-and-hold strategy',style=TITLE_STYLE), width = 6),
                ],
            ),
            dbc.Row(dbc.Col(dcc.Graph(figure = shortfall_prob_chart), width = True)),
        ]
    )

def get_goal_payoff_tab(goal_payoff_chart):
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.Div('Goal-Based strategy',style=TITLE_STYLE), width = 6),
                    dbc.Col(html.Div('Buy-and-hold strategy',style=TITLE_STYLE), width = 6),
                ],
            ),
            dbc.Row(dbc.Col(dcc.Graph(figure = goal_payoff_chart), width = True)),
        ]
    )

planner_tab = dcc.Tab(
    id = "planner_tab", 
    label = 'Investment Planner',
)

asset_allocation_tab = dcc.Tab(
    id = "asset_allocation_tab", 
    label = 'Solution: Asset Allocation', 
    disabled= True
)

shortfall_prob_tab = dcc.Tab(
    id = "shortfall_prob_tab", 
    label = 'Solution: Success Probability', 
    disabled= True
)

goal_payoff_tab = dcc.Tab(
    id = "goal_payoff_tab", 
    label = 'Solution: Payoff',
    disabled= True
)

content = html.Div(
    [
        dcc.Tabs(
            id = "content",
            children =[planner_tab, asset_allocation_tab, shortfall_prob_tab, goal_payoff_tab],
        )
    ],
    style = CONTENT_STYLE
)

app.layout = html.Div([sidebar, content])

@app.callback([Output('planner_tab', 'children')],[Input('update_chart','n_clicks'), State('example_id', 'value'),State('user_risk_profile', 'value')])#, prevent_initial_call = True)
def update_planner_graph(update_btn, example_id, user_risk_profile):
    if not(example_id is None):
        global problem
        problem = generate_example(example_id, user_risk_profile)
        problem_chart = ALMc.planner_chart(problem, bar_width=6)
        ALMc.standardized_chart(problem_chart, perc = False, showlegend= True)
        #problem_chart.update_layout(margin=dict(t=50,l=20,b=20,r=20))
    return [get_planner_tab(problem_chart)]

@app.callback([Output('asset_allocation_tab', 'children'), Output('shortfall_prob_tab', 'children'), Output('goal_payoff_tab', 'children'),Output('asset_allocation_tab', 'disabled'), Output('shortfall_prob_tab', 'disabled'), Output('goal_payoff_tab', 'disabled')],[Input('submit_problem','n_clicks'),State('user_risk_profile', 'value')], prevent_initial_call = True)
def submit_problem_output(update_btn, user_risk_profile):
    global problem
    GB_model = ALM.ALMGoalBased(problem)
    GB_model.solve()
    BaH_model = ALM.ALMBuyAndHold_2(problem)
    BaH_model.solve()

    sol = GB_model.solution
    sol_bah = BaH_model.solution
    
    colormap_ETF = {}
    it = -1
    for p in problem.P:
        it = it+1
        colormap_ETF[p] = px.colors.qualitative.Plotly[it%10]

    #asset = ALMc.AssetSplitDetailsChart(problem, sol, "ETF", colormap_ETF)    
    #asset_bah = ALMc.AssetSplitDetailsChart(problem, sol_bah, "ETF", colormap_ETF)
    n_scen = None
    perc = False
    plt = ALMc.AssetAllocationChart(problem,sol,n_scen=n_scen, perc = perc)
    plt_bah = ALMc.AssetAllocationChart(problem,sol_bah,n_scen=n_scen, perc = perc, showlegend= False)
    prob, avg = ALMc.GoalRiskDetails(problem, sol, perc)
    prob_bah, avg_bah = ALMc.GoalRiskDetails(problem, sol_bah, perc, showlegend = False)

    showlegend = True
    horizontal_spacing = 0.03

    fig1 = go.FigureWidget(make_subplots(rows = 1, cols = 2, shared_yaxes= True, horizontal_spacing = horizontal_spacing))
    fig1.add_traces(data=plt.data, rows = 1, cols = 1)
    fig1.add_traces(data=plt_bah.data, rows = 1, cols = 2)
    ALMc.standardized_chart(fig1, perc = perc, showlegend= showlegend)
    legend_top_position(fig1)

    fig2 = go.FigureWidget(make_subplots(rows = 1, cols = 2, shared_yaxes= True, horizontal_spacing = horizontal_spacing))
    fig2.add_traces(data=prob.data, rows = 1, cols = 1)
    fig2.add_traces(data=prob_bah.data, rows = 1, cols = 2)
    fig2.update_layout(barmode = "stack")
    fig2 = ALMc.standardized_chart(fig2, perc= True, showlegend=showlegend)
    legend_top_position(fig2)

    fig3 = go.FigureWidget(make_subplots(rows = 1, cols = 2, shared_yaxes= True, horizontal_spacing = horizontal_spacing))
    fig3.add_traces(data=avg.data, rows = 1, cols = 1)
    fig3.add_traces(data=avg_bah.data, rows = 1, cols = 2)
    fig3.update_layout(barmode = "overlay")
    fig3 = ALMc.standardized_chart(fig3, perc = perc, showlegend=showlegend)
    legend_top_position(fig3)

    return [[get_asset_allocation_tab(fig1)], [get_shortfall_prob_tab(fig2)], [get_goal_payoff_tab(fig3)], False, False, False]


if __name__=='__main__':
    app.run_server(debug = True)

