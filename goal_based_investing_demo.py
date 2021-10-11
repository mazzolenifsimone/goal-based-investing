import dash
import dash_table
from dash_bootstrap_components._components.Col import Col
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash_html_components.Font import Font
from dash_html_components.Hr import Hr
import plotly.express as px
import ALMplanner as ALM
import ALMChart as ALMc
import plotly.express as px





#############
# Constants #
#############

EXAMPLE_OPTIONS = [ {"label": "20y-feasible-3-goals", "value": 1},
                    #{"label": "20y-unfeasible-3-goals", "value":2},
                    {"label": "40y-feasible-pensionfund", "value":3},
                    {"label":"20y-feasible-singlegoal", "value":4},
]

USER_RISK_PROFILE_MARKS = { 23:{"label":"low", 'style':{"transform": "rotate(45deg)", 'color':'limegreen'}}, 
              37:{"label":"low-mid",'style':{"transform": "rotate(45deg)",'color':'greenyellow'}},
              54:{"label": "mid",'style':{"transform": "rotate(45deg)",'color':'gold'}},
              70:{"label":"mid-high",'style':{"transform": "rotate(45deg)",'color':'orange'}},
              84:{"label":"high",'style':{"transform": "rotate(45deg)",'color':'red'}},
}

FIRST_EXAMPLE_ID = 1

GB_STRAT = 'Goal-based strategy'
BAH_STRAT = 'Buy-and-hold strategy'


##############
# CSS Styles #
##############

SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'margin-left': '2%',
    'padding': '20px 10px',
}

CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}

SUBTAB_STYLE = {
    'margin-left': '5%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px',
}

TITLE_STYLE = {
    'textAlign':'center',
    'font-weight':'bold',
    'font-size':'150%',
    'margin-top':'1%',
    'margin-left':'5%',
}

SUBTITLE_STYLE = {
    'textAlign':'left',
    'font-weight':'bold',
    'font-size':'100%',
    'margin-bottom':'3%',
}

TEXT_STYLE = {
    'textAlign': 'center',
}

BUTTON_STYLE = {
    'textAlign':'center',
    'width':'100%',
    'border':'none',
}

SUCCESS_STYLE = {
    "color":"green",
    "font-size":"150%",
    'textAlign':'center',
    'width':'100%',
}

FAILURE_STYLE = {
    "color":"red",
    "font-size":"150%",
    'textAlign':'center',
    'width':'100%',
}

TOTAL_WEALTH_STYLE_1 = {
    "color":"blue",
    "font-size":"100%",
    'textAlign':'left',
    'width':'100%',
    "vertical-align":"middle",
}

TOTAL_WEALTH_STYLE_2 = {
    "color":"blue",
    "font-size":"150%",
    'textAlign':'center',
    'width':'100%',
    "vertical-align":"middle",
}

GOAL_PAYOFF_STYLE_1 = {
    "color":"green",
    "font-size":"100%",
    'textAlign':'left',
    'width':'100%',
    "vertical-align":"middle",
}

GOAL_PAYOFF_STYLE_2 = {
    "color":"green",
    "font-size":"150%",
    'textAlign':'center',
    'width':'100%',
    "vertical-align":"middle",
}

FAILURE_SHORTFALL_STYLE_1 = {
    "color":"red",
    "font-size":"100%",
    'textAlign':'left',
    'width':'100%',
    "vertical-align":"middle",
}

FAILURE_SHORTFALL_STYLE_2 = {
    "color":"red",
    "font-size":"150%",
    'textAlign':'center',
    'width':'100%',
    "vertical-align":"middle",
}




##############
# App Layout #
##############

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY],)

controls = dbc.FormGroup(
    [html.P('Select example', style = {'textAlign':'center'}),
    dcc.Dropdown(
        id = 'example_id',
        options = EXAMPLE_OPTIONS, 
        value = FIRST_EXAMPLE_ID,
    ),
    html.Hr(),
    html.P('Select user risk attitude', style = {'textAlign':'center'}),
    dcc.Slider(
        id = 'user_risk_profile',
        min = min(USER_RISK_PROFILE_MARKS.keys()),
        max = max(USER_RISK_PROFILE_MARKS.keys()),
        value = min(USER_RISK_PROFILE_MARKS.keys()),
        marks = USER_RISK_PROFILE_MARKS,
    ),
    html.Hr(),
    html.Button('Update chart', id='update_chart', style = BUTTON_STYLE, className='btn'),
    html.Hr(),
    html.Button('Submit problem', id='submit_problem', style = BUTTON_STYLE, className='btn'),
    ]
)

sidebar = html.Div([
    html.H2("Control Panel", style = TEXT_STYLE),
    html.Hr(),
    controls
    ],
    style=SIDEBAR_STYLE
)

solution_header = [dbc.Col(html.Div(GB_STRAT,style=TITLE_STYLE), width = 6),
                dbc.Col(html.Div(BAH_STRAT,style=TITLE_STYLE), width = 6),]

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




###########################
# Callbacks and functions #
###########################

def get_planner_datatable(df):
    df_index = df.reset_index().rename(columns= {"index" : "Label"})
    return dash_table.DataTable(
        id='assettable',
        columns=[{"name": i, "id": i} for i in df_index.columns],
        data=df_index.to_dict('records'),
    )

def plan_succes_prob_form(success_prob,fail_prob):
    return html.Div(
        [
            dbc.Row([dbc.Col(f"Success: {str(success_prob)}%", style = SUCCESS_STYLE,), dbc.Col(f"Failure: {str(fail_prob)}%", style = FAILURE_STYLE),],),
        ]
    )

def eop_wealth_summary_form(tot_wealth_avg,paid_advance_avg,avg_loss_in_failure):
    return html.Div(
        [
            dbc.Row([dbc.Col("Expected total wealth:", style = TOTAL_WEALTH_STYLE_1,), dbc.Col(str(int(tot_wealth_avg))+"€",style=TOTAL_WEALTH_STYLE_2)]),
            dbc.Row([dbc.Col("Expected goals payoff:", style = GOAL_PAYOFF_STYLE_1), dbc.Col(str(int(paid_advance_avg))+"€", style=GOAL_PAYOFF_STYLE_2)]),
            dbc.Row([dbc.Col("Shortfall in case of failure:", style = FAILURE_SHORTFALL_STYLE_1), dbc.Col(str(int(avg_loss_in_failure))+"€", style = FAILURE_SHORTFALL_STYLE_2)]),
        ],
    )

def get_planner_tab(problem):
    problem_chart = ALMc.planner_chart(problem, bar_width=6)
    ALMc.standardized_chart(problem_chart, perc = False, showlegend= True)
    return html.Div(
        [
            dbc.Row([dbc.Col('Investment Planner',style=TITLE_STYLE)]),
            html.Hr(),
            dbc.Row([dbc.Col(dcc.Graph(figure = problem_chart), width = True)]),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(html.Div('Assets input',style=SUBTITLE_STYLE), width = 6),
                    dbc.Col(html.Div('Goals input' ,style=SUBTITLE_STYLE), width = 6),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(get_planner_datatable(problem.assets.lists().drop("Date", axis = 1)), width = 6),
                    dbc.Col(get_planner_datatable(problem.liabilities.lists().drop(["Date", "CVaR Level"], axis = 1)), width = 6)
                ]
            )

        ],
        style = SUBTAB_STYLE,
    )


def get_asset_allocation_tab(asset_allocation_chart):
    return html.Div(
        [
            dbc.Row(dbc.Col(html.Div('Strategic asset allocation evolution',style=SUBTITLE_STYLE))),
            dbc.Row([
                dbc.Col(dcc.Graph(figure = asset_allocation_chart), width = True),
            ]),
            dbc.Row(solution_header),
        ],
        style = SUBTAB_STYLE,
    )

def get_shortfall_prob_tab(title,shortfall_prob_chart, plan_succes_prob):
    return html.Div(
        [
            dbc.Row([dbc.Col(html.Div(title,style=TITLE_STYLE), width = True)]),
            html.Hr(),
            dbc.Row(dbc.Col(html.Div('Plan success and failure probabilities',style=SUBTITLE_STYLE))),
            dbc.Row([dbc.Col(plan_succes_prob, width = True)]),
            html.Hr(),
            dbc.Row(dbc.Col(html.Div('Shortfall occurrence odds at goal\'s payoff',style=SUBTITLE_STYLE))),
            dbc.Row([dbc.Col(dcc.Graph(figure = shortfall_prob_chart), width = True)]),
        ],
        style = SUBTAB_STYLE,
    )

def get_goal_payoff_tab(title,goal_payoff_chart, eop_wealth_summary):
    return html.Div(
        [
            dbc.Row([dbc.Col(html.Div(title,style=TITLE_STYLE), width = True)]),
            html.Hr(),
            dbc.Row([dbc.Col(html.Div("Investment plan forecast summary", style = SUBTITLE_STYLE), width = True)]),
            dbc.Row([dbc.Col(eop_wealth_summary, width = True)]),
            html.Hr(),
            dbc.Row([dbc.Col(html.Div("Expected and worst-case payoffs by goals", style = SUBTITLE_STYLE), width = True)]),
            dbc.Row([dbc.Col(dcc.Graph(figure = goal_payoff_chart), width = True)]),
        ],
        style = SUBTAB_STYLE,
    )

@app.callback(
    [
        Output('planner_tab', 'children')
    ],
    [
        Input('update_chart','n_clicks'),
        State('example_id', 'value'),
        State('user_risk_profile', 'value')
    ],
    #prevent_initial_call = True,
    )
def update_planner_graph(update_btn, example_id, user_risk_profile):
    if not(example_id is None):
        global problem
        problem = ALM.generate_example(example_id, user_risk_profile)
    return [get_planner_tab(problem)]

@app.callback(
    [
        Output('asset_allocation_tab', 'children'), 
        Output('shortfall_prob_tab', 'children'), 
        Output('goal_payoff_tab', 'children'),
        Output('asset_allocation_tab', 'disabled'), 
        Output('shortfall_prob_tab', 'disabled'), 
        Output('goal_payoff_tab', 'disabled')
    ],
    [
        Input('submit_problem','n_clicks'),
        State('user_risk_profile', 'value')
    ], 
    prevent_initial_call = True
    )
def submit_problem_output(update_btn, user_risk_profile):
    global problem
    GB_model = ALM.ALMGoalBased(problem)
    GB_model.solve()
    BaH_model = ALM.ALMBuyAndHold_2(problem)
    BaH_model.solve()

    sol = GB_model.solution
    sol_bah = BaH_model.solution

    perc = False
    AAChart_gb = ALMc.AssetAllocationChart(problem,sol,perc = perc)
    AAChart_bah = ALMc.AssetAllocationChart(problem,sol_bah, perc = perc, showlegend= False)
    AACharts = ALMc.AssetAllocationComparison(AAChart_gb,AAChart_bah)
    prob, avg = ALMc.GoalRiskDetails(problem, sol, perc)
    prob_bah, avg_bah = ALMc.GoalRiskDetails(problem, sol_bah, perc)
    success_prob, fail_prob, tot_wealth_avg, paid_advance_avg, avg_loss_in_failure = ALMc.EoPWealthInfo(problem, sol)
    success_prob_bah, fail_prob_bah, tot_wealth_avg_bah, paid_advance_avg_bah, avg_loss_in_failure_bah = ALMc.EoPWealthInfo(problem, sol_bah)

    ps_Div_gb = plan_succes_prob_form(success_prob,fail_prob)
    ps_Div_bah = plan_succes_prob_form(success_prob_bah,fail_prob_bah)
    eop_Div_gb = eop_wealth_summary_form(tot_wealth_avg,paid_advance_avg,avg_loss_in_failure)
    eop_Div_bah = eop_wealth_summary_form(tot_wealth_avg_bah,paid_advance_avg_bah,avg_loss_in_failure_bah)

    return [
        [get_asset_allocation_tab(AACharts)], 
        [dbc.Row([
            dbc.Col(get_shortfall_prob_tab(GB_STRAT,prob,ps_Div_gb), width = 6), 
            dbc.Col(get_shortfall_prob_tab(BAH_STRAT,prob_bah,ps_Div_bah), width = 6)])
            ], 
        [dbc.Row([
            dbc.Col(get_goal_payoff_tab(GB_STRAT,avg,eop_Div_gb), width = 6), 
            dbc.Col(get_goal_payoff_tab(BAH_STRAT,avg_bah,eop_Div_bah), width = 6)])
            ],
        False, 
        False, 
        False
        ]

if __name__=='__main__':
    app.run_server(debug = True)

