#%% Library, path and constants
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ALMplanner as ALM
import ALMChart as ALMc
import os
from ipywidgets import widgets, interact_manual, GridBox, Layout
from IPython.display import display

path_example = os.path.join(os.getcwd(), "example")

buyandhold_portfolios = ALM.load_scenario("scenario","eff_front")
user_portfolio_code = {
    0:23,
    1:37,
    2:56,
    3:74,
    4:84
    }

def generate_example(example_ID, user_risk_profile):
    if example_ID == 1:
        problem = ALM.ALMplanner(start = "2021", end = "2041", user_risk_profile = user_portfolio_code[user_risk_profile])
        problem.liabilities.insert("car", "2026", 25000, 25000*0.65)
        problem.liabilities.insert("university", "2029", 50000, 50000*0.95)
        problem.liabilities.insert("hawaii", "2037",25000, 25000*0.85) 
        problem.assets.insert("init","Jan 2021",30000)
        ALM.add_recurrent(problem, start = "Jan 2022", end = "Jan 2027", type = "asset", value = 10000, label = "ass")
    elif example_ID == 2:
        problem = ALM.ALMplanner(start = "Jan 2021", end = "Jan 2061", user_risk_profile = user_portfolio_code[user_risk_profile])
        ALM.add_recurrent(problem, start = "Jan 2021", end = "Jan 2040", type = "asset", value = 1000, label = "ass")
        ALM.add_recurrent(problem, start = "Jan 2041", end = "Jan 2060", type = "goal", value_tg = 1500, value_lb = 1100, label = "ret")
    elif example_ID == 3:
        problem = ALM.ALMplanner(start = "2021", end = "2041", user_risk_profile = user_portfolio_code[user_risk_profile])
        problem.liabilities.insert("car", "2036", 45000, 45000*0.65) 
        problem.assets.insert("init","Jan 2021",40000)
    elif example_ID == 4:
        problem = ALM.ALMplanner(start = "2021", end = "2041", user_risk_profile = user_portfolio_code[user_risk_profile])
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
        xanchor="right",
        x=1)
    )

def plan_succes_prob_widget(success_prob,fail_prob):
    return widgets.HTML(
    f"""<p style=\"text-align:center\"><b>Plan success/failure probability</b></p><br />  
    <p style=\"color:green; font-size:150%; text-align:center\">Success:</p> 
    <p style=\"color:green; font-size:200%; text-align:center\">{success_prob}%</p> 
    <p style=\"color:red; font-size:150%; text-align:center\">Failure:</p> 
    <p style=\"color:red; font-size:200%; text-align:center\">{fail_prob}%</p>"""
    )

def eop_wealth_summary_widget(tot_wealth_avg,paid_advance_avg,avg_loss_in_failure):
    return widgets.HTML(
    f"""<p style=\"text-align:center\"><b>End-of-period wealth summary</b></p>
    <p style=\"color:blue; font-size:150%; text-align:center\">Expected wealth</p>
    <p style=\"color:blue; font-size:200%; text-align:center\">{tot_wealth_avg}€</p>
    <p style=\"color:gold; font-size:150%; text-align:center\">Thereof paid in advance</p>
    <p style=\"color:gold; font-size:200%; text-align:center\">{paid_advance_avg}€</p>
    <p style=\"color:red; font-size:150%; text-align:center\">Shortfall in case of failure</p>
    <p style=\"color:red; font-size:200%; text-align:center\">{avg_loss_in_failure}€</p>"""
    )

# %% 
# Wealth planner
first_example_ID = 1
first_urp = 0

problem = generate_example(first_example_ID, first_urp)
problem_chart = go.FigureWidget(ALMc.planner_chart(problem, bar_width=6))
ALMc.standardized_chart(problem_chart, perc = False, showlegend= True)
problem_chart.update_layout(margin=dict(t=50,l=20,b=20,r=20))

SMF = problem.assets.lists()

example_options = [("20y-feasible-3-goals",1), ("20y-unfeasible-3-goals",4),("40y-feasible-pensionfund",2),("20y-feasible-singlegoal",3)]
urp_options = [("low risk",0), ("low-mid risk",1),("mid risk",2),("mid-high risk",3),("high risk",4)]

@interact_manual(example_select = widgets.Dropdown(options = example_options, value = 1, style = {'description': 'Select example:'}), user_risk_profile = widgets.Dropdown(options = urp_options, value = first_urp, style = {'description': 'User risk profile:'}))
def planner_demo(example_select, user_risk_profile):
    global problem
    problem = generate_example(example_select, user_risk_profile)
    new_plan = go.FigureWidget(ALMc.planner_chart(problem, bar_width=6))
    with problem_chart.batch_update():
        n_index = int(len(problem_chart.data))
        for i in np.arange(n_index):
            problem_chart.data[i].y = new_plan.data[i].y
            problem_chart.data[i].x = new_plan.data[i].x
    ALMc.standardized_chart(problem_chart, perc = False, showlegend= True)
    problem_chart.update_layout(margin=dict(t=50,l=20,b=20,r=20))

display(problem_chart)
# %% 
# Feasibility check and Assets-multiplier to get feasibility

problem.check_feasibility()
problem.get_feasibility()
#%% 
# Solve Model

GB_model = ALM.ALMGoalBased(problem)
GB_model.solve()

BaH_model = ALM.ALMBuyAndHold_2(problem)
BaH_model.solve()

sol = GB_model.solution
sol_bah = BaH_model.solution
#%% 
# Display Solution: init charts
colormap_ETF = {}
it = -1
for p in problem.P:
    it = it+1
    colormap_ETF[p] = px.colors.qualitative.Plotly[it%10]

n_scen = None
perc = False
showlegend = True
horizontal_spacing = 0.03
current_user_risk_profile = problem.user_risk_profile

fig1 = go.FigureWidget(make_subplots(rows = 1, cols = 2, shared_yaxes= True, horizontal_spacing = horizontal_spacing))
plt = ALMc.AssetAllocationChart(problem,sol,n_scen=n_scen, perc = perc)
#fig1 = go.FigureWidget(plt)
fig1.add_traces(data=plt.data, rows = 1, cols = 1)
plt = ALMc.AssetAllocationChart(problem,sol_bah,n_scen=n_scen, perc = perc, showlegend= False)
#fig1_bah = go.FigureWidget(plt)
fig1.add_traces(data=plt.data, rows = 1, cols = 2)

ALMc.standardized_chart(fig1, perc = perc, showlegend= showlegend)
legend_top_position(fig1)
#ALMc.standardized_chart(fig1_bah, perc = perc, showlegend= showlegend)

asset = ALMc.AssetSplitDetailsChart(problem, sol, "ETF", colormap_ETF)    
asset_bah = ALMc.AssetSplitDetailsChart(problem, sol_bah, "ETF", colormap_ETF)

prob, avg = ALMc.GoalRiskDetails(problem, sol, perc)
prob_bah, avg_bah = ALMc.GoalRiskDetails(problem, sol_bah, perc, showlegend = False)

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

success_prob, fail_prob, tot_wealth_avg, paid_advance_avg, avg_loss_in_failure = ALMc.EoPWealthInfo(problem, sol)
success_prob_bah, fail_prob_bah, tot_wealth_avg_bah, paid_advance_avg_bah, avg_loss_in_failure_bah = ALMc.EoPWealthInfo(problem, sol_bah)
plan_success_prob = plan_succes_prob_widget(success_prob,fail_prob)
plan_success_prob_bah = plan_succes_prob_widget(success_prob_bah,fail_prob_bah)
eop_wealth_summary = eop_wealth_summary_widget(int(tot_wealth_avg),int(paid_advance_avg),int(avg_loss_in_failure))
eop_wealth_summary_bah = eop_wealth_summary_widget(int(tot_wealth_avg_bah),int(paid_advance_avg_bah),int(avg_loss_in_failure_bah))

goal_based_summary = widgets.HTML(value='<br><br><b><h1><p style="text-align:center">Goal-based strategy</p></h1></b>', layout=widgets.Layout(display='flex', justify_content='center'))
BaH_summary = widgets.HTML(value='<br><br><b><h1><p style="text-align:center">Buy-and-hold strategy</p></h1></b>', layout=widgets.Layout(display='flex', justify_content='center'))

dashboard = GridBox(children=[goal_based_summary,fig1, BaH_summary,plan_success_prob,fig2,plan_success_prob_bah,eop_wealth_summary,fig3,eop_wealth_summary_bah],
        layout=Layout(
            width='100%',
            grid_template_rows='auto auto auto',
            grid_template_columns='25% 50% 25%',
            grid_template_areas=''' 
            "goal_based_summary fig1 BaH_summary" 
            "plan_success_prob fig2 plan_success_prob_bah" 
            "eop_wealth_summary fig3 eop_wealth_summary_bah"''')
       )
#%%
# Display Solution: interact manual widget

@interact_manual(
    percentual = perc, 
    BaH_strategy = widgets.IntSlider(
            range = [0,(len(buyandhold_portfolios)-1)], 
            value = problem.user_risk_profile, 
            style = {'description_width': 'initial'} 
        )
    )
def update(percentual,BaH_strategy):
    global current_user_risk_profile
    global sol
    global sol_bah
    perc = percentual
    ## 1 - Reset Charts
    with fig1.batch_update():
        n_index = int(len(fig1.data)/2)
        for i in np.arange(n_index):
            fig1.data[i].y = np.zeros(len(fig1.data[i].y))
            fig1.data[i+n_index].y = np.zeros(len(fig1.data[i+n_index].y))
        ALMc.standardized_chart(fig1, perc = perc)
    with fig2.batch_update():
        n_index = int(len(fig2.data)/2)
        for i in np.arange(n_index):
            fig2.data[i].y = np.zeros(len(fig2.data[i].y))
            fig2.data[i+n_index].y = np.zeros(len(fig2.data[i+n_index].y))
        ALMc.standardized_chart(fig2, perc = True, showlegend= True)
    with fig3.batch_update():
        n_index = int(len(fig3.data)/2)
        for i in np.arange(n_index):
            fig3.data[i].y = np.zeros(len(fig3.data[i].y))
            fig3.data[i+n_index].y = np.zeros(len(fig3.data[i+n_index].y))
        ALMc.standardized_chart(fig3, perc = perc, showlegend= True)
    plan_success_prob.value = plan_succes_prob_widget("-","-").value
    plan_success_prob_bah.value = plan_succes_prob_widget("-","-").value
    eop_wealth_summary.value = eop_wealth_summary_widget("-","-","-").value
    eop_wealth_summary_bah.value = eop_wealth_summary_widget("-","-","-").value
    
    # 2 - Update Charts
    if not(current_user_risk_profile == BaH_strategy):
        current_user_risk_profile = BaH_strategy
        sol.update_end(problem,buyandhold_portfolios[BaH_strategy])
        BaH_model2 = ALM.ALMBuyAndHold_2(problem)
        BaH_model2.solve(buyandhold_portfolios[BaH_strategy])
        sol_bah = BaH_model2.solution
    ## Update data with new input
    plt1 = ALMc.AssetAllocationChart(problem,sol,n_scen=n_scen, perc = perc, portfolio_strategy = buyandhold_portfolios[BaH_strategy])
    plt2 = ALMc.AssetAllocationChart(problem,sol_bah,n_scen=n_scen, perc = perc, showlegend= False, portfolio_strategy = buyandhold_portfolios[BaH_strategy])
    prob, avg = ALMc.GoalRiskDetails(problem, sol, perc)
    prob_bah, avg_bah = ALMc.GoalRiskDetails(problem, sol_bah, perc, showlegend = False)
    success_prob, fail_prob, tot_wealth_avg, paid_advance_avg, avg_loss_in_failure = ALMc.EoPWealthInfo(problem, sol)
    success_prob_bah, fail_prob_bah, tot_wealth_avg_bah, paid_advance_avg_bah, avg_loss_in_failure_bah = ALMc.EoPWealthInfo(problem, sol_bah)
    ## update chart with new output
    with fig1.batch_update():
        n_index = int(len(fig1.data)/2)
        for i in np.arange(n_index):
            fig1.data[i].y = plt1.data[i].y
            fig1.data[i+n_index].y = plt2.data[i].y
        ALMc.standardized_chart(fig1, perc = perc, showlegend= True)
    with fig2.batch_update():
        n_index = int(len(fig2.data)/2)
        for i in np.arange(n_index):
            fig2.data[i].y = prob.data[i].y
            fig2.data[i+n_index].y = prob_bah.data[i].y
        ALMc.standardized_chart(fig2, perc = True, showlegend= True)
    with fig3.batch_update():
        n_index = int(len(fig3.data)/2)
        for i in np.arange(n_index):
            fig3.data[i].y = avg.data[i].y
            fig3.data[i+n_index].y = avg_bah.data[i].y
        ALMc.standardized_chart(fig3, perc = perc, showlegend= True)
    plan_success_prob.value = plan_succes_prob_widget(success_prob,fail_prob).value
    plan_success_prob_bah.value = plan_succes_prob_widget(success_prob_bah,fail_prob_bah).value
    eop_wealth_summary.value = eop_wealth_summary_widget(int(tot_wealth_avg),int(paid_advance_avg),int(avg_loss_in_failure)).value
    eop_wealth_summary_bah.value = eop_wealth_summary_widget(int(tot_wealth_avg_bah),int(paid_advance_avg_bah),int(avg_loss_in_failure_bah)).value


display(dashboard)
# %%
