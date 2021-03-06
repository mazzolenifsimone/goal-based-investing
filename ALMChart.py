import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def standardized_chart(plt, perc = False, showlegend = False):
    plt.update_layout(
        paper_bgcolor = "white",
        showlegend=showlegend,
        plot_bgcolor="white",
        margin=dict(t=20,l=20,b=20,r=20))
    plt.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
    plt.update_yaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
    plt.update_xaxes(showline=True, linewidth=1.3, linecolor='black', mirror = True)
    plt.update_yaxes(showline=True, linewidth=1.3, linecolor='black', mirror = True)
    if perc:
        plt.update_yaxes(range=[0, 1])
    else:
        plt.update_yaxes(autorange = True)

def legend_top_position(plt):
    plt.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="center",
        x=0.5)
    )

def planner_chart(planner, bar_width=6):
    ETF_GBM = planner.__DF_Scenario__
    portfolio = pd.Series(planner.user_portfolio.values(),index = planner.user_portfolio.keys())
    userpf_capfact = np.dot(np.exp(ETF_GBM[planner.P]), portfolio)
    cap_factor_ptf = np.reshape(userpf_capfact, (int(len(userpf_capfact)/len(planner.N)),len(planner.N)))
    SMF = planner.StandardModelForm()
    Ass_val = SMF["Asset Invested"].fillna(0)
    Liab_val = SMF["Optimal Goal"].fillna(0)
    low_Liab_val = SMF["Minimum Goal"].fillna(0)
    up_capitalized_value_n = np.zeros(shape = (SMF.shape[0], len(planner.N)))
    med_capitalized_value_n = np.zeros(shape = (SMF.shape[0], len(planner.N)))
    low_capitalized_value_n = np.zeros(shape = (SMF.shape[0], len(planner.N)))
    for i in SMF["Month since Start"]:
        if i==0:
            up_capitalized_value_n[i,:] = Ass_val[i]
            med_capitalized_value_n[i,:] = Ass_val[i]
            low_capitalized_value_n[i,:] = Ass_val[i]
        else:
            up_capitalized_value_n[i,:] = np.maximum(up_capitalized_value_n[i-1,:] * cap_factor_ptf[i,:] + Ass_val[i] - Liab_val[i],0)
            med_capitalized_value_n[i,:] = np.maximum(med_capitalized_value_n[i-1,:] * cap_factor_ptf[i,:] + Ass_val[i] - (Liab_val[i]+low_Liab_val[i])/2,0)
            low_capitalized_value_n[i,:] = np.maximum(low_capitalized_value_n[i-1,:] * cap_factor_ptf[i,:] + Ass_val[i] - low_Liab_val[i], 0)
    med_capitalized_value = np.quantile(med_capitalized_value_n, 0.5,axis=1)
    up_capitalized_value = np.quantile(up_capitalized_value_n,0.95, axis=1)
    low_capitalized_value = np.quantile(low_capitalized_value_n,0.05, axis=1)
    fig = go.Figure(data = [
        go.Bar(x = SMF["Month since Start"], y=SMF["Asset Invested"], width = bar_width, marker_color = "royalblue", name = "assets invested",marker_line = dict(width = 1.5, color = "steelblue")),
        go.Bar(x = SMF["Month since Start"], y=-SMF["Optimal Goal"], width = bar_width,marker_color = "gold", name = "optimal goals",marker_line = dict(width = 1.5, color = "dimgray")),
        go.Bar(x = SMF["Month since Start"], y=-SMF["Minimum Goal"], width = bar_width,marker_color = "white", marker_opacity = 1, marker_line = dict(width = 2, color = "dimgray"), name = "minimum goals"),
        go.Scatter(x = SMF["Month since Start"], y=up_capitalized_value, fill = None, mode = "lines", line_color = "lightblue", name ="90% confidence band", showlegend = False, legendgroup = "bell"),
        go.Scatter(x = SMF["Month since Start"], y=low_capitalized_value, fill = "tonexty", mode = "lines", line_color = "lightblue", name = "90% confidence band", legendgroup = "bell"),
                    go.Scatter(x = SMF["Month since Start"], y=np.cumsum(Ass_val-Liab_val), mode = "lines", line_color = "black", name = "wealth consumption"),
        go.Scatter(x = SMF["Month since Start"], y=med_capitalized_value, mode = "lines", line_color = "blue", name = "median value"),
        ],
        layout = go.Layout(barmode = "overlay")
    )
    standardized_chart(fig)
    lowerlimit = min(-max(Liab_val),min(np.cumsum(Ass_val-Liab_val)))
    upperlimit = max(med_capitalized_value)
    margin = -lowerlimit*0.25
    fig.update_yaxes(range=[lowerlimit - margin , upperlimit+margin])
    fig.update_xaxes(title_text='Months since start')
    fig.update_yaxes(title_text='Wealth')
    return fig

def AssetAllocationChart(planner, solution, n_scen=None, perc = False, portfolio_strategy = None, showlegend=True):
    P = planner.P
    L = planner.liabilities.set
    T = planner.T
    if portfolio_strategy is None:
        portfolio_strategy = planner.user_portfolio
    Q_nscen = np.zeros(shape = (len(P),len(L)))
    Val_tl = {}
    index = -1
    label_period = [[k, v] for k,v in planner.assets.period.items()]
    label_value = [[k, v] for k,v in planner.assets.value.items()]
    period_value_df = pd.DataFrame(label_period, columns = ["Label", "Period"]).merge(pd.DataFrame(label_value, columns = ["Label", "Value"]))

    for p in P:
        index = index+1
        Val_tl[p] = np.zeros(shape = (len(T), len(L)))
        for l in np.arange(len(L)):
            for t in np.arange(len(T)):
                if n_scen is None:
                    cap_factor = np.exp(planner.Scenario_mu[p])
                else :
                    cap_factor = np.exp(planner.Scenario[p][n_scen][t])
                if t < planner.liabilities.period[L[l]]:
                    new_asset_label = period_value_df.loc[period_value_df.Period == t , "Label"].values
                    new_asset = [solution.asset_to_goal[a][L[l]][p] for a in new_asset_label]
                    if t==0:
                        Val_tl[p][t,l] = sum(new_asset)*cap_factor
                    else:
                        Val_tl[p][t,l] = (Val_tl[p][t-1,l] + sum(new_asset))*cap_factor

                elif t == planner.liabilities.period[L[l]]:
                    Q_nscen[index,l] = Val_tl[p][t-1,l]

    Val_end_t = {}
    index = -1
    for p in P:
        index = index+1
        Val_end_t[p] = np.zeros(shape = (len(T)))
        for t in np.arange(len(T)):
            if n_scen is None:
                cap_factor = np.exp(planner.Scenario_mu[p])
            else:
                cap_factor = np.exp(planner.Scenario[p][n_scen][t])
            ex_wealth = 0
            for l in np.arange(len(L)):
                if t == planner.liabilities.period[L[l]]:
                    ex_wealth = ex_wealth + max(sum(Q_nscen[:,l]) - planner.liabilities.value_tg[L[l]], 0)

            new_asset_label = period_value_df.loc[period_value_df.Period == t , "Label" ].values
            new_asset = [solution.asset_to_exwealth[a][p] for a in new_asset_label]
            ex_wealth_p = ex_wealth*portfolio_strategy[p]
            if t==0:
                Val_end_t[p][t] = sum(new_asset)*cap_factor
            else:
                Val_end_t[p][t] = (Val_end_t[p][t-1] + sum(new_asset) + ex_wealth_p)*cap_factor


    Val_t = {}
    for p in P:
        Val_t[p] = np.sum(Val_tl[p], axis = 1) + Val_end_t[p] 

    AssetAllocationNominal = pd.DataFrame(Val_t)
    AssetAllocationNominal[AssetAllocationNominal<0] = 0
    AAN_perc = AssetAllocationNominal.divide(AssetAllocationNominal.sum(axis=1), axis=0)

    if perc:
        AAN_perc = AAN_perc.reset_index()
        AAN_perc = AAN_perc.melt(id_vars=['index'], var_name='P', value_name='evo')
        AAChart = px.area(AAN_perc, x="index", y = "evo", color = "P")
        AAChart.update_xaxes(range=[AAN_perc["index"].min(), AAN_perc["index"].max()])
        AAChart.update_yaxes(range=[0, 1])
    else:
        AssetAllocationNominal = AssetAllocationNominal.reset_index()
        AssetAllocationNominal = AssetAllocationNominal.melt(id_vars=['index'], var_name='P', value_name='evo')
        AAChart = px.area(AssetAllocationNominal, x="index", y = "evo", color = "P")
    standardized_chart(AAChart, perc = perc, showlegend = showlegend)
    for i in np.arange(len(AAChart.data)):
        AAChart.data[i]["showlegend"] = showlegend
    legend_top_position(AAChart)
    return AAChart

def AssetAllocationComparison(AAChart_gb,AAChart_bah, perc = False):
    AACharts = go.FigureWidget(make_subplots(rows = 1, cols = 2, shared_yaxes= True, horizontal_spacing = 0.03))
    AACharts.add_traces(data=AAChart_gb.data, rows = 1, cols = 1)
    AACharts.add_traces(data=AAChart_bah.data, rows = 1, cols = 2)
    standardized_chart(AACharts, perc = perc, showlegend = True)
    legend_top_position(AACharts)
    return AACharts

def AssetSplitDetailsChart(planner, solution, groupby, colormap):
    P = planner.P
    A = planner.assets.set
    L = planner.liabilities.set
    Assets_split = pd.DataFrame(index = np.arange(len(P)*len(A)*(len(L)+1)), columns = ["Asset", "Goal", "ETF", "Value"])
    it= -1
    for a in A:
        for p in P:
            for l in L:
                it = it+1
                Assets_split["Asset"][it] = a
                Assets_split["Goal"][it] = l
                Assets_split["ETF"][it] = p
                Assets_split["Value"][it] = solution.asset_to_goal[a][l][p]
            it = it+1
            Assets_split["Asset"][it] = a
            Assets_split["Goal"][it] = "extra_wealth"
            Assets_split["ETF"][it] = p
            Assets_split["Value"][it] = solution.asset_to_exwealth[a][p]
    AssetGroupedBy = Assets_split[["Asset", groupby, "Value"]].groupby(by=[groupby, "Asset"]).sum().reset_index()
    AssetPivot = AssetGroupedBy.pivot(index = "Asset", columns=groupby, values = "Value")
    AssetPivot["Period"] = pd.Series(planner.assets.period)
    AssetPivot = AssetPivot.sort_values(by = "Period")
    data = []
    if groupby == "Goal":
        groupby_set = list(L)
        groupby_set.append("extra_wealth")
    elif groupby == "ETF":
        groupby_set = P
    else:
        return
    data = []
    for e in groupby_set:
        data.append(go.Bar(x = AssetPivot.index, y = AssetPivot[e], marker_color = colormap[e], name = e))
    ASDChart = go.FigureWidget(data = data, layout = go.Layout(barmode = "stack"))
    standardized_chart(ASDChart)
    legend_top_position(ASDChart)
    return ASDChart

def GoalRiskDetails(planner, solution, perc, showlegend = True):
    L = planner.liabilities.set
    N = planner.N
    # Compute DF structure from solution
    Q_ln = {}
    for l in L:
        Q_ln[l] = np.zeros(shape = (len(N)))
        for n in N:
            Q_ln[l][n]=solution.goal_distr[l][N[n]]
    Q_ln["ex_wealth"] = np.zeros(shape = (len(N)))
    for n in N:
        Q_ln["ex_wealth"][n]=solution.final_exwealth[N[n]]
    df_Q_ln = pd.DataFrame(Q_ln)
    # Goal Shortfall Prob Chart
    df = pd.DataFrame(index = planner.liabilities.set)
    #  - DF for matrix comparison
    conf_tg = pd.DataFrame(planner.liabilities.value_tg, columns = planner.liabilities.set, index = N)
    conf_lb = pd.DataFrame(planner.liabilities.value_lb, columns = planner.liabilities.set, index = N)
    #  - Affordable Shortfall and Failure Shortfall Probabilities by goals
    df["Affordable"] = np.logical_and(df_Q_ln[planner.liabilities.set] < conf_tg, df_Q_ln[planner.liabilities.set] >= conf_lb).mean()
    df["Failure"] = (df_Q_ln[planner.liabilities.set] < conf_lb).mean()
    GSPChart = go.Figure(
        data = [
                go.Bar(x = df.index, y=df["Affordable"], marker_color = "royalblue", name = "Affordable Shortfall", showlegend = showlegend, legendgroup = "1"),#,marker_line = dict(width = 1.5, color = "slategray")),
                go.Bar(x = df.index, y=df["Failure"], marker_color = "crimson", name = "Failure Shortfall", showlegend = showlegend, legendgroup = "2"),
                ],
        layout = go.Layout(barmode = "stack")
            )
    standardized_chart(GSPChart, perc = True, showlegend=showlegend)
    legend_top_position(GSPChart)
    # Goal Avg and Worst when shortfall
    df = pd.DataFrame(index = planner.liabilities.set)
    #  - Goal value 
    df["goal"] = pd.Series(planner.liabilities.value_tg, index = planner.liabilities.set)
    #  - Average wealth obtained (or # Average wealth obtained in case of shortfall)
    df["avg"] = np.round(df_Q_ln[planner.liabilities.set].mean(),0)
    #  - Goal lower bound
    df["lower_bound"] = pd.Series(planner.liabilities.value_lb, index = planner.liabilities.set)
    #  - Average worst 5% wealth obtained
    df["worst"] = np.round(df_Q_ln[planner.liabilities.set][(df_Q_ln[planner.liabilities.set] <= np.quantile(df_Q_ln[planner.liabilities.set],0.05,axis = 0))].mean(),0)
    if perc:
        df = np.divide(df.T, df["goal"], axis = 1).T
    GAWChart = go.Figure(
        data = [
                go.Bar(x = df.index, y=df["goal"], marker_color = "gold", name = "Optimal Goal", marker_line = dict(width = 1.5, color = "slategray"), hoverinfo = "skip", showlegend = showlegend, legendgroup = "1"),  
                go.Bar(x = df.index, y=df["avg"], marker_color = "limegreen", name ="Expected payoff", hoverinfo = "skip",legendgroup = "2",showlegend=showlegend),
                go.Bar(x = df.index, y=df["worst"], marker_color = "red", name ="Average worst 5% payoff", hoverinfo = "skip",legendgroup = "3",showlegend=showlegend),
                go.Bar(x = df.index, y=df["lower_bound"], marker_color = "white", marker_opacity = 1, name = "Minimum Goal", marker_line = dict(width = 2, color = "slategray"), hoverinfo = "skip",legendgroup = "1", showlegend=showlegend),
                go.Scatter(x = df.index, y=df["goal"], mode = "markers", name = "Optimal Goal", marker_color = "slategray", showlegend=False,legendgroup = "1"),
                go.Scatter(x = df.index, y=df["lower_bound"], mode = "markers", name = "Minimum Goal", marker_color = "slategray", showlegend=False, legendgroup = "1"),
                go.Scatter(x = df.index, y=df["avg"], mode = "markers", name ="Expected payoff", marker_color = "limegreen", showlegend=False,legendgroup = "2"),
                go.Scatter(x = df.index, y=df["worst"], mode = "markers", name ="Average worst 5% payoff", marker_color = "red", showlegend=False,legendgroup = "3")
                ],
        layout = go.Layout(barmode = "overlay")
            )
    standardized_chart(GAWChart, perc = perc, showlegend = showlegend)
    legend_top_position(GAWChart)
    return GSPChart, GAWChart

def EoPWealthInfo(planner, solution):
    L = planner.liabilities.set
    N = planner.N
    # Compute DF structure from solution
    Q_ln = {}
    for l in L:
        Q_ln[l] = np.zeros(shape = (len(N)))
        for n in N:
            Q_ln[l][n] = solution.goal_distr[l][n]
    Q_ln["ex_wealth"] = np.zeros(shape = (len(N)))
    for n in N:
        Q_ln["ex_wealth"][n]=solution.final_exwealth[n]
    df_Q_ln = pd.DataFrame(Q_ln)
    conf_lb = pd.DataFrame(planner.liabilities.value_lb, columns = planner.liabilities.set, index = N)
    # Success\Failure probability (if a goal fails, plan fails)
    fail_prob = np.round(np.any((df_Q_ln[planner.liabilities.set] < conf_lb), axis = 1).mean()*100, 1)
    success_prob = 100 - fail_prob
    # Wealth Statistics
    tot_wealth_avg = np.round(np.mean(np.sum(df_Q_ln, axis = 1)),0)
    paid_advance_avg = np.round(np.mean(np.sum(df_Q_ln[L], axis = 1)),0)
    #extra_wealth_avg = np.round(np.mean(df_Q_ln["ex_wealth"]),0)
    fail_loss = np.minimum(df_Q_ln[planner.liabilities.set]-conf_lb,0)
    loss_in_failure = fail_loss.sum(axis = 1)
    avg_loss_in_failure = loss_in_failure[loss_in_failure < 0].mean()
    return success_prob, fail_prob, tot_wealth_avg, paid_advance_avg, avg_loss_in_failure