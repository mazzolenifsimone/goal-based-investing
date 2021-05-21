import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def standardized_chart(plt, perc = False):
    plt.update_layout(
        paper_bgcolor = "white",
        showlegend=True,
        plot_bgcolor="white",
        margin=dict(t=20,l=20,b=20,r=20))
    plt.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
    plt.update_yaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
    plt.update_xaxes(showline=True, linewidth=1.3, linecolor='black', mirror = True)
    plt.update_yaxes(showline=True, linewidth=1.3, linecolor='black', mirror = True)
    if perc:
        plt.update_yaxes(range=[0, 1])
    return plt

def display(planner, bar_width=6):
        ETF_GBM = planner.__DF_Scenario__

        portfolio = pd.Series(planner.user_portfolio.values(),index = planner.user_portfolio.keys())
        userpf_capfact = np.dot(np.exp(ETF_GBM[planner.P]), portfolio)
        cap_factor_ptf = np.reshape(userpf_capfact, (int(len(userpf_capfact)/len(planner.N)),len(planner.N)))

        SMF = planner.StandardModelForm()

        Ass_val = SMF["Asset Value"].fillna(0)
        Liab_val = SMF["Goal Value"].fillna(0)
        low_Liab_val = SMF["Goal Lower Bound"].fillna(0)

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
            go.Scatter(x = SMF["Month since Start"], y=up_capitalized_value, fill = None, mode = "lines", line_color = "lightblue", name ="95% quantile"),
            go.Scatter(x = SMF["Month since Start"], y=low_capitalized_value, fill = "tonexty", mode = "lines", line_color = "lightblue", name = "5% quantile"),
            go.Scatter(x = SMF["Month since Start"], y=med_capitalized_value, mode = "lines", line_color = "blue", name = "50% quantile"),
            go.Bar(x = SMF["Month since Start"], y=SMF["Asset Value"], width = bar_width, marker_color = "royalblue", name = "Assets",marker_line = dict(width = 1.5, color = "steelblue")),
            go.Bar(x = SMF["Month since Start"], y=-SMF["Goal Value"], width = bar_width,marker_color = "gold", name = "Goals",marker_line = dict(width = 1.5, color = "dimgray")),
            go.Bar(x = SMF["Month since Start"], y=-SMF["Goal Lower Bound"], width = bar_width,marker_color = "white", marker_opacity = 1, marker_line = dict(width = 2, color = "dimgray"), name = "Goal LB"),
            go.Scatter(x = SMF["Month since Start"], y=np.cumsum(Ass_val-Liab_val), mode = "lines", line_color = "black", name = "Neutral wealth consumption")
            ],
            layout = go.Layout(barmode = "overlay")
        )

        fig = standardized_chart(fig)
        lowerlimit = min(-max(Liab_val),min(np.cumsum(Ass_val-Liab_val)))
        upperlimit = max(med_capitalized_value)
        margin = -lowerlimit*0.25
        fig.update_yaxes(range=[lowerlimit - margin , upperlimit+margin])
        fig.update_layout(showlegend=False)
        fig.show()
        return

def AssetAllocationChart(planner, solution, n_scen=None, perc = False):
    P = planner.P
    L = planner.liabilities.set
    T = planner.T
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
            ex_wealth_p = ex_wealth*planner.user_portfolio[p]
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
        AAChart = px.area(AAN_perc, x="index", y = "evo", color = "P" )
        AAChart.update_xaxes(range=[AAN_perc["index"].min(), AAN_perc["index"].max()])
        AAChart.update_yaxes(range=[0, 1])
    else:
        AssetAllocationNominal = AssetAllocationNominal.reset_index()
        AssetAllocationNominal = AssetAllocationNominal.melt(id_vars=['index'], var_name='P', value_name='evo')
        AAChart = px.area(AssetAllocationNominal, x="index", y = "evo", color = "P")
    AAChart = standardized_chart(AAChart)
    return AAChart

def AssetSplitDetailsChart(planner, solution, groupby):
    P = planner.P
    A = planner.assets.set
    L = planner.liabilities.set
    Assets_split = pd.DataFrame(index = np.arange(len(P)*len(A)*(len(L)+1)), columns = ["Asset", "Goal", "ETF", "Value"])
    iter = -1
    for a in A:
        for p in P:
            for l in L:
                iter = iter+1
                Assets_split["Asset"][iter] = a
                Assets_split["Goal"][iter] = l
                Assets_split["ETF"][iter] = p
                Assets_split["Value"][iter] = solution.asset_to_goal[a][l][p]
            iter = iter+1
            Assets_split["Asset"][iter] = a
            Assets_split["Goal"][iter] = "extra_wealth"
            Assets_split["ETF"][iter] = p
            Assets_split["Value"][iter] = solution.asset_to_exwealth[a][p]
    AssetGroupedBy = Assets_split[["Asset", groupby, "Value"]].groupby(by=[groupby, "Asset"]).sum().reset_index()
    ASDChart = px.bar(AssetGroupedBy, x="Asset", y = "Value", color = groupby)
    ASDChart = standardized_chart(ASDChart)
    return ASDChart

def GoalRiskDetails(planner, solution, perc):
    L = planner.liabilities.set
    N = planner.N
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
    conf_tg = pd.DataFrame(planner.liabilities.value_tg, columns = planner.liabilities.set, index = N)
    conf_lb = pd.DataFrame(planner.liabilities.value_lb, columns = planner.liabilities.set, index = N)
    df["Affordable"] = np.logical_and(df_Q_ln[planner.liabilities.set] < conf_tg, df_Q_ln[planner.liabilities.set] >= conf_lb).mean()
    df["Failure"] = (df_Q_ln[planner.liabilities.set] < conf_lb).mean()
    df = df.reset_index().rename(columns = {"index":"Goals"})
    dfchart = pd.melt(df, id_vars=["Goals"], value_vars = ["Affordable", "Failure"], var_name = "Shortfall Cathegory", value_name = "Shortfall Probabilities")
    GSPChart = px.bar(dfchart, x="Goals", y = "Shortfall Probabilities", color = "Shortfall Cathegory",color_discrete_sequence=['royalblue', "crimson"])
    GSPChart = standardized_chart(GSPChart, perc = True)
    # Goal Avg and Worst when shortfall
    df = pd.DataFrame(index = planner.liabilities.set)
    df["goal"] = pd.Series(planner.liabilities.value_tg, index = planner.liabilities.set)
    df["avg"] = np.round(df_Q_ln[planner.liabilities.set][(df_Q_ln[planner.liabilities.set] < conf_tg[planner.liabilities.set])].mean(),0)
    df["avg_shortfall"] = df["goal"] - df["avg"]
    df["lower_bound"] = pd.Series(planner.liabilities.value_lb, index = planner.liabilities.set)
    df["worst"] = np.round(df_Q_ln[planner.liabilities.set][(df_Q_ln[planner.liabilities.set] <= np.quantile(df_Q_ln[planner.liabilities.set],0.05,axis = 0))].mean(),0)
    if perc:
        df = np.divide(df.T, df["goal"], axis = 1).T
    GAWChart = go.Figure(
        data = [
                go.Bar(x = df.index, y=df["goal"], marker_color = "gold", name = "Goal",marker_line = dict(width = 1.5, color = "slategray")),
                go.Bar(x = df.index, y=df["avg"], marker_color = "limegreen", name = "Average Value"),
                go.Bar(x = df.index, y=df["lower_bound"], marker_color = "white", marker_opacity = 1, marker_line = dict(width = 2, color = "slategray"), name = "LB Value"),
                go.Scatter(x = df.index, y=df["worst"], mode = "markers", name ="worst 5%", marker_color = "red")
                ],
        layout = go.Layout(barmode = "overlay")
            )
    GAWChart = standardized_chart(GAWChart, perc)

    return GSPChart, GAWChart