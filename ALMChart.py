import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def standardized_chart(plt):
    plt.update_layout(
        paper_bgcolor = "white",
        showlegend=True,
        plot_bgcolor="white",
        margin=dict(t=20,l=20,b=20,r=20))
    plt.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
    plt.update_yaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
    plt.update_xaxes(showline=True, linewidth=1.3, linecolor='black', mirror = True)
    plt.update_yaxes(showline=True, linewidth=1.3, linecolor='black', mirror = True)
    return plt

def display(planner, bar_width=6):
        ETF_GBM = planner.__DF_Scenario__

        portfolio = pd.Series(planner.user_portfolio.values(),index = planner.user_portfolio.keys())
        userpf_capfact = np.dot(np.exp(ETF_GBM[planner.P]), portfolio)
        cap_factor_ptf = np.reshape(userpf_capfact, (int(len(userpf_capfact)/len(planner.N)),len(planner.N)))

        SMF = planner.StandardModelForm()

        Ass_val = SMF["Asset Value"]
        Liab_val = SMF["Goal Value"]
        low_Liab_val = SMF["Goal Lower Bound"]

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
        plot_df = pd.DataFrame()
        
        plot_df["x"] = SMF["Month since Start"]
        plot_df["ass"] = SMF["Asset Value"]
        plot_df["liab"] = -SMF["Goal Value"]
        plot_df["cumsum"] = np.cumsum(plot_df["ass"]+plot_df["liab"])
        plot_df["up_cap"] = up_capitalized_value
        plot_df["low_cap"] = low_capitalized_value
        plot_df["med_cap"] = med_capitalized_value
        
        fig = go.Figure(data = [
            go.Scatter(x = SMF["Month since Start"], y=up_capitalized_value, fill = None, mode = "lines", line_color = "lightblue", name ="95% quantile"),
            go.Scatter(x = SMF["Month since Start"], y=low_capitalized_value, fill = "tonexty", mode = "lines", line_color = "lightblue", name = "5% quantile"),
            go.Scatter(x = SMF["Month since Start"], y=med_capitalized_value, mode = "lines", line_color = "blue", name = "50% quantile"),
            go.Bar(x = SMF["Month since Start"], y=SMF["Asset Value"], width = bar_width, marker_color = "lightgreen", name = "Assets"),
            go.Bar(x = SMF["Month since Start"], y=-SMF["Goal Value"], width = bar_width,marker_color = "red", name = "Goals"),
            go.Scatter(x = SMF["Month since Start"], y=plot_df["cumsum"], mode = "lines", line_color = "black", name = "Neutral wealth consumption")
        ])

        fig = standardized_chart(fig) 
        fig.update_layout(showlegend=False)
        fig.show()
        return