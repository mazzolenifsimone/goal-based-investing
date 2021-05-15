import numpy as np
import pandas as pd
import pulp as lp
# import matplotlib.pyplot as plt
import pickle as pkl
# import datetime as dt
import time
import os
import plotly.express as px
import plotly.graph_objects as go

buyandhold_portfolio = {
    0:{"Cash":0.02, "GVT_EU13":0.39, "GVT_EU57":0.39, "EM":0.04, "EQ_EU":0.08, "EQ_US":0.08}, 
    1:{"Cash":0, "GVT_EU13":0.25, "GVT_EU57":0.25, "EM":0.10, "EQ_EU":0.20, "EQ_US":0.20}, 
    2:{"Cash":0, "GVT_EU13":0.10, "GVT_EU57":0.10, "EM":0.16, "EQ_EU":0.32, "EQ_US":0.32},
}


class ALMplanner:

    def __init__(self, start = np.datetime64("2021-01-01"), end = np.datetime64("2070-12-31"), path_scenario = "scenario", scen_name = "Scenario", scen_df_name = "ETF_GBM", user_risk_profile = 1, buyandhold_portfolio = buyandhold_portfolio):
        self.start = start
        self.end = end
        self.T = pd.date_range(start = start, end=end, freq = "M")
        self.Scenario = load_scenario(path_scenario, scen_name)
        self.P = list(self.Scenario.keys())
        self.N = list(self.Scenario[self.P[0]].keys())
        self.__DF_Scenario__ = load_scenario(path_scenario,scen_df_name)
        self.Scenario_mu = self.__DF_Scenario__[self.P].mean()
        self.Scenario_sigma = self.__DF_Scenario__[self.P].cov()
        self.user_portfolio = buyandhold_portfolio[user_risk_profile]
        self.feasibility = -1
        self.liabilities = ALMLiability(self)
        self.assets = ALMAssets(self)
        return

    def generate_GB_model(self):
        tic = time.time()
        self.GB_model = ALMGoalBased(self)
        print(f"GoalBased model generated in {np.round(time.time()-tic,2)} s")
        return

    def generate_BaH_model(self):
        tic = time.time()
        self.BaH_model = ALMBuyAndHold(self)
        print(f"BuyAndHold model generated in {np.round(time.time()-tic,2)} s")
        return

    def check_feasibility(self):
        tic = time.time()
        HigherAsset = ALMCheckFeasibility(self)
        status = HigherAsset.formulation.solve()
        print(f"check feasibility ended in {np.round(time.time()-tic,2)} s with {lp.constants.LpStatus[status]} solution")
        self.feasibility_coeff = HigherAsset.mod_asset.varValue
        if self.feasibility_coeff > 0:
            self.feasibility = 0
            print(f"Unfeasible problem: suggested +{np.round(self.feasibility_coeff*100,2)}%  on assets")
        else:
            self.feasibility = 1
            print(f"Feasible problem")
        return

    def get_feasibility(self):
        if self.feasibility > -1:
            for label in self.assets.set:
                self.assets.value[label] = self.assets.value[label]*(1+self.feasibility_coeff)
        else:
            print("ERROR: No feasiblity information")
        return

    def solve_BAH(self):
        tic = time.time()
        status = self.BaH_model.formulation.solve()
        print(f"Solve ended in {np.round(time.time()-tic,2)} s with {lp.constants.LpStatus[status]} solution")
        self.solution_BAH = ALMSolution(status)
        A = self.assets.set
        L = self.liabilities.set
        P = self.P
        N = self.N

        if status == 1:
            for a in A:
                self.solution_BAH.asset_part[a] = {}
                for l in L:
                    self.solution_BAH.asset_part[a][l] = {}
                    for p in P:
                        self.solution_BAH.asset_part[a][l][p] = self.BaH_model.W[a][l][p].varValue
            for a in A:
                self.solution_BAH.asset_end_part[a]={}
                for p in P:
                    self.solution_BAH.asset_end_part[a][p] = self.BaH_model.W_end[a][p].varValue
            for l in L:
                self.solution_BAH.liab_distr[l] = {}
                self.solution_BAH.V_distr[l] = {}
                self.solution_BAH.ex_wealth[l] = {}
                for n in N:
                    self.solution_BAH.liab_distr[l][n] = self.BaH_model.Q[l][n].varValue
                    self.solution_BAH.ex_wealth[l][n] = self.BaH_model.Q_ex[l][n].varValue
            for n in N:
                self.solution_BAH.liab_end_distr[n] = self.BaH_model.Q_end[n].varValue
        
        return
    
    def solve(self):
        tic = time.time()
        status = self.GB_model.formulation.solve()
        print(f"Solve ended in {np.round(time.time()-tic,2)} s with {lp.constants.LpStatus[status]} solution")
        self.solution = ALMSolution(status)
        A = self.assets.set
        L = self.liabilities.set
        P = self.P
        N = self.N

        if status == 1:
            for a in A:
                self.solution.asset_part[a] = {}
                for l in L:
                    self.solution.asset_part[a][l] = {}
                    for p in P:
                        self.solution.asset_part[a][l][p] = self.GB_model.W[a][l][p].varValue
            for a in A:
                self.solution.asset_end_part[a]={}
                for p in P:
                    self.solution.asset_end_part[a][p] = self.GB_model.W_end[a][p].varValue
            for l in L:
                self.solution.liab_distr[l] = {}
                self.solution.V_distr[l] = {}
                self.solution.ex_wealth[l] = {}
                for n in N:
                    self.solution.liab_distr[l][n] = self.GB_model.Q[l][n].varValue
                    ###
                    self.solution.ex_wealth[l][n] = self.GB_model.Q_ex[l][n].varValue
                    ###
                    self.solution.V_distr[l][n] = self.GB_model.V[l][n].varValue
            for l in L:
                self.solution.VaR_liab[l] = self.GB_model.gamma[l].varValue
            for n in N:
                self.solution.liab_end_distr[n] = self.GB_model.Q_end[n].varValue
        return



class ALMLiability():

    def __init__(self, planner):
        self.set = []
        self.value_tg = {}
        self.value_lb = {}
        self.cvar_lim = {}
        self.period = {}
        self.date = {}
        self.start = planner.start
        return
    
    def insert(self, label, date, value_tg, value_lb, cvar_lim = 0.95):
        #self.set.add(label)
        self.date[label] = date
        self.set = sorted(self.date.keys(), key = lambda x: self.date[x])
        date_dt = pd.to_datetime(date)
        start_dt = pd.to_datetime(self.start)
        self.period[label] = (date_dt.year - start_dt.year)*12 + date_dt.month - start_dt.month 
        self.value_tg[label] = value_tg
        self.value_lb[label] = value_lb
        self.cvar_lim[label] = cvar_lim
        return

    def lists(self):
        ListOfLiabilities = pd.DataFrame(index=self.set, columns = ["Date", "Month since Start", "Target Liability", "Lowerbound Liability", "CVaR Level"])
        for i in ListOfLiabilities.index:
            ListOfLiabilities["Date"][i] = self.date[i]
            ListOfLiabilities["Month since Start"][i] = self.period[i]
            ListOfLiabilities["Target Liability"][i] = self.value_tg[i]
            ListOfLiabilities["Lowerbound Liability"][i] = self.value_lb[i]
            ListOfLiabilities["CVaR Level"][i] = self.cvar_lim[i]
        return ListOfLiabilities.sort_values(by = "Date")



class ALMAssets():

    def __init__(self, planner):
        self.set = []
        self.value= {}
        self.date = {}
        self.period = {}
        self.start = planner.start
        return
    
    def insert(self, label, date, value):
        self.date[label] = date
        self.set = sorted(self.date.keys(), key = lambda x: self.date[x])
        self.value[label] = value
        date_dt = pd.to_datetime(date)
        start_dt = pd.to_datetime(self.start)
        self.period[label] = (date_dt.year - start_dt.year)*12 + date_dt.month - start_dt.month
        return

    def lists(self):
        ListOfAsset = pd.DataFrame(index=self.set, columns = ["Date", "Month since Start", "Asset Value"])
        for i in ListOfAsset.index:
            ListOfAsset["Asset Value"][i] = self.value[i]
            ListOfAsset["Date"][i] = self.date[i]
            ListOfAsset["Month since Start"][i] = self.period[i]
        return ListOfAsset.sort_values(by = "Date")



class ALMCheckFeasibility():

    def __init__(self, planner):        
        self.formulation = lp.LpProblem(name = "ALMCheckFeasibility", sense = lp.LpMinimize)

        # Variables
        set_variables(self, planner)
        self.mod_liab = lp.LpVariable(name = "mod_liab", lowBound = 0, upBound= 1, cat = "Continuous")
        self.mod_asset = lp.LpVariable(name = "mod_asset", lowBound = 0, cat = "Continuous")
        
        # Objective Function
        self.formulation += self.mod_asset

        # Constraints
        # - Asset constraints GoalBased
        add_GB_asset_constr(self, planner, mod_asset = self.mod_asset)
        # - Liabilities Projection constraints
        add_portfolio_evolution_constr(self, planner, mod_liab = self.mod_liab)
        # - Extra wealth management constraints
        add_extrawealth_mgmt_constr(self, planner)
        # - Cvar Liabilities constraints
        add_GB_risk_constr(self, planner, mod_liab = self.mod_liab)
        # - TEMP: fix liab
        self.formulation += self.mod_liab == 0
    
        return


class ALMGoalBased():

    def __init__(self, planner):
        self.formulation = lp.LpProblem(name = "ALMGoalBased", sense = lp.LpMaximize)

        # Variables
        set_variables(self, planner)

        # Objective Function
        add_objective_function(self, planner)
        
        # Constraints
        # - Asset constraints GoalBased
        add_GB_asset_constr(self, planner)
        # - Liabilities Projection constraints
        add_portfolio_evolution_constr(self, planner)
        # - Extra wealth management constraints
        add_extrawealth_mgmt_constr(self, planner)
        # - Cvar Liabilities constraints
        add_GB_risk_constr(self, planner)
        return


class ALMBuyAndHold():

    def __init__(self, planner):
        self.formulation = lp.LpProblem(name = "ALMbuyandhold", sense = lp.LpMaximize)

        # Variables
        set_variables(self, planner)
        
        # Objective Function
        add_objective_function(self, planner)
        
        # Constraints
        # - Asset constraints GoalBased
        add_BaH_asset_constr(self, planner)
        # - Liabilities Projection constraints
        add_portfolio_evolution_constr(self, planner)
        # - Extra wealth management constraints
        add_extrawealth_mgmt_constr(self, planner)    
        return


class ALMSolution():
    
    def __init__(self, status):
        self.status = status
        self.asset_part = {}
        self.liab_distr = {}
        self.VaR_liab = {}
        self.V_distr = {}
        self.Z_distr = {}
        self.asset_end_part = {}
        self.liab_end_distr = {}
        self.ex_wealth = {}
        return



## FUNCTIONS

def load_scenario(path_scenario, scen_name):
    scen_path = os.path.join(path_scenario,scen_name)
    scenario_file = open(scen_path + ".pkl", "rb")
    scenario = pkl.load(scenario_file)
    scenario_file.close()
    return scenario

def set_variables(model, planner):
    model.W = lp.LpVariable.dicts(name = "W", indexs = (planner.assets.set,planner.liabilities.set,planner.P), lowBound = 0, cat = "Continuous")
    model.Q = lp.LpVariable.dicts(name = "Q", indexs = (planner.liabilities.set,planner.N), lowBound = 0, cat = "Continuous")
    model.Q_ex = lp.LpVariable.dicts(name = "Q_ex", indexs = (planner.liabilities.set,planner.N), lowBound = 0, cat = "Continuous")
    model.gamma = lp.LpVariable.dicts(name = "gamma", indexs = (planner.liabilities.set), cat = "Continuous")
    model.V = lp.LpVariable.dicts(name = "V", indexs = (planner.liabilities.set,planner.N), lowBound = 0, cat = "Continuous")
    model.W_end = lp.LpVariable.dicts(name = "W_end", indexs = (planner.assets.set,planner.P), lowBound = 0, cat = "Continuous")
    model.Q_end = lp.LpVariable.dicts(name = "Q_end", indexs = (planner.N), lowBound = 0, cat = "Continuous")

def add_objective_function(model, planner):
    model.formulation += lp.lpSum(model.Q[l][n] for l in planner.liabilities.set for n in planner.N)/len(planner.N) + lp.lpSum(model.W_end[a][p] for a in planner.assets.set for p in planner.P)
    return

def add_GB_asset_constr(model, planner, mod_asset = 0):
    for a in planner.assets.set:
            L_feas = np.array(list(planner.liabilities.period.keys()))[np.array(list(planner.liabilities.period.values()))>planner.assets.period[a]]
            L_unfeas = np.array(list(planner.liabilities.period.keys()))[np.array(list(planner.liabilities.period.values()))<=planner.assets.period[a]]
            model.formulation += lp.lpSum(model.W[a][l][p] for l in L_feas for p in planner.P) + lp.lpSum(model.W_end[a][p] for p in planner.P) == (1+mod_asset)*planner.assets.value[a]
            model.formulation += lp.lpSum(model.W[a][l][p] for l in L_unfeas for p in planner.P) == 0
            for p in planner.P:
                model.formulation += model.W_end[a][p] == lp.lpSum(model.W_end[a][p] for p in planner.P)*planner.user_portfolio[p]
    return

def add_BaH_asset_constr(model, planner):
    Asset_al_split, Asset_aend_split = smart_asset_allocation(planner)
    for a in planner.assets.set:
        L_feas = np.array(list(planner.liabilities.period.keys()))[np.array(list(planner.liabilities.period.values()))>planner.assets.period[a]]
        L_unfeas = np.array(list(planner.liabilities.period.keys()))[np.array(list(planner.liabilities.period.values()))<=planner.assets.period[a]]
        model.formulation += lp.lpSum(model.W[a][l][p] for l in L_unfeas for p in planner.P) == 0
        for p in planner.P:
            model.formulation += model.W_end[a][p] == Asset_aend_split[a]*planner.user_portfolio[p]
            for l in L_feas:
                model.formulation += model.W[a][l][p] == Asset_al_split[l][a]*planner.user_portfolio[p]
    return

def add_portfolio_evolution_constr(model, planner, mod_liab = 0):
    for l in planner.liabilities.set:
        for n in planner.N:
            model.formulation += model.Q[l][n] == lp.lpSum(model.W[a][l][p]*np.exp(np.sum(planner.Scenario[p][n][planner.assets.period[a]:planner.liabilities.period[l]])) for a in planner.assets.set for p in planner.P) - model.Q_ex[l][n]
            model.formulation += model.Q[l][n] <= planner.liabilities.value_tg[l]*(1-mod_liab)
    return

def add_extrawealth_mgmt_constr(model, planner):
    for n in planner.N:
        model.formulation += model.Q_end[n] == lp.lpSum(model.W_end[a][p]*np.exp(np.sum(planner.Scenario[p][n][planner.assets.period[a]:])) for a in planner.assets.set for p in planner.P) + lp.lpSum(model.Q_ex[l][n]*planner.user_portfolio[p]*np.exp(np.sum(planner.Scenario[p][n][planner.liabilities.period[l]:])) for l in planner.liabilities.set for p in planner.P)
    return

def add_GB_risk_constr(model, planner, mod_liab = 0):
    for l in planner.liabilities.set:
        model.formulation += model.gamma[l] + lp.lpSum(model.V[l][n] for n in planner.N)/(len(planner.N)*(1-planner.liabilities.cvar_lim[l])) <= (planner.liabilities.value_tg[l] - planner.liabilities.value_lb[l])*(1-mod_liab)
        for n in planner.N:
            model.formulation += model.V[l][n] >= planner.liabilities.value_tg[l]*(1-mod_liab) - model.Q[l][n] - model.gamma[l]
    return

def smart_asset_allocation(planner):
    Assets_list = planner.assets.lists()
    Liabs_list = planner.liabilities.lists()
    
    Lt = Liabs_list["Month since Start"]
    L = Liabs_list.index
    Liab_tg = Liabs_list["Target Liability"]
    Liab_lb = Liabs_list["Lowerbound Liability"]
    At = Assets_list["Month since Start"]
    A = Assets_list.index
    Assets = Assets_list["Asset Value"]

    asset_split = pd.DataFrame(0,columns = L, index = A)
    total_asset = Assets.sum()
    total_liab_lb = Liab_lb.sum()
    budget = {}
    
    for l in L:
        budget[l] = min(Liab_lb[l], Liab_lb[l]/total_liab_lb*total_asset)
    
    for a in A:
        L_feas = Lt[Lt>At[a]]
        for l in L_feas.index:
            asset_split[l][a] = min(max(Assets[a] - asset_split.loc[a,:].T.sum(),0), budget[l] - asset_split[l].sum())

    asset_end = Assets - asset_split.T.sum()
    return asset_split, asset_end

### CHARTS

def standardized_chart(plt):
    plt.update_layout(
        paper_bgcolor = "white",
        showlegend=True,
        plot_bgcolor="white",
        margin=dict(t=20,l=20,b=20,r=20)
    )
    plt.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
    plt.update_yaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
    plt.update_xaxes(showline=True, linewidth=1.3, linecolor='black', mirror = True)
    plt.update_yaxes(showline=True, linewidth=1.3, linecolor='black', mirror = True)
    return plt

def display(planner, bar_width):
        ETF_GBM = planner.__DF_Scenario__

        portfolio = pd.Series(planner.user_portfolio.values(),index = planner.user_portfolio.keys())
        userpf_capfact = np.dot(np.exp(ETF_GBM[planner.P]), portfolio)
        cap_factor_ptf = np.reshape(userpf_capfact, (int(len(userpf_capfact)/len(planner.N)),len(planner.N)))

        T = pd.date_range(planner.start, planner.end, freq="M")
        month = np.arange(len(T))

        Assets = planner.assets.lists()
        Liabilities = planner.liabilities.lists()

        SMF = pd.DataFrame({"Date":T,"Month since Start":month}).merge(Assets[["Month since Start","Asset Value"]], how = "left", on = "Month since Start").merge(Liabilities[["Month since Start","Target Liability", "Lowerbound Liability", "CVaR Level"]], how = "left", on = "Month since Start").fillna(0).reset_index()

        Ass_val = SMF["Asset Value"]
        Liab_val = SMF["Target Liability"]
        low_Liab_val = SMF["Lowerbound Liability"]

        up_capitalized_value_n = np.zeros(shape = (len(month), len(planner.N)))
        med_capitalized_value_n = np.zeros(shape = (len(month), len(planner.N)))
        low_capitalized_value_n = np.zeros(shape = (len(month), len(planner.N)))
        for i in month:
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
        plot_df["liab"] = -SMF["Target Liability"]
        plot_df["cumsum"] = np.cumsum(plot_df["ass"]+plot_df["liab"])
        plot_df["up_cap"] = up_capitalized_value
        plot_df["low_cap"] = low_capitalized_value
        plot_df["med_cap"] = med_capitalized_value
        
        fig = go.Figure(data = [
            go.Scatter(x = plot_df["x"], y=up_capitalized_value, fill = None, mode = "lines", line_color = "lightblue", name ="95% quantile"),
            go.Scatter(x = plot_df["x"], y=low_capitalized_value, fill = "tonexty", mode = "lines", line_color = "lightblue", name = "5% quantile"),
            go.Scatter(x = plot_df["x"], y=med_capitalized_value, mode = "lines", line_color = "blue", name = "50% quantile"),
            go.Bar(x = plot_df["x"], y=plot_df["ass"], width = bar_width, marker_color = "lightgreen", name = "Assets"),
            go.Bar(x = plot_df["x"], y=plot_df["liab"], width = bar_width,marker_color = "red", name = "Liabilities"),
            go.Scatter(x = plot_df["x"], y=plot_df["cumsum"], mode = "lines", line_color = "black", name = "Assets-Liabilities")
        ])

        fig = standardized_chart(fig) 
        fig.update_layout(showlegend=False)
        fig.show()
        return

if __name__ == "__main__":
    problem = ALMplanner(start = "Jan 2021", end = "Jan 2041")
    # set planned liabilities
    problem.liabilities.insert("car", "Jan 2026", 25000, 25000*0.65)
    problem.liabilities.insert("university", "Jan 2029", 50000, 50000*0.95)
    problem.liabilities.insert("hawaii", "Jan 2037", 34000, 34000*0.85)
    # set planned assets 
    problem.assets.insert("ass_0","Jan 2021",30000)
    recurrent_dates = ["Jan 2022", "Jan 2023", "Jan 2024", "Jan 2025", "Jan 2026", "Jan 2027"]
    for i in np.arange(len(recurrent_dates)):
        problem.assets.insert("ass_" + str(i+1),recurrent_dates[i],10000)

    problem.display(bar_width = 6)

    # generate problem
    
    #problem.generate_model()
    #print("model generated")
    #problem.solve()