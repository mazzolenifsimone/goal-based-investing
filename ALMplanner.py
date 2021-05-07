import numpy as np
import pandas as pd
import pulp as lp
import matplotlib.pyplot as plt
import pickle as pkl
import datetime as dt
import time
import os
import plotly.express as px
import plotly.graph_objects as go

buyandhold_portfolio = {
    0:{"Cash":0.02, "GVT_EU13":0.39, "GVT_EU57":0.39, "EM":0.04, "EQ_EU":0.08, "EQ_US":0.08}, 
    1:{"Cash":0, "GVT_EU13":0.25, "GVT_EU57":0.25, "EM":0.10, "EQ_EU":0.20, "EQ_US":0.20}, 
    2:{"Cash":0, "GVT_EU13":0.10, "GVT_EU57":0.10, "EM":0.16, "EQ_EU":0.32, "EQ_US":0.32},
}


class ALMPlanner:

    def __init__(self, start = np.datetime64("2021-01-01"), end = np.datetime64("2070-12-31"), path_scenario = "scenario", scen_name = "Scenario", scen_df_name = "ETF_GBM", user_risk_profile = 1, buyandhold_portfolio = buyandhold_portfolio):
        self.start = start
        self.end = end
        self.T = pd.date_range(start = start, end=end, freq = "M")
        scen_path = os.path.join(path_scenario,scen_name)
        Scenario_file = open( scen_path + ".pkl", "rb")
        self.Scenario = pkl.load(Scenario_file)
        Scenario_file.close()
        self.P = list(self.Scenario.keys())
        self.N = list(self.Scenario[self.P[0]].keys())
        scen_df_path = os.path.join(path_scenario,scen_df_name)
        Scenario_df_file = open(scen_df_path + ".pkl", "rb")
        self.__DF_Scenario__ = pkl.load(Scenario_df_file)
        Scenario_df_file.close()
        

        self.Scenario_mu = self.__DF_Scenario__[self.P].mean()
        self.Scenario_sigma = self.__DF_Scenario__[self.P].cov()
        
        self.model = None
        self.buyandhold = None

        self.user_portfolio = buyandhold_portfolio[user_risk_profile]


        self.cvar_end = 1
        self.cvar_end_lim = 0.95

        self.liabilities = ALMLiability(self)
        self.assets = ALMAssets(self)
        return

    def generate_model(self, model_version = "V1"):
        tic = time.time()
        if model_version == "V1":
            self.model = ALMModel_V1(self)
            print(f"Model generated in {np.round(time.time()-tic,2)} s")
        elif model_version == "V2":
            self.model = ALMModel_V2(self)
            print(f"Model generated in {np.round(time.time()-tic,2)} s")
        else:
            print(f"No valid version named {model_version}")
        return

    def generate_buyandhold(self):
        tic = time.time()
        self.buyandhold = ALMBuyAndHold(self)
        print(f"Buy-And-Hold generated in {np.round(time.time()-tic,2)} s")
        return

    def solve_BAH(self):
        tic = time.time()
        status = self.buyandhold.formulation.solve()
        print(f"Solve ended in {np.round(time.time()-tic,2)} s with {lp.constants.LpStatus[status]} solution")
        self.solution_BAH = ALMSolution(status)
        A = self.model.A
        L = self.model.L
        P = self.model.P
        N = self.model.N

        if status == 1:
            for a in A:
                self.solution_BAH.asset_part[a] = {}
                for l in L:
                    self.solution_BAH.asset_part[a][l] = {}
                    for p in P:
                        self.solution_BAH.asset_part[a][l][p] = self.buyandhold.W[a][l][p].varValue
            for a in A:
                self.solution_BAH.asset_end_part[a]={}
                for p in P:
                    self.solution_BAH.asset_end_part[a][p] = self.buyandhold.W_end[a][p].varValue
            for l in L:
                self.solution_BAH.liab_distr[l] = {}
                self.solution_BAH.V_distr[l] = {}
                for n in N:
                    self.solution_BAH.liab_distr[l][n] = self.buyandhold.Q[l][n].varValue
            for l in L:
                self.solution_BAH.Z_distr[l] = self.buyandhold.Z[l].varValue
            for n in N:
                self.solution_BAH.liab_end_distr[n] = self.buyandhold.Q_end[n].varValue
        
        return
    
    def solve(self):
        tic = time.time()
        status = self.model.formulation.solve()
        print(f"Solve ended in {np.round(time.time()-tic,2)} s with {lp.constants.LpStatus[status]} solution")
        self.solution = ALMSolution(status)
        A = self.model.A
        L = self.model.L
        P = self.model.P
        N = self.model.N

        if status == 1:
            for a in A:
                self.solution.asset_part[a] = {}
                for l in L:
                    self.solution.asset_part[a][l] = {}
                    for p in P:
                        self.solution.asset_part[a][l][p] = self.model.W[a][l][p].varValue
            for a in A:
                self.solution.asset_end_part[a]={}
                for p in P:
                    self.solution.asset_end_part[a][p] = self.model.W_end[a][p].varValue
            for l in L:
                self.solution.liab_distr[l] = {}
                self.solution.V_distr[l] = {}
                for n in N:
                    self.solution.liab_distr[l][n] = self.model.Q[l][n].varValue
                    self.solution.V_distr[l][n] = self.model.V[l][n].varValue
            for l in L:
                self.solution.VaR_liab[l] = self.model.gamma[l].varValue
                self.solution.Z_distr[l] = self.model.Z[l].varValue
            for n in N:
                self.solution.liab_end_distr[n] = self.model.Q_end[n].varValue
                #self.solution.V_end_distr[n] = self.model.V_end[n].varValue
            #self.solution.VaR_end = self.model.gamma_end[1].varValue
        
        return


    def display_old(self, bar_width):
        portfolio = pd.Series(self.user_portfolio.values(),index = self.user_portfolio.keys())

        mu = np.dot(self.Scenario_mu, portfolio)
        sigma = np.sqrt(np.dot(np.dot(portfolio,self.Scenario_sigma),portfolio))

        T = pd.date_range(self.start, self.end, freq="M")
        month = np.arange(len(T))

        Assets = self.assets.lists()
        Liabilities = self.liabilities.lists()

        SMF = pd.DataFrame({"Date":T,"Month since Start":month}).merge(Assets[["Month since Start","Asset Value"]], how = "left", on = "Month since Start").merge(Liabilities[["Month since Start","Target Liability", "Lowerbound Liability", "CVaR Level"]], how = "left", on = "Month since Start").fillna(0).reset_index()

        Ass_val = SMF["Asset Value"]
        Liab_val = SMF["Target Liability"]
        low_Liab_val = SMF["Lowerbound Liability"]

        instants = np.arange(len(SMF.index))
        inst_mat = np.tile(instants, (len(instants),1))
        inst_mat = np.maximum(inst_mat - inst_mat.T,0)
        
        mu_mat = inst_mat * mu
        sigma_mat = np.sqrt(inst_mat) * sigma

        avg_Cap_mat = np.exp(mu_mat)
        up_Cap_mat = np.exp(mu_mat + 2*sigma_mat)
        low_Cap_mat = np.exp(mu_mat - 2*sigma_mat)
        
        Asset_flow = np.triu(np.tile(Ass_val, ((len(instants),1))).T)

        tot_asset = Ass_val.sum()

        Liab_flow = np.triu(np.tile(Liab_val, ((len(instants),1))).T)
        low_Liab_flow = np.triu(np.tile(low_Liab_val, ((len(instants),1))).T)

        avg_capitalized_value = np.maximum(np.sum(np.multiply(avg_Cap_mat,Asset_flow - Liab_flow), axis = 0),0)
        up_capitalized_value = np.maximum(np.sum(np.multiply(up_Cap_mat,Asset_flow - Liab_flow), axis = 0),0)
        low_capitalized_value = np.maximum(np.sum(np.multiply(low_Cap_mat,Asset_flow - low_Liab_flow), axis = 0),0)

        plt.figure(figsize = (15,10))
        plt.bar(x = SMF.index, height = SMF["Asset Value"], width = bar_width, color = "lightgreen")
        plt.bar(x = SMF.index, height = -SMF["Target Liability"], width = bar_width, color = "red")
        plt.plot(SMF.index, np.cumsum(SMF["Asset Value"])-np.cumsum(SMF["Target Liability"]), color = "black")
        plt.plot(SMF.index, avg_capitalized_value, color = "blue")
        plt.fill_between(SMF.index,up_capitalized_value,low_capitalized_value, color = "lightblue")
        #plt.fill_between(SMF.index,avg_capitalized_value,low_capitalized_value, color = "lightblue")
        plt.grid()

        plt.show()
        return

    def display(self, bar_width):
        ETF_GBM = self.__DF_Scenario__

        portfolio = pd.Series(self.user_portfolio.values(),index = self.user_portfolio.keys())
        userpf_capfact = np.dot(np.exp(ETF_GBM[self.P]), portfolio)
        cap_factor_ptf = np.reshape(userpf_capfact, (int(len(userpf_capfact)/len(self.N)),len(self.N)))

        T = pd.date_range(self.start, self.end, freq="M")
        month = np.arange(len(T))

        Assets = self.assets.lists()
        Liabilities = self.liabilities.lists()

        SMF = pd.DataFrame({"Date":T,"Month since Start":month}).merge(Assets[["Month since Start","Asset Value"]], how = "left", on = "Month since Start").merge(Liabilities[["Month since Start","Target Liability", "Lowerbound Liability", "CVaR Level"]], how = "left", on = "Month since Start").fillna(0).reset_index()

        Ass_val = SMF["Asset Value"]
        Liab_val = SMF["Target Liability"]
        low_Liab_val = SMF["Lowerbound Liability"]

        up_capitalized_value_n = np.zeros(shape = (len(month), len(self.N)))
        med_capitalized_value_n = np.zeros(shape = (len(month), len(self.N)))
        low_capitalized_value_n = np.zeros(shape = (len(month), len(self.N)))
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


class ALMLiability():

    def __init__(self, Planner):
        self.set = []
        self.value_tg = {}
        self.value_lb = {}
        self.cvar_lim = {}
        self.period = {}
        self.date = {}
        self.start = Planner.start
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

    def __init__(self, Planner):
        self.set = []
        self.value= {}
        self.date = {}
        self.period = {}
        self.start = Planner.start
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



class ALMModel_V1():

    def __init__(self, Planner):
        
        model = lp.LpProblem(name = "ALMplanner", sense = lp.LpMaximize)

        # Sets:
        T = Planner.T
        self.T = T
        A = Planner.assets.set 
        self.A = A
        L = Planner.liabilities.set
        self.L = L
        P = Planner.P
        self.P = P
        N = Planner.N
        self.N = N

        # Parameters: 
        Liab_TG = Planner.liabilities.value_tg
        Liab_LB = Planner.liabilities.value_lb
        Assets = Planner.assets.value
        alfa = Planner.liabilities.cvar_lim
        Scenario= Planner.Scenario
        base_portfolio = Planner.user_portfolio
        #cvar_end = Planner.cvar_end
        #alfa_end = Planner.cvar_end_lim

        At = Planner.assets.period
        Lt = Planner.liabilities.period

        # Variables
        W = lp.LpVariable.dicts(name = "W", indexs = (A,L,P), lowBound = 0, cat = "Continuous")
        Q = lp.LpVariable.dicts(name = "Q", indexs = (L,N), lowBound = 0, cat = "Continuous")
        Z = lp.LpVariable.dicts(name = "Z", indexs = (L), lowBound = 0, cat = "Continuous")
        gamma = lp.LpVariable.dicts(name = "gamma", indexs = (L), cat = "Continuous")
        V = lp.LpVariable.dicts(name = "V", indexs = (L,N), lowBound = 0, cat = "Continuous")
        W_end = lp.LpVariable.dicts(name = "W_end", indexs = (A,P), lowBound = 0, cat = "Continuous")
        Q_end = lp.LpVariable.dicts(name = "Q_end", indexs = (N), lowBound = 0, cat = "Continuous")
        #gamma_end = lp.LpVariable.dicts(name = "gamma_end", indexs = [1], cat = "Continuous")
        #V_end = lp.LpVariable.dicts(name = "V_end", indexs = (N), lowBound = 0, cat = "Continuous")
        
        # Objective Function
        model += lp.lpSum(Z[l] for l in L) + lp.lpSum(Q_end[n] for n in N)/(len(N)*sum(Liab_TG.values()))
        
        # Constraints
        for l in L:
            model += Z[l] <= Liab_TG[l]
            model += Z[l]*len(N) <= lp.lpSum(Q[l][n] for n in N)
        
        for l in L:
            for n in N:
                model += Q[l][n] == lp.lpSum(W[a][l][p]*np.exp(np.sum(Scenario[p][n][At[a]:Lt[l]])) for a in A for p in P)
        
        for n in N:
            model += Q_end[n] == lp.lpSum(W_end[a][p]* np.exp(np.sum(Scenario[p][n][At[a]:])) for a in A for p in P)
        
        for a in A:
            L_feas = np.array(list(Lt.keys()))[np.array(list(Lt.values()))>At[a]]
            L_unfeas = np.array(list(Lt.keys()))[np.array(list(Lt.values()))<=At[a]]
            model += lp.lpSum(W[a][l][p] for l in L_feas for p in P) + lp.lpSum(W_end[a][p] for p in P) == Assets[a]
            model += lp.lpSum(W[a][l][p] for l in L_unfeas for p in P) == 0
            for p in P:
                model += W_end[a][p] == lp.lpSum(W_end[a][p] for p in P)*base_portfolio[p]
        
        for l in L:
            model += gamma[l] + lp.lpSum(V[l][n] for n in N)/(len(N)*(1-alfa[l])) <= Liab_TG[l] - Liab_LB[l]
            for n in N:
                model += V[l][n] >= Liab_TG[l] - Q[l][n] - gamma[l]
        
        #model += gamma_end + lp.lpSum(V_end[n] for n in N)/(len(N)*(1-alfa_end)) <= cvar_end*lp.lpSum(W_end[a][p] for a in A for p in P)
        #for n in N:
        #    model += V_end[n] >= lp.lpSum(W_end[a][p] for a in A for p in P) - Q_end[n] - gamma_end
    
        # init
        self.formulation = model
        self.W = W
        self.Q = Q
        self.Z = Z
        self.gamma = gamma
        self.V = V
        self.W_end = W_end
        self.Q_end = Q_end
        #self.gamma_end = gamma_end
        #self.V_end = V_end
        return

        

class ALMModel_V2():

    def __init__(self, Planner):
        
        model = lp.LpProblem(name = "ALMplanner", sense = lp.LpMinimize)

        # Sets:
        T = Planner.T
        self.T = T
        A = Planner.assets.set 
        self.A = A
        L = Planner.liabilities.set
        self.L = L
        P = Planner.P
        self.P = P
        N = Planner.N
        self.N = N

        # Parameters: 
        Liab_TG = Planner.liabilities.value_tg
        Max_Assets = Planner.assets.value
        alfa = Planner.liabilities.cvar_lim
        ETF_GBM = Planner.ETF_GBM
        At = Planner.assets.period
        Lt = Planner.liabilities.period
        
        # Variables
        W = lp.LpVariable.dicts(name = "W", indexs = (A,L,P), lowBound = 0, cat = "Continuous")
        Q = lp.LpVariable.dicts(name = "Q", indexs = (L,N), lowBound = 0, cat = "Continuous")
        Z = lp.LpVariable.dicts(name = "Z", indexs = (L), lowBound = 0, cat = "Continuous")
        gamma = lp.LpVariable.dicts(name = "gamma", indexs = (L), cat = "Continuous")
        V = lp.LpVariable.dicts(name = "V", indexs = (L,N), lowBound = 0, cat = "Continuous")
        
        # Objective Function
        model += lp.lpSum(W[a][l][p] for a in A for l in L for p in P)
        
        # Constraints      
        for l in L:
            for n in N:
                model += Q[l][n] == lp.lpSum(W[a][l][p]*np.exp(np.sum(Scenario[p][n][At[a]:Lt[l]])) for a in A for p in P)
        
        for a in A:
            L_feas = np.array(list(Lt.keys()))[np.array(list(Lt.values()))>At[a]]
            L_unfeas = np.array(list(Lt.keys()))[np.array(list(Lt.values()))<=At[a]]
            model += lp.lpSum(W[a][l][p] for l in L_feas for p in P) <= Max_Assets[a]
            model += lp.lpSum(W[a][l][p] for l in L_unfeas for p in P) == 0
        
        for l in L:
            model += gamma[l] + lp.lpSum(V[l][n] for n in N)/(len(N)*(1-alfa[l])) <= 0
            for n in N:
                model += V[l][n] >= Liab_TG[l] - Q[l][n] - gamma[l]
        
        # init
        self.formulation = model
        self.W = W
        self.Q = Q
        self.Z = Z
        self.gamma = gamma
        self.V = V
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
        self.VaR_end = {}
        self.V_end_distr = {}
        return




class ALMBuyAndHold():

    def __init__(self, Planner):
        model = lp.LpProblem(name = "ALMbuyandhold", sense = lp.LpMaximize)

        # Sets:
        T = Planner.T
        self.T = T
        A = Planner.assets.set 
        self.A = A
        L = Planner.liabilities.set
        self.L = L
        P = Planner.P
        self.P = P
        N = Planner.N
        self.N = N

        # Parameters: 
        Liab_TG = Planner.liabilities.value_tg
        Liab_LB = Planner.liabilities.value_lb
        Assets = Planner.assets.value
        alfa = Planner.liabilities.cvar_lim
        Scenario= Planner.Scenario
        base_portfolio = Planner.user_portfolio
        #cvar_end = Planner.cvar_end
        #alfa_end = Planner.cvar_end_lim

        At = Planner.assets.period
        Lt = Planner.liabilities.period

        # Variables
        W = lp.LpVariable.dicts(name = "W", indexs = (A,L,P), lowBound = 0, cat = "Continuous")
        Q = lp.LpVariable.dicts(name = "Q", indexs = (L,N), lowBound = 0, cat = "Continuous")
        Z = lp.LpVariable.dicts(name = "Z", indexs = (L), lowBound = 0, cat = "Continuous")
        W_end = lp.LpVariable.dicts(name = "W_end", indexs = (A,P), lowBound = 0, cat = "Continuous")
        Q_end = lp.LpVariable.dicts(name = "Q_end", indexs = (N), lowBound = 0, cat = "Continuous")

        
        # Objective Function
        model += lp.lpSum(Z[l] for l in L) + lp.lpSum(Q_end[n] for n in N)/(len(N)*sum(Liab_TG.values()))
        
        # Constraints
        for l in L:
            model += Z[l] <= Liab_TG[l]
            model += Z[l]*len(N) <= lp.lpSum(Q[l][n] for n in N)
        
        for l in L:
            for n in N:
                model += Q[l][n] == lp.lpSum(W[a][l][p]*np.exp(np.sum(Scenario[p][n][At[a]:Lt[l]])) for a in A for p in P)
        
        for n in N:
            model += Q_end[n] == lp.lpSum(W_end[a][p]* np.exp(np.sum(Scenario[p][n][At[a]:])) for a in A for p in P)
            
        Asset_al_split, Asset_aend_split = smart_asset_allocation(Planner)

        for a in A:
            L_feas = np.array(list(Lt.keys()))[np.array(list(Lt.values()))>At[a]]
            L_unfeas = np.array(list(Lt.keys()))[np.array(list(Lt.values()))<=At[a]]
            model += lp.lpSum(W[a][l][p] for l in L_unfeas for p in P) == 0
            for p in P:
                model += W_end[a][p] == Asset_aend_split[a]*base_portfolio[p]
                for l in L_feas:
                    model += W[a][l][p] == Asset_al_split[l][a]*base_portfolio[p]

        # init
        self.formulation = model
        self.W = W
        self.Q = Q
        self.Z = Z
        self.W_end = W_end
        self.Q_end = Q_end
        #self.gamma_end = gamma_end
        #self.V_end = V_end
        return


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


def smart_asset_allocation(Planner):
    Assets_list = Planner.assets.lists()
    Liabs_list = Planner.liabilities.lists()
    
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

if __name__ == "__main__":
    problem = ALMPlanner(start = "Jan 2021", end = "Jan 2041")
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