import numpy as np
import pandas as pd
import pulp as lp
import matplotlib.pyplot as plt
import pickle as pkl
import datetime as dt
import time


buyandhold_portfolio = {
    0:{"Cash":0.02, "GVT_EU13":0.39, "GVT_EU57":0.39, "EM":0.04, "EQ_EU":0.08, "EQ_US":0.08}, 
    1:{"Cash":0, "GVT_EU13":0.25, "GVT_EU57":0.25, "EM":0.10, "EQ_EU":0.20, "EQ_US":0.20}, 
    2:{"Cash":0, "GVT_EU13":0.10, "GVT_EU57":0.10, "EM":0.16, "EQ_EU":0.32, "EQ_US":0.32},
}


class ALMPlanner:

    def __init__(self, start = np.datetime64("2021-01-01"), end = np.datetime64("2070-12-31"), scen_name = "scenario\Scenario", user_risk_profile = 1, buyandhold_portfolio = buyandhold_portfolio):
        self.start = start
        self.end = end
        self.T = pd.date_range(start = start, end=end, freq = "M")
        Scenario_file = open( scen_name + ".pkl", "rb")
        Scenario = pkl.load(Scenario_file)
        Scenario_file.close()
        Scenario_mu_file = open( scen_name + "_mu.pkl", "rb")
        Scenario_mu = pkl.load(Scenario_mu_file)
        Scenario_mu_file.close()
        Scenario_sigma_file = open( scen_name + "_sigma.pkl", "rb")
        Scenario_sigma = pkl.load(Scenario_sigma_file)
        Scenario_sigma_file.close()
        self.Scenario = Scenario
        self.Scenario_mu = Scenario_mu
        self.Scenario_sigma = Scenario_sigma
        self.P = list(Scenario.keys())
        self.N = list(Scenario[self.P[0]].keys())
        self.model = None

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
                self.solution.V_end_distr[n] = self.model.V_end[n].varValue
            self.solution.VaR_end = self.model.gamma_end[1].varValue
        
        return


    def display(self, bar_width):
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
        cvar_end = Planner.cvar_end
        alfa_end = Planner.cvar_end_lim
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
        gamma_end = lp.LpVariable.dicts(name = "gamma_end", indexs = [1], cat = "Continuous")
        V_end = lp.LpVariable.dicts(name = "V_end", indexs = (N), lowBound = 0, cat = "Continuous")
        
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
            #for l in L:
            #    if At[a]>=Lt[l]:
            #        model += lp.lpSum(W[a][l][p] for p in P) == 0
        
        for l in L:
            model += gamma[l] + lp.lpSum(V[l][n] for n in N)/(len(N)*(1-alfa[l])) <= Liab_TG[l] - Liab_LB[l]
            for n in N:
                model += V[l][n] >= Liab_TG[l] - Q[l][n] - gamma[l]
        
        model += gamma_end + lp.lpSum(V_end[n] for n in N)/(len(N)*(1-alfa_end)) <= cvar_end*lp.lpSum(W_end[a][p] for a in A for p in P)
        for n in N:
            model += V_end[n] >= lp.lpSum(W_end[a][p] for a in A for p in P) - Q_end[n] - gamma_end
    
        # init
        self.formulation = model
        self.W = W
        self.Q = Q
        self.Z = Z
        self.gamma = gamma
        self.V = V
        self.W_end = W_end
        self.Q_end = Q_end
        self.gamma_end = gamma_end
        self.V_end = V_end
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
        
        #model = lp.LpProblem(name = "ALMplanner", sense = lp.LpMaximize)

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

        self.user_risk_profile 

        # Parameters: 
        Liab_TG = Planner.liabilities.value_tg
        Liab_LB = Planner.liabilities.value_lb
        Assets = Planner.assets.value
        alfa = Planner.liabilities.cvar_lim
        Scenario= Planner.Scenario
        cvar_end = Planner.cvar_end
        alfa_end = Planner.cvar_end_lim
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
        gamma_end = lp.LpVariable.dicts(name = "gamma_end", indexs = [1], cat = "Continuous")
        V_end = lp.LpVariable.dicts(name = "V_end", indexs = (N), lowBound = 0, cat = "Continuous")
        
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
            #for l in L:
            #    if At[a]>=Lt[l]:
            #        model += lp.lpSum(W[a][l][p] for p in P) == 0
        
        for l in L:
            model += gamma[l] + lp.lpSum(V[l][n] for n in N)/(len(N)*(1-alfa[l])) <= Liab_TG[l] - Liab_LB[l]
            for n in N:
                model += V[l][n] >= Liab_TG[l] - Q[l][n] - gamma[l]
        
        model += gamma_end + lp.lpSum(V_end[n] for n in N)/(len(N)*(1-alfa_end)) <= cvar_end*lp.lpSum(W_end[a][p] for a in A for p in P)
        for n in N:
            model += V_end[n] >= lp.lpSum(W_end[a][p] for a in A for p in P) - Q_end[n] - gamma_end
    
        # init
        self.formulation = model
        self.W = W
        self.Q = Q
        self.Z = Z
        self.gamma = gamma
        self.V = V
        self.W_end = W_end
        self.Q_end = Q_end
        self.gamma_end = gamma_end
        self.V_end = V_end
        return







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