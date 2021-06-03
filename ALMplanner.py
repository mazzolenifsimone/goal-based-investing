import numpy as np
import pandas as pd
import pulp as lp
import pickle as pkl
import time
import os

#buyandhold_portfolio = {
#    0:{"Cash":0.02, "GVT_EU13":0.39, "GVT_EU57":0.39, "EM":0.04, "EQ_EU":0.08, "EQ_US":0.08}, 
#    1:{"Cash":0, "GVT_EU13":0.25, "GVT_EU57":0.25, "EM":0.10, "EQ_EU":0.20, "EQ_US":0.20}, 
#    2:{"Cash":0, "GVT_EU13":0.10, "GVT_EU57":0.10, "EM":0.16, "EQ_EU":0.32, "EQ_US":0.32},
#}



class ALMplanner:

    def __init__(self, start = np.datetime64("2021-01-01"), end = np.datetime64("2070-12-31"), path_scenario = "scenario", scen_name = "Scenario", scen_df_name = "ETF_GBM", user_risk_profile = 1, buyandhold_portfolios = "buyandhold_portfolios"):
        self.start = start
        self.end = end
        self.T = pd.date_range(start = start, end=end, freq = "M")
        self.Scenario = load_scenario(path_scenario, scen_name)
        self.P = list(self.Scenario.keys())
        self.N = list(self.Scenario[self.P[0]].keys())
        self.__DF_Scenario__ = load_scenario(path_scenario,scen_df_name)
        self.Scenario_mu = self.__DF_Scenario__[self.P].mean()
        self.Scenario_sigma = self.__DF_Scenario__[self.P].cov()
        buyandhold_portfolios = load_scenario(path_scenario,buyandhold_portfolios)
        self.user_portfolio = buyandhold_portfolios[user_risk_profile]
        self.feasibility = -1
        self.liabilities = ALMLiability(self)
        self.assets = ALMAssets(self) 
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
    
    def StandardModelForm(self):
        Assets = self.assets.lists()
        Liabilities = self.liabilities.lists()
        SMF = pd.DataFrame({"Date":self.T,"Month since Start":np.arange(len(self.T))}).merge(Assets[["Month since Start","Asset Value"]], how = "left", on = "Month since Start").merge(Liabilities[["Month since Start","Goal Value", "Goal Lower Bound", "CVaR Level"]], how = "left", on = "Month since Start").reset_index()#.fillna(0).reset_index()
        return SMF



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
        date_dt = pd.to_datetime(date)
        self.date[label] = date_dt
        self.set = sorted(self.date.keys(), key = lambda x: self.date[x])
        start_dt = pd.to_datetime(self.start)
        self.period[label] = (date_dt.year - start_dt.year)*12 + date_dt.month - start_dt.month 
        self.value_tg[label] = value_tg
        self.value_lb[label] = value_lb
        self.cvar_lim[label] = cvar_lim
        return

    def lists(self):
        ListOfLiabilities = pd.DataFrame(index=self.set, columns = ["Date", "Month since Start", "Goal Value", "Goal Lower Bound", "CVaR Level"])
        for i in ListOfLiabilities.index:
            ListOfLiabilities["Date"][i] = self.date[i]
            ListOfLiabilities["Month since Start"][i] = self.period[i]
            ListOfLiabilities["Goal Value"][i] = self.value_tg[i]
            ListOfLiabilities["Goal Lower Bound"][i] = self.value_lb[i]
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
        date_dt = pd.to_datetime(date)
        self.date[label] = date_dt
        self.set = sorted(self.date.keys(), key = lambda x: self.date[x])
        self.value[label] = value
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
        ###### PULP MODEL #######
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
        #########################
        return



class ALMGoalBased():

    def __init__(self, planner):
        tic = time.time()
        self.P = planner.P
        self.A = planner.assets.set
        self.L = planner.liabilities.set
        self.N = planner.N
        ###### PULP MODEL #######
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
        #########################
        print(f"GoalBased model generated in {np.round(time.time()-tic,2)} s")
        return
    
    def solve(self):
        model_solve(self)     
        return



class ALMBuyAndHold():

    def __init__(self, planner):
        tic = time.time()
        self.P = planner.P
        self.A = planner.assets.set
        self.L = planner.liabilities.set
        self.N = planner.N
        ###### PULP MODEL #######
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
        #########################
        print(f"BuyAndHold model generated in {np.round(time.time()-tic,2)} s")   
        return

    def solve(self):
        model_solve(self)     
        return



class ALMBuyAndHold_2():

    def __init__(self, planner):
        self.P = planner.P
        self.A = planner.assets.set
        self.L = planner.liabilities.set
        self.N = planner.N
        self.planner = planner
        return

    def solve(self, portfolio_strategy = None):
        model_solve_BaH(self, portfolio_strategy = portfolio_strategy)     
        return



class ALMSolution():
    
    def __init__(self, status):
        self.status = status
        self.asset_to_goal = {}
        self.goal_distr = {}
        self.goal_VaR = {}
        self.loss_distr = {}
        self.asset_to_exwealth = {}
        self.goal_exwealth = {}
        self.final_exwealth = {}
        return



## FUNCTIONS
def add_recurrent(planner, start, end, type, label, value = 0, value_tg = 0, value_lb = 0):
    recurrent_dates = pd.date_range(start = start, end = end, freq = pd.offsets.YearBegin(1))
    for i in np.arange(len(recurrent_dates)):
        index = str(int(i/10))+str(i-int(i/10)*10)
        if type == "asset":
            planner.assets.insert(label + "_" + index,recurrent_dates[i], value)
        elif type == "goal":
            planner.liabilities.insert(label + "_" + index,recurrent_dates[i], value_tg, value_lb)
    return

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
        model.formulation += model.Q_end[n] == lp.lpSum(model.W_end[a][p]*np.exp(np.sum(planner.Scenario[p][n][planner.assets.period[a]:len(planner.T)])) for a in planner.assets.set for p in planner.P) + lp.lpSum(model.Q_ex[l][n]*planner.user_portfolio[p]*np.exp(np.sum(planner.Scenario[p][n][planner.liabilities.period[l]:len(planner.T)])) for l in planner.liabilities.set for p in planner.P)
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
    # in .lists method elements are ordered by Date
    Lt = Liabs_list["Month since Start"]
    L = Liabs_list.index
    #Liab_tg = Liabs_list["Goal Value"]
    Liab_lb = Liabs_list["Goal Lower Bound"]
    At = Assets_list["Month since Start"]
    A = Assets_list.index
    Assets = Assets_list["Asset Value"]
    # DataFrame structure 
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

def model_solve_BaH(BaH_model, portfolio_strategy):
    if portfolio_strategy is None:
        portfolio_strategy = BaH_model.planner.user_portfolio
    Asset_al_split, Asset_aend_split = smart_asset_allocation(BaH_model.planner)
    W, W_end = get_BaH_W(BaH_model, Asset_al_split, Asset_aend_split, portfolio_strategy)
    Q, Q_ex, Q_end =  get_Q(BaH_model, BaH_model.planner, W, W_end, portfolio_strategy)
    ###################################################################

    ###### Aggiungere oggetto solution con i valori standardizzati

    ###################################################################
    return

def get_BaH_W(model, Asset_al_split, Asset_aend_split, portfolio_strategy):
    W = {}
    W_end = {}
    for a in model.A:
        W[a] = {}
        W_end[a] = {}
        for l in model.L:
            W[a][l] = {}
            for p in model.P:
                W[a][l][p] = Asset_al_split[l][a]*portfolio_strategy[p]
        for p in model.P:
            W_end[a][p] = Asset_aend_split[a]*portfolio_strategy[p]
    return W, W_end

def get_Q(model, planner, W, W_end, portfolio_strategy):
    Q = {}
    Q_ex = {}
    Q_end = {}
    for l in model.L:
        Q[l] = {}
        Q_ex[l] = {}
        for n in model.N:
            proj_assets = 0
            for a in model.A:
                for p in model.P:
                    proj_assets = proj_assets + W[a][l][p]*np.exp(np.sum(planner.Scenario[p][n][planner.assets.period[a]:planner.liabilities.period[l]]))
            Q_ex[l][n] = max(proj_assets - planner.liabilities.value_tg[l], 0)
            Q[l][n] == proj_assets - Q_ex[l][n]
    for n in model.N:
        extra_wealth_l = 0
        for l in model.L:
            for p in model.P:
                extra_wealth_l = extra_wealth_l + Q_ex[l][n]*portfolio_strategy[p]*np.exp(np.sum(planner.Scenario[p][n][planner.liabilities.period[l]:len(planner.T)]))
        extra_wealth_a = 0
        for a in model.A:
            for p in model.P:
                extra_wealth_a = extra_wealth_a + W_end[a][p]*np.exp(np.sum(planner.Scenario[p][n][planner.assets.period[a]:len(planner.T)]))
        Q_end[n] = extra_wealth_l + extra_wealth_a 
    return Q, Q_ex, Q_end

def model_solve(model):
    tic = time.time()
    status = model.formulation.solve()
    print(f"Solve ended in {np.round(time.time()-tic,2)} s with {lp.constants.LpStatus[status]} solution")
    model.solution = ALMSolution(status)
    if model.solution.status == 1:
        save_solution(model)
    return 

def save_solution(model):
    for a in model.A:
        model.solution.asset_to_goal[a] = {}
        for l in model.L:
            model.solution.asset_to_goal[a][l] = {}
            for p in model.P:
                model.solution.asset_to_goal[a][l][p] = model.W[a][l][p].varValue
    for a in model.A:
        model.solution.asset_to_exwealth[a]={}
        for p in model.P:
            model.solution.asset_to_exwealth[a][p] = model.W_end[a][p].varValue
    for l in model.L:
        model.solution.goal_distr[l] = {}
        model.solution.loss_distr[l] = {}
        model.solution.goal_exwealth[l] = {}
        model.solution.goal_VaR[l] = model.gamma[l].varValue
        for n in model.N:
            model.solution.goal_distr[l][n] = model.Q[l][n].varValue
            model.solution.goal_exwealth[l][n] = model.Q_ex[l][n].varValue
            model.solution.loss_distr[l][n] = model.V[l][n].varValue
    for n in model.N:
        model.solution.final_exwealth[n] = model.Q_end[n].varValue
    return