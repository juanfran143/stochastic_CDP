import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

""" 
def box_plot(output_relibility, output_simulation_penalitation_cost):
    df = pd.read_excel("output/ResumeOutputs.xlsx")
    if not output_relibility:
        df_stochastic_alpha = pd.DataFrame(df.loc[np.logical_and(df["deterministic"] == False, df["type_simulation"] == True), :])
        OBS_S_S_Alpha = df_stochastic_alpha[["Instance", "variance", "stochastic_of"]].groupby(["Instance", "variance"]).mean().reset_index()
        OBS_S_S_Alpha = OBS_S_S_Alpha.rename(columns = {"stochastic_of": "OBS_S_S_Alpha"}, inplace = False)

        df_stochastic_Penal = pd.DataFrame(df.loc[np.logical_and(df["deterministic"] == False, df["type_simulation"] == False), :])
        OBS_S_S_Penal = df_stochastic_Penal[["Instance", "variance", "stochastic_of"]].groupby(["Instance", "variance"]).mean().reset_index()
        OBS_S_S_Penal = OBS_S_S_Penal.rename(columns = {"stochastic_of": "OBS_S_S_Penal"}, inplace = False)

        df_Deterministic_Penal = pd.DataFrame(df.loc[np.logical_and(df["deterministic"] == True, df["type_simulation"] == False), :])
        OBS_D_S_Penal = df_Deterministic_Penal[["Instance", "variance", "stochastic_of"]].groupby(["Instance", "variance"]).mean().reset_index()
        OBS_D_S_Penal = OBS_D_S_Penal.rename(columns = {"stochastic_of": "OBS_D_S_Penal"}, inplace = False)

        df_deterministic = pd.DataFrame(df.loc[df["deterministic"], :])
        max_determinsit_cost = df_deterministic[["Instance", "CostSol"]].groupby(["Instance"]).max().reset_index()

        General = pd.merge(max_determinsit_cost, OBS_S_S_Alpha, how='left', on=["Instance"])
        General = pd.merge(General, OBS_S_S_Penal, how='left', on=["Instance", "variance"])
        General = pd.merge(General, OBS_D_S_Penal, how='left', on=["Instance", "variance"])

        General["OBS-D vs OBS-D-S"] = (General["CostSol"]-General["OBS_D_S_Penal"])/General["CostSol"] * 100
        General["OBS-D vs OBS-S-S-Alpha"] = (General["CostSol"]-General["OBS_S_S_Alpha"])/General["CostSol"] * 100
        General["OBS-D vs OBS_S_S_Penal"] = (General["CostSol"]-General["OBS_S_S_Penal"])/General["CostSol"] * 100

        Boxplot = pd.DataFrame(columns=["type_gap", "var", "Gap"])


        if output_simulation_penalitation_cost:
            variable = ["OBS-D vs OBS-D-S", "OBS-D vs OBS_S_S_Penal"]
            for k, i in enumerate(General.values):
                for j in range(2):
                    Boxplot.loc[k*2+j, "type_gap"] = variable[j]
                    Boxplot.loc[k*2+j, "var"] = i[2]
                    if j==0:
                        Boxplot.loc[k*2+j, "Gap"] = i[6]
                    if j==1:
                        Boxplot.loc[k * 2 + j, "Gap"] = i[8]
        else:
            variable = ["OBS-D vs OBS-D-S", "OBS-D vs OBS-S-S-Alpha"]
            for k, i in enumerate(General.values):
                for j in range(2):
                    Boxplot.loc[k*2+j, "type_gap"] = variable[j]
                    Boxplot.loc[k*2+j, "var"] = i[2]
                    if j==0:
                        Boxplot.loc[k*2+j, "Gap"] = i[6]
                    if j==1:
                        Boxplot.loc[k * 2 + j, "Gap"] = i[7]

        #boxplot = General.boxplot(column=["OBS-D vs OBS-D-S", "OBS-D vs OBS-S-S-Alpha", "OBS-D vs OBS_S_S_Penal"])
        ax = sns.boxplot(x="var", y="Gap", hue="type_gap", data=Boxplot, palette="Set3")
        plt.title("Objective function")
        plt.show()


    else:
        df_stochastic_alpha = pd.DataFrame(df.loc[np.logical_and(df["deterministic"] == False, df["type_simulation"] == True), :])
        OBS_S_S_Alpha = df_stochastic_alpha[["Instance", "variance", "reliability"]].groupby(["Instance", "variance"]).mean().reset_index()
        OBS_S_S_Alpha = OBS_S_S_Alpha.rename(columns = {"reliability": "OBS_S_S_Alpha"}, inplace = False)

        df_stochastic_Penal = pd.DataFrame(df.loc[np.logical_and(df["deterministic"] == False, df["type_simulation"] == False), :])
        OBS_S_S_Penal = df_stochastic_Penal[["Instance", "variance", "reliability"]].groupby(["Instance", "variance"]).mean().reset_index()
        OBS_S_S_Penal = OBS_S_S_Penal.rename(columns = {"reliability": "OBS_S_S_Penal"}, inplace = False)

        df_Deterministic_Penal = pd.DataFrame(df.loc[np.logical_and(df["deterministic"] == True, df["type_simulation"] == False), :])
        OBS_D_S_Penal = df_Deterministic_Penal[["Instance", "variance", "reliability"]].groupby(["Instance", "variance"]).mean().reset_index()
        OBS_D_S_Penal = OBS_D_S_Penal.rename(columns = {"reliability": "OBS_D_S_Penal"}, inplace = False)

        df_deterministic = pd.DataFrame(df.loc[df["deterministic"], :])
        max_determinsit_cost = df_deterministic[["Instance", "CostSol"]].groupby(["Instance"]).max().reset_index()

        General = pd.merge(OBS_S_S_Penal, OBS_S_S_Alpha, how='left', on=["Instance", "variance"])
        General = pd.merge(General, OBS_D_S_Penal, how='left', on=["Instance", "variance"])

        Boxplot = pd.DataFrame(columns=["type_gap", "var", "Relibility"])


        if output_simulation_penalitation_cost:
            variable = ["OBS-D-S", "OBS_S_S_Penal"]
            for k, i in enumerate(General.values):
                for j in range(2):
                    Boxplot.loc[k*2+j, "type_gap"] = variable[j]
                    Boxplot.loc[k*2+j, "var"] = i[1]
                    if j==0:
                        Boxplot.loc[k*2+j, "Relibility"] = i[4]
                    if j==1:
                        Boxplot.loc[k * 2 + j, "Relibility"] = i[2]
        else:
            variable = ["OBS-D-S", "OBS-S-S-Alpha"]
            for k, i in enumerate(General.values):
                for j in range(2):
                    Boxplot.loc[k*2+j, "type_gap"] = variable[j]
                    Boxplot.loc[k*2+j, "var"] = i[1]
                    if j==0:
                        Boxplot.loc[k*2+j, "Relibility"] = i[4]
                    if j==1:
                        Boxplot.loc[k * 2 + j, "Relibility"] = i[3]

        #boxplot = General.boxplot(column=["OBS-D vs OBS-D-S", "OBS-D vs OBS-S-S-Alpha", "OBS-D vs OBS_S_S_Penal"])
        ax = sns.boxplot(x="var", y="Relibility", hue="type_gap", data=Boxplot, palette="Set3")
        plt.title("Relibility")
        plt.show()

def box_plot_results_good():
    df = pd.read_excel("output/Results.xlsx")
    data_box = df.loc[:, ['Gap [2] - [3]', 'Gap [2] - [4]']] * 100
    data_box.columns = ['Forward-BR', 'Backward-BR']

    data_box.loc[10:29, "Type"] = "GKD-b"
    data_box.loc[30:39, "Type"] = "GKD-c"
    data_box.loc[40:, "Type"] = "MDG-b"
    data_box = data_box.loc[10:, :].reset_index()
    data_f = pd.DataFrame()
    for i in range(len(data_box)):
        data_f.loc[i, "Algorithm"] = "Forward-BR"
        data_f.loc[i, "OF"] = data_box.loc[i, "Forward-BR"]
        data_f.loc[i, "Instances"] = data_box.loc[i, "Type"]
        data_f.loc[i+len(data_box), "Algorithm"] = "Backward-BR"
        data_f.loc[i+len(data_box), "OF"] = data_box.loc[i, "Backward-BR"]
        data_f.loc[i+len(data_box), "Instances"] = data_box.loc[i, "Type"]
    data_f = data_f.reset_index(drop=True)

    ax = sns.boxplot(x = "Instances", y = "OF", hue = "Algorithm", data=data_f, palette="Set3")
    ax.axhline(0, ls='--', color='r')
    plt.ylabel("Gap (%) w.r.t the original Strategy")
    plt.title("Gap between algorithms for instances")
    plt.show()


def box_plot_results_modify_good():
    df = pd.read_excel("output/Modify instances.xlsx")
    data_box = df.loc[:, ['Gap 0,4', 'Gap 0,6', 'Gap 0,8']] * 100
    data_box.loc[:, "Instance"] = "GKD-b"
    data_box.loc[20:29, "Instance"] = "GKD-c"
    data_box.loc[30:, "Instance"] = "MDG-b"
    #data_box.columns = ['Forward-BR', 'Backward-BR']

    data_f = pd.DataFrame()
    for i in range(len(data_box)):
        data_f.loc[i, "Open facilities"] = "80%"
        data_f.loc[i, "Gap"] = data_box.loc[i, "Gap 0,8"]
        data_f.loc[i, "Instance"] = data_box.loc[i, "Instance"]

        data_f.loc[i + len(data_box), "Open facilities"] = "60%"
        data_f.loc[i + len(data_box), "Gap"] = data_box.loc[i, "Gap 0,6"]
        data_f.loc[i + len(data_box), "Instance"] = data_box.loc[i, "Instance"]

        data_f.loc[i + len(data_box)*2, "Open facilities"] = "40%"
        data_f.loc[i + len(data_box)*2, "Gap"] = data_box.loc[i, "Gap 0,4"]
        data_f.loc[i + len(data_box)*2, "Instance"] = data_box.loc[i, "Instance"]
    data_f = data_f.reset_index(drop=True)

    ax = sns.boxplot(x = "Instance", y = "Gap", hue = "Open facilities", data=data_f, palette="Set3",
            showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"5"})


    ax.axhline(0, ls='--', color='r')

    plt.ylabel("Gap (%) w.r.t our forward algorithm")
    plt.title("Gap between differents open facilities")
    plt.show()
"""
#
def tabla(variance, alpha, open, BKS, Nombre_excel, writer):
    if alpha:
        var = "OBS-S-S-alpha [4]"
    else:
        var = "OBS-S-S-penalization [4]"
    General = pd.DataFrame(columns=["Instance", "BKS [1]", "OBD-D [2]", "time (s)", "[1]-[2]", "OBS-D-S [3]", "Reliability", var, " Reliability", " time (s)", "[2]-[3]", "[2]-[4]"])
    df = pd.read_excel("output/ResumeOutputs_def.xlsx")
    df = df.loc[df["variance"] == variance, :]
    df = df.loc[df["inversa"] == open, :]

    General["Instance"] = df.Instance.unique()

    deterministic_Cost = pd.DataFrame(df.loc[np.logical_and(df["deterministic"].values == True, df["type_simulation"].values == False), :])

 #
    time = 0
    for i, inst in enumerate(General["Instance"]):
        inst_cost = deterministic_Cost.loc[deterministic_Cost["Instance"] == inst, :]
        coste = 0
        for j in inst_cost.values:
            if coste < j[2]:
                coste = j[2]
                time = j[3]
                sto_coste = j[7]
                det_reliability = j[5]
            if coste == j[2] and det_reliability < j[5]:
                coste = j[2]
                time = j[3]
                sto_coste = j[7]
                det_reliability = j[5]

        General.loc[i, "time (s)"] = time
        General.loc[i, "OBD-D [2]"] = coste
        General.loc[i, "time (s)"] = time
        if alpha:
            General.loc[i, "OBS-D-S [3]"] = coste
        else:
            General.loc[i, "OBS-D-S [3]"] = sto_coste
        General.loc[i, "Reliability"] = det_reliability

    stochastic_Cost = pd.DataFrame(df.loc[np.logical_and(df["deterministic"].values == False, df["type_simulation"].values == alpha), :])
    time = 0
    reliability = 0
    for i, inst in enumerate(General["Instance"]):
        inst_cost = stochastic_Cost.loc[stochastic_Cost["Instance"] == inst, :]
        coste = 0
        for j in inst_cost.values:
            if coste < j[7]:
                coste = j[7]
                time = j[3]
                reliability = j[5]
            if coste == j[7] and reliability<j[5]:
                coste = j[7]
                time = j[3]
                reliability = j[5]
        General.loc[i, " time (s)"] = time
        General.loc[i, var] = coste
        General.loc[i, " Reliability"] = reliability

    """ 
    stochastic_Cost = pd.DataFrame(df.loc[np.logical_and(df["deterministic"].values == True, df["type_simulation"].values == False), :])
    time = 0
    reliability = 0
    for i, inst in enumerate(General["Instance"]):
        cost_sol = General.loc[General["Instance"] == inst,"OBD-D [2]"].values[0]
        inst_cost = stochastic_Cost.loc[np.logical_and(stochastic_Cost["Instance"] == inst, stochastic_Cost["CostSol"] == cost_sol ), :]
        coste = 0
        for j in inst_cost.values:
            if coste < j[7]:
                coste = j[7]
                time = j[3]
                reliability = j[5]
        General.loc[i, "time (s)"] = time
        General.loc[i, "OBS-D-S [3]"] = coste
        General.loc[i, "Reliability"] = reliability
    """
    #BKS = [147.2, 178.1, 96.1, 84.6, 154.9, 77.7, 41.8, 108.5, 119.1, 115.3, 164.2, 84.3, 63.3, 103.3, 106.6, 124.5, 163.4, 100.2, 166.3, 111.2, 4,4,5,4,4,4,4,4,4,4]
    #BKS = [147.2, 178.1, 96.1, 84.6, 154.9, 77.7, 41.8, 108.5, 119.1, 115.3]
    BKS = [125] * len(General)
    General["BKS [1]"] = BKS
    for i in range(len(General)):
        General.loc[i, "[1]-[2]"] = (General.loc[i, "BKS [1]"]-General.loc[i, 'OBD-D [2]'])/General.loc[i, "BKS [1]"] * 100
        General.loc[i, '[2]-[3]'] = (General.loc[i, 'OBD-D [2]'] - General.loc[i, 'OBS-D-S [3]']) / General.loc[i, 'OBD-D [2]'] * 100
        General.loc[i, '[2]-[4]'] = (General.loc[i, 'OBD-D [2]'] - General.loc[i, var]) / General.loc[i, 'OBD-D [2]'] * 100

    last = len(General)
    columns = General.columns
    General.loc[last, columns[0]] = "Mean"
    for i in columns[1:(len(columns)-1)]:
        General.loc[last, i] = np.mean(General.loc[0:(last-1), i])


    for i in range(len(General)):
        General.loc[i, "[1]-[2]"] = str((General.loc[i, "BKS [1]"]-General.loc[i, 'OBD-D [2]'])/General.loc[i, "BKS [1]"] * 100) + "%"
        General.loc[i, '[2]-[3]'] = str((General.loc[i, 'OBD-D [2]'] - General.loc[i, 'OBS-D-S [3]']) / General.loc[i, 'OBD-D [2]'] * 100) + "%"
        General.loc[i, '[2]-[4]'] = str((General.loc[i, 'OBD-D [2]'] - General.loc[i, var]) / General.loc[i, 'OBD-D [2]'] * 100) + "%"


    #General.to_excel(Nombre_excel, index=False)
    General.to_excel(writer, sheet_name=Nombre_excel)

#BKS = [147.2, 178.1, 96.1, 84.6, 154.9, 77.7, 41.8, 108.5, 119.1, 115.3, 164.2, 84.3, 63.3, 103.3, 106.6, 124.5, 163.4, 100.2, 166.3, 111.2, 4,4,5,4,4,4,4,4,4,4]


def all_tables():
    BKS = [147.2, 178.1, 96.1, 84.6, 154.9, 77.7, 41.8, 108.5, 119.1, 115.3, 164.2, 84.3, 63.3, 103.3, 106.6, 124.5, 163.4, 100.2, 166.3, 111.2]
    #BKS = [50.6, 46, 45.8, 47.6, 49.1, 46.6, 47.3, 48.7, 48.8, 50.5,9.4, 9.5, 9.4, 9.3, 9.3, 9.4, 9.3, 9.6, 9.3, 9.5]
    BKS = [125]*39
    #variance = [0.1, 0.15, 0.2]
    #open = [0, 0.4, 0.6, 0.8]
    #alpha = [False, True]

    variance = [0.1, 0.15]
    open = [0]
    alpha = [True]
    writer = pd.ExcelWriter('Output_def_2.xlsx', engine='xlsxwriter')
    for i in variance:
        for j in alpha:
            for k in open:

                #Nombre = "output/Var_"+str(i)+"_alpha_"+str(j)+ "_open_" + str(k)+"_2"+".xlsx"
                Nombre = str(i) + str(j) + str(k)

                print(Nombre)
                tabla(i, j, k, BKS, Nombre, writer)

    writer.save()

if __name__ == "__main__":
    all_tables()






