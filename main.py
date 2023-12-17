import Experiments.exp as exp

def Main():
    # The experiments are functions contained in Experiments/exp.py
    # For some experiments, raw data is directly printed,
    # for others, it is instead saved as .pkl file
    # (This depends on whether or not this eperiment has a related plot)

    # The following 7 functions print their data directly
    exp.QA_preddiff_trees()
    exp.QA_preddiff_forests()
    exp.four_trees_table()
    exp.AUC_table()
    exp.Forest_prediction_table()
    exp.RWD_table_trees()
    exp.RWD_table_forests()

    # The other functions are paired with pots
    exp.S_shift()
    exp.Plot_s_shift()

    exp.Heatmap_data()
    exp.Plot_heatmap()

    exp.n_bins_data()
    exp.Plot_n_bins()

    exp.Forest_data()
    # Plot forest takes the parameter n
    exp.Plot_forest_comparison(1000)






if __name__ == "__main__":
    Main()