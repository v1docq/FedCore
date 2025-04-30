from fedcore.tools.visualisation.optimisation_history_vis import OptHistoryVisualizer

OPT_HISTORY = './history_90_min.json'
SAVE_FOLDER = 'examples/api_example/pruning/time_series_task/result_viz/history_vis'
if __name__ == "__main__":
    visualiser = OptHistoryVisualizer(history=OPT_HISTORY, folder=SAVE_FOLDER)
    visualiser.boxplots_gif_create()
    visualiser.pareto_gif_create()
    visualiser.visualise_history()
    print('End')
