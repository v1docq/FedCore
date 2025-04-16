import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# from pygmo import hypervolume
from golem.visualisation.opt_viz_extra import OptHistoryExtraVisualizer
from fedot.core.repository.tasks import TaskTypesEnum, Task


def multiobjective_visualization(fedcore_cls, folder:str = './multiobjective_vis'):
    history = fedcore_cls.manager.solver.history
    objective_names = history.objective.metric_names
    visualiser = OptHistoryExtraVisualizer(history=history,folder=folder)
    visualiser.visualise_history()
    visualiser.pareto_gif_create(objectives_names=objective_names)
    visualiser.boxplots_gif_create()


def viz_pareto_fronts_comparison(fronts, labels, objectives_order=(1, 0),
                                 objectives_names=('ROC-AUC penalty metric', 'Computation time'),
                                 name_of_dataset=None, save=True):
    fig, ax = plt.subplots()
    current_palette = sns.color_palette('Dark2')
    for i, pareto_front in enumerate(fronts):
        color = np.array(current_palette[i])
        c = color.reshape(1, -1)
        ax.scatter(pareto_front[objectives_order[0]], pareto_front[objectives_order[1]], c=c, linewidths=4)
        ax.plot(pareto_front[objectives_order[0]], pareto_front[objectives_order[1]], color=color, label=labels[i],
                markersize=30, linewidth=2.5)
    plt.xlabel(objectives_names[objectives_order[0]], fontsize=20)
    plt.ylabel(objectives_names[objectives_order[1]], fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_title('Pareto frontiers', fontsize=20)
    plt.xticks(fontsize=20)
    ax.legend(loc='lower right', shadow=False, fontsize=20)
    fig.set_figwidth(9)
    fig.set_figheight(9)
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')

        file_name = name_of_dataset + '_pareto_fronts_comp.png'
        path = f'../../tmp/{file_name}'
        plt.savefig(path, bbox_inches='tight')

    plt.show()

# def viz_hv_comparison(labels, iterations, all_history_report, name_of_dataset='None',
#                       color_pallete=sns.color_palette("husl", 8), task: Task = Task(TaskTypesEnum.classification)):
#     all_history_report_transformed = [
#         [[front.items for front in comp_run.history.archive_history] for comp_run in hist] for hist in
#         all_history_report]
#     fitness_history_gp = [[[[[1 + it.fitness.values[0], it.fitness.values[1]] for it in front.items] for front in
#                             comp_run.history.archive_history] for comp_run in hist] for hist in all_history_report]
#
#     inds_history_gp = all_history_report_transformed
#
#     ref = [[], []]
#     for exp_history in inds_history_gp:
#         max_qual, max_compl = [], []
#         for run_history in exp_history:
#             all_objectives = objectives_transform(run_history, objectives_numbers=(0, 1),
#                                                   transform_from_minimization=True)
#             max_qual.append(max(all_objectives[0]) + 0.0001)
#             max_compl.append(max(all_objectives[1]) + 0.0001)
#         ref[0].append(max(max_qual))
#         ref[1].append(max(max_compl))
#     ref_point = (max(ref[0]), max(ref[1]))
#
#     hv_set = []
#     for exp_num, exp_history in enumerate(fitness_history_gp):
#         hv_set.append([])
#         for run_num, run_history in enumerate(exp_history):
#             hv_set[exp_num].append(
#                 [hypervolume(pop).compute(ref_point) for pop in fitness_history_gp[exp_num][run_num]])
#
#     show_history_optimization_comparison(optimisers_fitness_history=hv_set,
#                                          iterations=[_ for _ in range(iterations)],
#                                          labels=labels, color_pallete=color_pallete, ylabel='Hypervolume',
#                                          name_of_dataset=name_of_dataset, task=task)
#
#     try:
#         path_to_save_hv = name_of_dataset + '_hv_set_gp'
#         np.save(path_to_save_hv, hv_set)
#     except Exception as ex:
#         print(ex)
