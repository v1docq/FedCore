# from pygmo import hypervolume
import itertools
import os
from copy import deepcopy
from matplotlib.offsetbox import AnnotationBbox, TextArea
from typing import Any, List, Sequence, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt
from golem.core.optimisers.opt_history_objects.individual import Individual
from imageio import get_writer, v2

from fedcore.repository.constanst_repository import HISTORY_VIZ_PARAMS

DURATION = HISTORY_VIZ_PARAMS.frame_duration.value
PRUNING_VIZ_PARAMS = dict(pruning_model=HISTORY_VIZ_PARAMS.pruning_params.value)


def _get_pareto_features(front, objectives_numbers):
    pareto_obj_first, pareto_obj_second = [], []
    front_filtred_dict = {}
    for idx, ind in enumerate(front):
        fit_first = ind.fitness.values[objectives_numbers[0]]
        _ = {}
        pareto_obj_first.append(abs(fit_first))
        fit_second = ind.fitness.values[objectives_numbers[1]]
        pareto_obj_second.append(abs(fit_second))
        model_name = ind.graph.nodes[0].name
        model_params_dict = ind.graph.nodes[0].parameters
        for param in model_params_dict.keys():
            if param in PRUNING_VIZ_PARAMS[model_name]:
                param_val = model_params_dict[param]
                if isinstance(param_val, float):
                    param_val = round(param_val, 2)
                _.update({param: param_val})
        front_filtred_dict.update({idx: _})
    return pareto_obj_first, pareto_obj_second, front_filtred_dict


def _generate_params_box(pareto_dict, obj_first, obj_second, ax):
    # box_style = dict(boxstyle='round,pad=0.5', fc='white', ec='black', alpha=1.0)
    for idx, params_val in pareto_dict.items():
        textstr = str(params_val)
        textstr = textstr.replace(',', '\n')
        textstr = textstr.replace(':', ' - ')
        coord_x, coord_y = obj_first[idx], obj_second[idx]
        offsetbox = TextArea(textstr)
        ab = AnnotationBbox(offsetbox, (coord_x, coord_y),
                            xybox=(-20, 50),
                            xycoords='data',
                            boxcoords="offset points",
                            fontsize=6,
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)
    return ax


def visualise_pareto(front: Sequence[Individual],
                     objectives_numbers: Tuple[int, int] = (0, 1),
                     objectives_names: Sequence[str] = ('ROC-AUC', 'Complexity'),
                     file_name: str = 'result_pareto.png', show: bool = False, save: bool = True,
                     folder: str = '../../tmp/pareto',
                     generation_num: int = None,
                     individuals: Sequence[Individual] = None,
                     minmax_x: List[float] = None,
                     minmax_y: List[float] = None):
    fig, ax = plt.subplots()
    pareto_obj_first, pareto_obj_second, front_dict = _get_pareto_features(front, objectives_numbers)
    ax = _generate_params_box(front_dict, pareto_obj_first, pareto_obj_second, ax)
    if individuals is not None:
        non_pareto_obj_first, non_pareto_obj_second, non_front_dict = _get_pareto_features(individuals,
                                                                                           objectives_numbers)
        ax.scatter(non_pareto_obj_first, non_pareto_obj_second, c='green')
        ax = _generate_params_box(non_front_dict, non_pareto_obj_first, non_pareto_obj_second, ax)
    plt.plot(pareto_obj_first, pareto_obj_second, color='r')

    if generation_num is not None:
        ax.set_title(f'Pareto frontier, Generation: {generation_num}', fontsize=15)
    else:
        ax.set_title('Pareto frontier', fontsize=15)
    plt.xlabel(objectives_names[0], fontsize=15)
    plt.ylabel(objectives_names[1], fontsize=15)

    if minmax_x is not None:
        plt.xlim(minmax_x[0], minmax_x[1])
    if minmax_y is not None:
        plt.ylim(minmax_y[0], minmax_y[1])
    fig.set_figwidth(14)
    fig.set_figheight(14)
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')
        if not os.path.isdir(f'{folder}'):
            os.mkdir(f'{folder}')

        path = f'{folder}/{file_name}'
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()
    plt.cla()
    plt.clf()
    plt.tight_layout()
    plt.close('all')



def create_gif_using_images(gif_path: str, files: List[str]):
    try:
        with get_writer(gif_path, mode='I', duration=DURATION) as writer:
            for filename in files:
                image = v2.imread(filename)
                writer.append_data(image)
    except Exception:
        _ = 1


def extract_objectives(individuals: List[List[Any]], objectives_numbers: Tuple[int, ...] = None,
                       transform_from_minimization=True):
    if not objectives_numbers:
        objectives_numbers = [i for i in range(len(individuals[0][0].fitness.values))]
    all_inds = list(itertools.chain(*individuals))
    all_objectives = [[ind.fitness.values[i] for ind in all_inds] for i in objectives_numbers]
    if transform_from_minimization:
        transformed_objectives = []
        for obj_values in all_objectives:
            are_objectives_positive = all(np.array(obj_values) > 0)
            if not are_objectives_positive:
                transformed_obj_values = list(np.array(obj_values) * (-1))
            else:
                transformed_obj_values = obj_values
            transformed_objectives.append(transformed_obj_values)
    else:
        transformed_objectives = all_objectives
    return transformed_objectives


def figure_to_array(fig):
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def objectives_lists(individuals: List[Any], objectives_numbers: Tuple[int] = None):
    num_of_objectives = len(objectives_numbers) if objectives_numbers else len(individuals[0].fitness.values)
    objectives_numbers = objectives_numbers if objectives_numbers else [i for i in range(num_of_objectives)]
    objectives_values_set = [[] for _ in range(num_of_objectives)]
    for obj_num in range(num_of_objectives):
        for individual in individuals:
            value = individual.fitness.values[objectives_numbers[obj_num]]
            objectives_values_set[obj_num].append(value if value > 0 else -value)
    return objectives_values_set
