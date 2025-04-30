
from datetime import datetime
from glob import glob
from os import remove
from typing import Union

import pandas as pd
import seaborn as sns

from golem.core.dag.graph import Graph
from golem.core.log import default_log
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.paths import default_data_dir
from golem.visualisation.graph_viz import GraphVisualizer
from PIL import Image

from fedcore.architecture.utils.paths import PROJECT_PATH
from fedcore.tools.visualisation.vis_utils import *


class OptHistoryVisualizer:
    """ Implements legacy history visualizations that are not available via `history.show()`
    Args:
        history: history of optimisation
        folder: path to folder to save results of visualization
    """

    def __init__(self, history: Union[OptHistory,str], folder: Optional[str] = default_data_dir()):
        self.save_path = os.path.join(PROJECT_PATH, folder)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if isinstance(history, str):
            self.history = OptHistory().load(history)
        else:
            self.history = history
        self.log = default_log(self)
        self.graphs_imgs = []
        self.convergence_imgs = []
        self.best_graphs_imgs = []
        self.merged_imgs = []
        self.graph_visualizer = GraphVisualizer

    def pareto_gif_create(self,
                          objectives_numbers: Tuple[int, int] = (0, 1),
                          objectives_names: Tuple[str] = None):
        files = []
        pareto_fronts = self.history.archive_history
        individuals = self.history.generations
        if objectives_names is None:
            objectives_names = self.history.objective.metric_names
        array_for_analysis = individuals if individuals else pareto_fronts
        all_objectives = extract_objectives(array_for_analysis, objectives_numbers)
        min_x, max_x = min(all_objectives[0]) - 0.01, max(all_objectives[0]) + 0.01
        min_y, max_y = min(all_objectives[1]) - 0.01, max(all_objectives[1]) + 0.01
        folder = f'{self.save_path}'
        for i, front in enumerate(pareto_fronts):
            file_name = f'pareto{i}.png'
            visualise_pareto(front, file_name=file_name, save=True, show=False,
                             folder=folder, generation_num=i, individuals=individuals[i],
                             minmax_x=[min_x, max_x], minmax_y=[min_y, max_y],
                             objectives_numbers=objectives_numbers,
                             objectives_names=objectives_names)
            files.append(f'{folder}/{file_name}')

        create_gif_using_images(gif_path=f'{folder}/pareto_history.gif', files=files)

    def _visualise_graphs(self, graphs: List[Graph], fitnesses: List[float]):
        fitnesses = deepcopy(fitnesses)
        last_best_graph = graphs[0]
        prev_fit = fitnesses[0]
        fig = plt.figure(figsize=(10, 10))
        for ch_id, graph in enumerate(graphs):
            self.graph_visualizer(graph).draw_nx_dag()
            fig.canvas.draw()
            img = figure_to_array(fig)
            self.graphs_imgs.append(img)
            plt.clf()
            if fitnesses[ch_id] > prev_fit:
                fitnesses[ch_id] = prev_fit
            else:
                last_best_graph = graph
            prev_fit = fitnesses[ch_id]
            plt.clf()
            self.graph_visualizer(last_best_graph).draw_nx_dag()
            fig.canvas.draw()
            img = figure_to_array(fig)
            self.best_graphs_imgs.append(img)
            plt.clf()
        plt.close('all')

    def _visualise_convergence(self, fitness_history):
        fitness_history = deepcopy(fitness_history)
        prev_fit = fitness_history[0]
        for fit_id, fit in enumerate(fitness_history):
            if fit > prev_fit:
                fitness_history[fit_id] = prev_fit
            prev_fit = fitness_history[fit_id]
        ts_set = list(range(len(fitness_history)))
        df = pd.DataFrame(
            {'ts': ts_set, 'fitness': [-f for f in fitness_history]})

        fig = plt.figure(figsize=(10, 10))
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['axes.labelsize'] = 20
        for ts in ts_set:
            plt.plot(df['ts'], df['fitness'], label='Optimizer')
            plt.xlabel('Evaluation', fontsize=18)
            plt.ylabel('Best metric', fontsize=18)
            plt.axvline(x=ts, color='black')
            plt.legend(loc='upper left')
            fig.canvas.draw()
            img = figure_to_array(fig)
            self.convergence_imgs.append(img)
            plt.clf()
        plt.close('all')

    def visualise_history(self, metric_index: int = 0):
        try:
            self._clean(with_gif=True)
            all_historical_fitness = self.history.all_historical_quality(metric_index)
            historical_graphs = [ind.graph
                                 for ind in list(itertools.chain(*self.history.generations))]
            self._visualise_graphs(historical_graphs, all_historical_fitness)
            self._visualise_convergence(all_historical_fitness)
            self._merge_images()
            self._combine_gifs()
            self._clean()
        except Exception as ex:
            self.log.error(f'Visualisation failed with {ex}')

    def _merge_images(self):
        for i in range(1, len(self.graphs_imgs)):
            im1 = self.graphs_imgs[i]
            im2 = self.best_graphs_imgs[i]
            im3 = self.convergence_imgs[i]
            imgs = (im1, im2, im3)
            merged = np.concatenate(imgs, axis=1)
            self.merged_imgs.append(Image.fromarray(np.uint8(merged)))

    def _combine_gifs(self):
        date_time = datetime.now().strftime('%B-%d-%Y_%H-%M-%S_%p')
        save_path = os.path.join(self.save_path, f'history_visualisation_{date_time}.gif')
        imgs = self.merged_imgs[1:]
        self.merged_imgs[0].save(save_path, save_all=True, append_images=imgs,
                                 optimize=False, duration=DURATION, loop=0)
        self.log.info(f"Visualizations were saved to {save_path}")

    def _clean(self, with_gif=False):
        files = glob(f'{self.save_path}*.png')
        if with_gif:
            files += glob(f'{self.save_path}*.gif')
        for file in files:
            remove(file)

    def _create_boxplot(self, individuals: List[Any], generation_num: int = None,
                        objectives_names: Tuple[str] = ('ROC-AUC', 'Complexity'), file_name: str = 'obj_boxplots.png',
                        folder: str = None, y_limits: Tuple[float] = None):
        folder = f'{self.save_path}/boxplots' if folder is None else folder
        objectives = objectives_lists(individuals)
        df_objectives = pd.DataFrame({objectives_names[i]: objectives[i] for i in range(len(objectives))})
        fig, ax = plt.subplots(1, len(objectives))
        for idx, col in enumerate(df_objectives.columns):
            sns.boxplot(ax=ax[idx], data=df_objectives[col], palette="Blues", native_scale=True)
            ax[idx].set_title(f'Gen-{generation_num}.{col}')
            #plt.ylim(y_limits[0], y_limits[1])
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')
        if not os.path.isdir(f'{folder}'):
            os.mkdir(f'{folder}')
        path = f'{folder}/{file_name}'
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')

    def boxplots_gif_create(self, objectives_names: Tuple[str] = None):
        individuals = self.history.generations
        if objectives_names is None:
            objectives_names = self.history.objective.metric_names
        objectives = extract_objectives(individuals)
        objectives = list(itertools.chain(*objectives))
        min_y, max_y = min(objectives), max(objectives)
        files = []
        folder = f'{self.save_path}'
        for generation_num, individuals_in_genaration in enumerate(individuals):
            file_name = f'{generation_num}.png'
            self._create_boxplot(individuals_in_genaration, generation_num, objectives_names,
                                 file_name=file_name, folder=folder, y_limits=(min_y, max_y))
            files.append(f'{folder}/{file_name}')
        create_gif_using_images(gif_path=f'{folder}/boxplots_history.gif', files=files)
        plt.tight_layout()
        plt.cla()
        plt.clf()
        plt.close('all')