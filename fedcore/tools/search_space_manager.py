"""RL-агент для адаптивного сужения пространства поиска гиперпараметров.

Модуль реализует агента обучения с подкреплением (PPO), который обучается
модифицировать пространство поиска гиперпараметров (например,
``fedcore_search_space`` из ``fedcore.interfaces.search_space``) на основе
обратной связи от метрик моделей, хранящихся в ``ModelRegistry``.

Ключевые компоненты:
    * :func:`vectorize_search_space` — векторизация пространства поиска для
      совместимости с фреймворками RL (Gymnasium);
    * :class:`HyperparamSearchEnv` — среда Gymnasium, в которой действия агента
      модифицируют области выборки гиперпараметров (исключение опций, сдвиг,
      сжатие и сброс диапазонов);
    * :class:`SearchSpaceManager` — управляющий модуль, инкапсулирующий
      инициализацию и итеративные шаги взаимодействия со средой.

Примечание: gymnasium и stable-baselines3 являются опциональными
зависимостями; при их отсутствии модуль импортируется, но создание среды
и агента невозможно.
"""

import copy
import random

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import Env, spaces
except ImportError:  # опциональная зависимость
    gym = None
    Env = object
    spaces = None

from fedcore.tools.registry.model_registry import ModelRegistry

__all__ = [
    "vectorize_search_space",
    "HyperparamSearchEnv",
    "SearchSpaceManager",
]


def vectorize_search_space(search_space):
    vec = []
    for module in search_space.values():
        for param in module.values():
            scope = param["sampling-scope"][0]
            if param["type"] == "categorical":
                vec.append(len(scope))  # Количество категориальных опций
            elif param["type"] == "continuous":
                vec.extend([scope.min(), scope.max()])
            else:
                raise ValueError(f"Неизвестный тип параметра: {param['type']}")
    return np.array(vec, dtype=np.float32)


class HyperparamSearchEnv(Env):
    """Среда Gymnasium для адаптивной модификации пространства поиска гиперпараметров.

    Действия агента зависят от режима работы (списковый или параметрический)
    и типа выбранного параметра. В списковом режиме действие — индекс элемента
    для удаления из текущего пространства. В параметрическом режиме действие
    раскладывается на индекс параметра и тип операции (исключение, сдвиг,
    сжатие, сброс).

    Args:
        search_space: иерархическая структура пространства поиска
            (по умолчанию — ``fedcore_search_space``).
        max_steps: максимальное количество шагов эпизода.
        is_list_mode: списковый режим (наблюдения — популяция моделей).
    """

    N_ACTION_TYPES = 4  # число типов операций на параметр
    CONTINUOUS_SCOPE_POINTS = 15  # дискретизация непрерывных областей

    def __init__(self, search_space=None, max_steps=100, is_list_mode=False):
        if gym is None:
            raise ImportError(
                "gymnasium не установлен: среда HyperparamSearchEnv недоступна. "
                "Установите опциональную зависимость: pip install gymnasium"
            )
        if search_space is None:
            from fedcore.interfaces.search_space import fedcore_search_space

            search_space = fedcore_search_space
        self.is_list_mode = is_list_mode
        self.current_step = 0
        self.max_steps = max_steps
        self.initial_space = (
            self._normalize_scope(copy.deepcopy(search_space))
            if not is_list_mode
            else copy.deepcopy(search_space)
        )
        self.current_space = copy.deepcopy(self.initial_space)
        self.params = (
            [
                (module_name, param_name, param["type"])
                for module_name, module in search_space.items()
                for param_name, param in module.items()
            ]
            if not is_list_mode
            else []
        )
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        n_actions = (
            len(self.current_space)
            if self.is_list_mode
            else len(self.params) * self.N_ACTION_TYPES
        )
        self.action_space = spaces.Discrete(n_actions)

    @classmethod
    def _normalize_scope(cls, search_space):
        """Приводит области выборки к плоскому виду (см. листинг действий)."""
        for module in search_space.values():
            for param in module.values():
                scope = param["sampling-scope"]
                if param["type"] == "continuous":
                    if len(scope) == 1 and isinstance(scope[0], np.ndarray):
                        param["sampling-scope"] = scope[0]
                    else:
                        low, high = scope
                        param["sampling-scope"] = np.linspace(
                            low, high, cls.CONTINUOUS_SCOPE_POINTS
                        )
                elif param["type"] == "categorical" and len(scope) == 1:
                    param["sampling-scope"] = scope[0]
        return search_space

    def _get_obs(self):
        if self.is_list_mode:
            try:
                return np.array(self.current_space, dtype=np.float32)
            except (ValueError, TypeError):
                # неметрические наблюдения: используем только размер популяции
                return np.array([len(self.current_space)], dtype=np.float32)
        nested_view = {
            module_name: {
                param_name: {
                    **param,
                    "sampling-scope": [param["sampling-scope"]],
                }
                for param_name, param in module.items()
            }
            for module_name, module in self.current_space.items()
        }
        return vectorize_search_space(nested_view)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_space = copy.deepcopy(self.initial_space)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        if self.is_list_mode:
            if action < len(self.current_space):
                # Remove the point at index
                del self.current_space[action]
            return
        param_idx = action // 4
        action_type = action % 4
        module_name, param_name, param_type = self.params[param_idx]
        param = self.current_space[module_name][param_name]
        if action_type == 0 and param_type == "categorical":
            scope = param["sampling-scope"]
            if len(scope) > 1:
                exclude_idx = np.random.randint(len(scope))
                param["sampling-scope"] = np.delete(scope, exclude_idx)
        elif action_type == 1 and param_type == "continuous":
            scope = param["sampling-scope"]
            shift = (scope.max() - scope.min()) * 0.1
            direction = random.choice([-1, 1])
            param["sampling-scope"] = scope + direction * shift
        elif action_type == 2 and param_type == "continuous":
            scope = param["sampling-scope"]
            mid = (scope.min() + scope.max()) / 2
            half_range = (scope.max() - scope.min()) / 4
            param["sampling-scope"] = np.linspace(mid - half_range, mid + half_range, len(scope))
        elif action_type == 3:
            param["sampling-scope"] = copy.deepcopy(self.initial_space[module_name][param_name]["sampling-scope"])
        improvements = ModelRegistry().get_metrics_change_by_params(param)
        reward = np.mean(improvements)  # Агрегация n изменений в одну награду
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}


class SearchSpaceManager:
    """Управляющий модуль динамической оптимизации пространства поиска гиперпараметров.

    Инкапсулирует инициализацию среды и политики PPO, а также итеративные шаги
    взаимодействия с окружением, ориентированные на адаптивную эволюцию
    конфигураций на основе истории оптимизации из ``ModelRegistry``.

    Args:
        search_space: пространство поиска для среды
            (по умолчанию — ``fedcore_search_space``).
        max_steps: максимальное количество шагов эпизода среды.
        is_list_mode: списковый режим среды.
        **ppo_kwargs: дополнительные аргументы конструктора ``PPO``.
    """

    def __init__(self, search_space=None, max_steps=100, is_list_mode=False, **ppo_kwargs):
        if gym is None:
            raise ImportError(
                "gymnasium не установлен: SearchSpaceManager недоступен. "
                "Установите опциональные зависимости: pip install gymnasium stable-baselines3"
            )
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 не установлен: SearchSpaceManager недоступен. "
                "Установите опциональную зависимость: pip install stable-baselines3"
            ) from exc
        self.env = HyperparamSearchEnv(
            search_space=search_space, max_steps=max_steps, is_list_mode=is_list_mode
        )
        self.model = PPO("MlpPolicy", self.env, **ppo_kwargs)

    @property
    def search_space(self):
        """Текущее (возможно, модифицированное агентом) пространство поиска."""
        return self.env.current_space

    def step(self, obs):
        self.model.learn(total_timesteps=10)
        for ob in obs:
            action, _ = self.model.predict(ob, ModelRegistry().get_metrics_change_by_params(ob))
            self.env.step(action)
