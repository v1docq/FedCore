"""Optional observations of the actual FedCore optimizer, without changing selection."""
import math
from pathlib import Path
from time import perf_counter
from fedcore.tools.atomic_json import write_atomic_json


def individual_record(individual):
    fitness = individual.fitness
    values = getattr(fitness, 'values', ()) or ()
    values = [float(value) if value is not None and math.isfinite(float(value)) else None
              for value in values]
    parent = getattr(individual, 'parent_operator', None)
    return {
        'uid': str(individual.uid), 'graph': individual.graph.descriptive_id,
        'fitness': values, 'valid': bool(fitness.valid),
        'variation': None if parent is None else {
            'type': str(parent.type_),
            'operators': [getattr(op, '__name__', str(op)) for op in parent.operators],
            'parents': [str(item.uid) for item in parent.parent_individuals]},
    }


class SearchTrace:
    """Batch wall time includes cache lookup; it is not per-candidate compute time."""
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.events = []

    def record(self, event, **fields):
        if self.path is None:
            return
        self.events.append({'event': event, **fields})
        write_atomic_json(self.path, {
            'schema_version': 1, 'engine': 'FedcoreEvoOptimizer',
            'hypervolume_role': 'not_used_by_this_optimizer',
            'events': self.events})

    def evaluator(self, evaluate):
        def observed(individuals):
            if self.path is None:
                return evaluate(individuals)
            submitted = [individual_record(item) for item in individuals]
            start = perf_counter()
            try:
                result = evaluate(individuals)
            except Exception as error:
                self.record('evaluation_batch', submitted=submitted, returned=[],
                            elapsed_seconds=perf_counter() - start,
                            error_type=type(error).__name__)
                raise
            self.record('evaluation_batch', submitted=submitted,
                        returned=[individual_record(item) for item in (result or [])],
                        elapsed_seconds=perf_counter() - start, error_type=None)
            return result
        return observed
