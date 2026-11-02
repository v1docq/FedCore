"""G1 adapter for the existing GOLEM/SPEA2 engine. No alternate searcher."""
import math
import random
from .policy import SearchArm, mutate_with_policy


def common_initial(space, population, seed):
    """Uniform sample without replacement from the finite canonical product."""
    from .cv import HeterogeneousGenotype, HeterogeneousGene, validate_space
    from fedcore.algorithm.low_rank.plans import ValidationFailure
    failure = validate_space(space)
    if type(failure) is ValidationFailure:
        raise ValueError(str(failure))
    total = math.prod(len(d.alternatives) for d in space.domains)
    if type(population) is not int or not 1 <= population <= total or type(seed) is not int or seed < 0:
        raise ValueError('Require a seed and enough distinct canonical initial genotypes')
    result = []
    for code in random.Random(seed).sample(range(total), population):
        genes = []
        for domain in space.domains:
            code, index = divmod(code, len(domain.alternatives))
            genes.append(HeterogeneousGene(domain.path, domain.alternatives[index]))
        result.append(HeterogeneousGenotype(tuple(genes)))
    return tuple(result)


def run_search(space, config, evaluate, *, arm, fitness_context,
               initial_genotypes=None, context_guard=None, **options):
    from .cv import (validate_space, normalize_genotype, generate_genotype,
                     crossover_genotypes, compile_genotype, genotype_payload)
    from fedcore.algorithm.low_rank.grammar_search import GrammarAdapter, run_grammar_search
    from fedcore.algorithm.low_rank.plans import ValidationFailure
    from .contracts import fingerprint
    if type(arm) is not SearchArm:
        raise ValueError('The heterogeneous panel accepts E0/E1/E2 only')
    if config.max_evaluations <= config.population_size:
        raise ValueError('The paid budget must exceed the shared initial population')
    if any(key in options for key in ('grammar_adapter','charge_before_compile','strict_deadline')):
        raise ValueError('G1 always charges before compilation and excludes late fitness')
    initial_genotypes = (common_initial(space, config.population_size, config.seed)
                         if initial_genotypes is None else tuple(initial_genotypes))
    normalized = tuple(normalize_genotype(g, space) for g in initial_genotypes)
    if (len(normalized) != config.population_size or any(type(g) is ValidationFailure for g in normalized)
            or len({fingerprint(g) for g in normalized}) != config.population_size):
        raise ValueError('The shared paid prefix must contain exactly population_size distinct valid plans')
    initial_genotypes = normalized

    def plan(parent, domain, **context):
        return mutate_with_policy(parent, domain, arm=arm, **context)

    adapter = GrammarAdapter(validate_space, normalize_genotype, generate_genotype,
        # Used only by legacy API contracts; the explicit planner owns G1 variation.
        lambda parent, domain, **kw: mutate_with_policy(parent, domain, arm=arm,
            seed=kw['seed'], charged=config.population_size,
            maximum=config.max_evaluations, initial=config.population_size).genotype,
        crossover_genotypes, compile_genotype, genotype_payload,
        domain='heterogeneous CIFAR G1; Evo-only; fixed recovery outside genotype',
        context_guard=context_guard, mutation_planner=plan)
    result = run_grammar_search(space, config, evaluate, fitness_context=fitness_context,
        initial_genotypes=initial_genotypes, grammar_adapter=adapter,
        charge_before_compile=True, strict_deadline=True, **options)
    result['search_arm'] = arm.value
    result['selection_contract'] = 'SPEA2 unchanged; family archive diagnostic only'
    return result
