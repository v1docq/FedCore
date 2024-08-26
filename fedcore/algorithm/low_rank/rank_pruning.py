import numpy as np

from fedcore.models.network_impl.layers import DecomposedConv2d


def rank_threshold_pruning(conv: DecomposedConv2d, threshold: float = 0.95, strategy: str = 'constant') -> None:
    """Prune the weight matrices to the threshold (in-place).
    Args:
        conv: The optimizable layer.
        threshold: hyperparameter must be in the range (0, 1].
    Raises:
        Assertion Error: If ``energy_threshold`` is not in (0, 1].
    """
    assert 0 < threshold <= 1, "energy_threshold must be in the range (0, 1]"
    if strategy.__contains__('constant'):
        S, indices = conv.S.sort()
        U = conv.U[:, indices]
        Vh = conv.Vh[indices, :]
        sum = (S ** 2).sum()
        threshold = threshold * sum
        for i, s in enumerate(S):
            sum -= s ** 2
            if sum < threshold:
                conv.set_U_S_Vh(U[:, i:].clone(), S[i:].clone(), Vh[i:, :].clone())
                break
    elif strategy.__contains__('explained_dispersion'):
        S, indices = conv.S.sort()
        U = conv.U[:, indices]
        Vh = conv.Vh[indices, :]
        singular_values = [abs(x) for x in S]
        n_components = [x / sum(singular_values) * 100 for x in singular_values]
        n_components = [x for x in n_components if x > threshold]
        n_components = len(n_components)
        conv.set_U_S_Vh(U[:, n_components:].clone(), S[n_components:].clone(), Vh[n_components:, :].clone())

    elif strategy.__contains__('median'):
        S, indices = conv.S.sort()
        U = conv.U[:, indices]
        Vh = conv.Vh[indices, :]
        median_sv = np.median(S)
        sv_threshold = 2.31 * median_sv
        n_components = max(np.sum(S >= sv_threshold), 1)
        conv.set_U_S_Vh(U[:, n_components:].clone(), S[n_components:].clone(), Vh[n_components:, :].clone())
