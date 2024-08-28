import torch
from fedcore.models.network_impl.layers import DecomposedConv2d


def rank_threshold_pruning(conv: DecomposedConv2d, threshold: float = 0.95,
                           strategy: str = 'constant', module_name: str ='conv') -> None:
    """Prune the weight matrices to the threshold (in-place).
    Args:
        conv: The optimizable layer.
        threshold: hyperparameter must be in the range (0, 1].
    Raises:
        Assertion Error: If ``energy_threshold`` is not in (0, 1].
    """
    assert 0 < threshold <= 1, "energy_threshold must be in the range (0, 1]"
    S, indices = conv.S.sort()
    U = conv.U[:, indices]
    Vh = conv.Vh[indices, :]
    if strategy.__contains__('constant'):
        sum = (S ** 2).sum()
        threshold = threshold * sum
        for i, s in enumerate(S):
            sum -= s ** 2
            if sum < threshold:
                n_components = i
                break
    elif strategy.__contains__('explained_dispersion'):
        singular_values = S.cpu().detach().numpy()
        singular_values = [abs(x) for x in singular_values]
        n_components = [x / sum(singular_values) * 100 for x in singular_values]
        explained_dispersion = 0
        for index, comp in enumerate(n_components):
            explained_dispersion += comp
            if explained_dispersion > threshold*100:
                n_components = index
                break
    elif strategy.__contains__('median'):
        median_sv = torch.median(S)
        sv_threshold = 2.31 * median_sv
        n_components = max(torch.sum(S >= sv_threshold).cpu().detach().numpy().min(), 1)
        n_components = indices.cpu().detach().numpy().max() - n_components
    conv.set_U_S_Vh(U[:, n_components:].clone(), S[n_components:].clone(), Vh[n_components:, :].clone())
    print('After rank pruning left only {} % of {} layer params'.format(
        100 * n_components / len(indices), module_name))
