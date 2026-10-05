"""Real in-process Dask resources survive a caller-owned FedCore lifecycle."""
from distributed import Client, LocalCluster
from tests.refactoring.test_core_contracts import configuration, dataset, identity_model
from fedcore.api.main import FedCore


def test_real_caller_owned_dask_client_survives_training(tmp_path):
    model = identity_model()
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False,
                           protocol='inproc', dashboard_address=None, memory_limit=0)
    client = Client(cluster)
    try:
        api = FedCore(configuration(model, tmp_path), dask_client=client, dask_cluster=cluster)
        api.fit_no_evo(dataset(model))
        api.shutdown()
        assert client.status == 'running'
        assert len(client.scheduler_info()['workers']) == 1
        assert client.submit(sum, [1, 2, 3]).result(timeout=10) == 6
    finally:
        client.close()
        cluster.close()
