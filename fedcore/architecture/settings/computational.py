class BackendMethods:
    def __init__(self, device_type: str = "cpu"):
        self.backend = self.define_backend(device_type)

    def define_backend(self, device_type: str = "cpu"):
        if device_type == "CUDA":
            # TODO:
            import numpy
            import scipy.linalg

            return numpy, scipy.linalg
            # import cupy, cupyx.scipy.linalg
            # return cupy, cupyx.scipy.linalg
        else:
            import numpy
            import scipy.linalg

            return numpy, scipy.linalg



backend_methods, backend_scipy = BackendMethods("cpu").backend


def global_imports(
    object_name: str, short_name: str = None, context_module_name: str = None
):
    """Imports from local function as global import. Use this statement to import inside
    a function, but effective as import at the top of the module.

    Args:
        object_name: the object name want to import, could be module or function
        short_name: the short name for the import
        context_module_name: the context module name in the import

    Examples:
        Do this::
            import os -> global_imports("os")
            from fedot_ind.core.architecture.settings.computational import backend_methods
            as np -> global_imports("numpy", "np")
            from collections import Counter ->
                global_imports("Counter", None, "collections")
            from google.cloud import storage ->
                global_imports("storage", None, "google.cloud")

    """

    if not short_name:
        short_name = object_name
    if not context_module_name:
        globals()[short_name] = __import__(object_name)
    else:
        context_module = __import__(context_module_name, fromlist=[object_name])
        globals()[short_name] = getattr(context_module, object_name)
