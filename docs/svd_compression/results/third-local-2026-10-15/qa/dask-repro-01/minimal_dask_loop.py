"""Read-only, bounded Dask event-loop diagnostic; no FedCore imports."""
import faulthandler,json,sys,time
import dask,distributed
from distributed.utils import LoopRunner
from distributed.config import get_loop_factory
print(json.dumps({'python':sys.version,'dask':dask.__version__,'distributed':distributed.__version__,'loop_factory':get_loop_factory().__name__}),flush=True)
faulthandler.enable()
faulthandler.dump_traceback_later(3,repeat=False)
started=time.perf_counter()
runner=LoopRunner()
try:
    runner.start()
    print(json.dumps({'status':'started','seconds':time.perf_counter()-started}),flush=True)
    runner.stop()
except BaseException as error:
    print(json.dumps({'status':'failed','type':type(error).__name__,'message':str(error),'seconds':time.perf_counter()-started}),flush=True)
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
