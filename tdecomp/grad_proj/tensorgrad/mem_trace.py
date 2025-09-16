import torch
import numpy as np
from pathlib import Path

import socket
from datetime import datetime
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

from torch.profiler._memory_profiler import MemoryProfileTimeline

categories = {"PARAMETER": 0,
              "OPT": 1,
              "INPUT": 2,
              "TEMP": 3,
              "ACTIVATION": 4,
              "GRADS": 5,
              "AUTOGRAD_DETAIL": 6,}
              #"INTERMEDIATE_ACTIVATION": 7} # Nones/7 are reservations without allocation. Drop them. 
GB = 1024**3
device_str = "cuda:0"

class TimelineTracer(object):
    def __init__(self, out_file):
        super().__init__()
        print(f"Created TimelineTracer with out file {out_file}")
        self.out_file = out_file

        self._totals = {c: 0. for c in categories.keys()}
        self._totals["INTERMEDIATE"] = 0.
        self.steps = 0

    def __call__(self, prof):
        self.steps += 1
        # export raw memory timeline
        mem_tl = MemoryProfileTimeline(prof._memory_profile())
        times, sizes = mem_tl._coalesce_timeline(device_str)
        times = np.array(times)
        sizes = np.array(sizes)
        #sizes = sizes[:, :8] # Drop Nones - they correspond to reservation not allocation

        t_min = min(times)
        times -= t_min
        device = torch.device(device_str)

        msg= f"Memory Breakdown at Peak for {device_str}\n---------\n"
        #max_memory_allocated = torch.cuda.max_memory_allocated(device)
        #max_memory_reserved = torch.cuda.max_memory_reserved(device)
        
        #msg += f"Max CUDA reserved (GB): {max_memory_reserved / GB :.2f}\n"

        # Compute totals, find peak usage, collect values
        ## First, we find the true peak memory usage.
        true_totals = np.sum(sizes, axis=1)
        true_peak_mem_ts = np.argmax(true_totals)
        true_peak_mem = true_totals[true_peak_mem_ts] / GB

        ## Then, we find the peak timestep without taking Nones into account
        ## to perform a more correct breakdown of all other memory categories 
        ## as the peak occurs before the memory is tagged properly
        ## None: index 8 in table
        breakdown_totals = np.sum(sizes[:,:8], axis=1)
        breakdown_peak_mem_ts = np.argmax(breakdown_totals)
        breakdown_peak_mem = breakdown_totals[breakdown_peak_mem_ts] / GB

        for key, idx in categories.items():
            key_peak_mem = sizes[breakdown_peak_mem_ts, idx+1]
            self._totals[key] += key_peak_mem

            msg += f"Max {key} (GB): {(self._totals[key] / self.steps) / GB :.2f}\n"
        
        ## To find the true amount of "None" / intermediate memory, we subtract 
        ## the total found from the second "properly tagged peak" from the total
        ## found at the TRUE peak to compute the true value of "intermediate activation"
        ## (aka intermediates, temps, improperly tagged memory and unallocated but reserved memory.)
        intermediate_mem = true_peak_mem - breakdown_peak_mem
        self._totals["INTERMEDIATE"] += intermediate_mem
        intermed_total = self._totals["INTERMEDIATE"]

        msg += f"Max INTERMEDIATE (GB): {(intermed_total / self.steps) / GB :.2f}\n"

        msg += f"Peak total memory (GB): {true_peak_mem:.2f}\n"
        with open(self.out_file, "w") as f:
            f.write(msg)
        f.close()

def trace_handler(prof):
    # Current timestamp to distinguish trace files
    time_stamp = datetime.now().strftime(TIME_FORMAT_STR)
    host_name = socket.gethostname()
    
    # Create output directory for traces if it doesn't exist
    out_dir = Path("./memory_traces")
    out_dir.mkdir(exist_ok=True)
    
    # Create output file path
    out_file = out_dir / f"mem_trace_{host_name}_{time_stamp}.txt"
    
    # Use the TimelineTracer to generate the memory report
    tracer = TimelineTracer(out_file)
    tracer(prof)
    
    return

def trace_handler_obj(run_name, out_dir=None):
    """
    Returns a trace handler function that saves traces with the given run_name
    """
    if out_dir is None:
        out_dir = "./memory_traces"
    
    # Create output directory
    out_dir = Path(out_dir) / "memory_traces"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a function that will be called by the profiler
    def handler(prof):
        time_stamp = datetime.now().strftime(TIME_FORMAT_STR)
        host_name = socket.gethostname()
        
        # Create output file path with run name
        out_file = out_dir / f"{run_name}_mem_trace_{host_name}_{time_stamp}.txt"
        
        # Use the TimelineTracer to generate the memory report
        tracer = TimelineTracer(out_file)
        tracer(prof)
    
    return handler
