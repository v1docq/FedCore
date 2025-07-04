import json, threading, os
from collections import OrderedDict, namedtuple
import torch, tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit


class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batches, cache_file):
        super().__init__()
        self.batches, self.idx = batches, 0
        self.cache_file = str(cache_file)
        self.device_mem = cuda.mem_alloc(batches[0].nbytes)

    def get_batch_size(self):
        return self.batches[0].shape[0]

    def get_batch(self, names):
        if self.idx >= len(self.batches):
            return None
        cuda.memcpy_htod(self.device_mem, np.ascontiguousarray(self.batches[self.idx]))
        self.idx += 1
        return [int(self.device_mem)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)



class TensorRTInferenceModel(torch.nn.Module):
    def __init__(self, engine_path: str):
        super().__init__()
        self.device = torch.device("cuda")
        self.lock = threading.Lock()
        self.logger = trt.Logger(trt.Logger.INFO)

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            meta_len = int.from_bytes(f.read(4), "little")
            self.metadata = json.loads(f.read(meta_len).decode())
            self.engine = rt.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        Bind = namedtuple("Bind", ("dtype", "tensor", "ptr"))
        self.bindings = OrderedDict()
        self.input_name = self.engine.get_tensor_name(0)  # один вход
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            tensor = torch.empty(tuple(shape), dtype=torch.float32, device=self.device)
            self.bindings[name] = Bind(trt.nptype(self.engine.get_tensor_dtype(name)),
                                       tensor, int(tensor.data_ptr()))
        self.addr_list = list(b.ptr for b in self.bindings.values())

    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        with self.lock:
            self.context.set_input_shape(self.input_name, tuple(x.shape))
            self.addr_list[0] = int(x.data_ptr())
            self.context.execute_v2(self.addr_list)
            outputs = [bind.tensor.clone().detach().cpu() for bind in list(self.bindings.values())[1:]]
        return outputs if len(outputs) > 1 else outputs[0]

    def to(self, *_):  # no‑op
        return self

    def size(self):
        return os.path.getsize(self.engine_path)