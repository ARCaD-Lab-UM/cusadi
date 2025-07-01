from cusadi import *
import torch
import importlib
import time

class CusadiTorchFunction:
    # Public variables:
    fn_casadi = None
    fn_name = None
    num_instances = 0
    inputs_sparse = []
    outputs_sparse = []
    outputs_dense = []

    # Private variables:
    _device = 'cuda'
    _fn_library = None
    _work_tensor = []
    _input_tensors = []
    _output_tensors = []
    _fn_input = []
    _fn_work = []
    _fn_output = []

    # ! Public methods:
    def __init__(self, fn_casadi, num_instances, dtype=torch.double):
        assert torch.cuda.is_available()
        if dtype == torch.double:
            self.dtype_str = 'double'
        elif dtype == torch.float:
            self.dtype_str = 'float'
        else:
            raise ValueError(f'Cusadi function dtypes can only be torch.double or torch.float, but found {self.dtype}')

        self.fn_casadi = fn_casadi
        self.fn_name = fn_casadi.name()
        self.num_instances = num_instances
        self.dtype = dtype
        self._fn_library = self._load_evaluate_fn(self.fn_name)
        print("Loaded CasADi function: ", self.fn_casadi)
        print("Loaded library: ", self._fn_library)
        self._setup()

    def evaluate(self, inputs):
        # Validate inputs
        for i, t in enumerate(inputs):
            # Check dtype
            if t.dtype != self.dtype:
                raise ValueError(f"input {i} ({self.fn_casadi.name_in(i)}) is of dtype {t.dtype}, but it must match this function's dtype {self.dtype}")
            
            # Check device
            if not t.is_cuda:
                raise ValueError(f"input {i} ({self.fn_casadi.name_in(i)}) is not CUDA. Inputs to CusADi functions must be stored on GPU")
            
            # Check dimensions
            if t.ndim == 1:
                if t.shape != (self.num_instances,):
                    raise ValueError(f"input {i} ({self.fn_casadi.name_in(i)}) is of shape {t.shape}, but must be of shape {(self.num_instances,)}. Have you flattened and sparsified this input?")
            else:
                if t.shape != (self.num_instances, self.fn_casadi.nnz_in(i)):
                    raise ValueError(f"input {i} ({self.fn_casadi.name_in(i)}) is of shape {t.shape}, but must be of shape {(self.num_instances, self.fn_casadi.nnz_in(i))}. Have you flattened and sparsified this input?")
        
        self._clearTensors()
        self._prepareInputTensor(inputs)
        
        start_time = time.time()
        self._fn_library(
            self._output_tensors,
            self._input_tensors,
            self._work_tensor
        )
        torch.cuda.synchronize()
        self.eval_time = time.time() - start_time

    def getDenseOutput(self, out_idx = None):
        env_idx = torch.tensor(range(self.num_instances), device=self._device).repeat_interleave(self.fn_casadi.nnz_out(out_idx))
        row_idx = torch.tensor((self.fn_casadi.sparsity_out(out_idx).get_triplet()[0]), device=self._device) \
            .repeat(self.num_instances)
        col_idx = torch.tensor((self.fn_casadi.sparsity_out(out_idx).get_triplet()[1]), device=self._device) \
            .repeat(self.num_instances)
        dim_dense = (self.num_instances, self.fn_casadi.size1_out(out_idx), self.fn_casadi.size2_out(out_idx))
        return torch.sparse_coo_tensor(torch.vstack((env_idx, row_idx, col_idx)),
                                       self.outputs_sparse[out_idx].reshape(-1),
                                       dim_dense).to_dense()
    
    def checkInputDimensions(self, inputs):
        self.input_CPU = [tensor[0, :].cpu().numpy() for tensor in inputs]
        try :
            out = (self.fn_casadi.call(self.input_CPU)[0]).full()
            print("CPU call successful. Tensor dimensions are correct for inputs.")
        except:
            print("Error in Casadi function call. Exiting...")
            sys.exit(1)

    # ! Private methods:
    def _load_evaluate_fn(self, fn_name):
        lib_filepath = os.path.join(CUSADI_CODEGEN_DIR, f"{fn_name}.py")
        
        # Get module name from the file name
        module_name = os.path.splitext(os.path.basename(lib_filepath))[0]

        # Load the module from file
        spec = importlib.util.spec_from_file_location(module_name, lib_filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function from the module
        fn_library = getattr(module, f'evaluate_{self.fn_name}')
        
        return fn_library
    
    def _setup(self):
        self._input_tensors = [torch.zeros((self.num_instances, self.fn_casadi.nnz_in(i)),
                                            device=self._device, dtype=self.dtype).contiguous()
                               for i in range(self.fn_casadi.n_in())]
        self._output_tensors = [torch.zeros(self.num_instances, self.fn_casadi.nnz_out(i),
                                            device=self._device, dtype=self.dtype).contiguous()
                                for i in range(self.fn_casadi.n_out())]
        self._output_tensors_dense = [torch.zeros((self.num_instances,
                                                   self.fn_casadi.size1_out(i),
                                                   self.fn_casadi.size2_out(i)),
                                      device=self._device, dtype=self.dtype).contiguous()
                                      for i in range(self.fn_casadi.n_out())]
        self._work_tensor = torch.zeros((self.num_instances, self.fn_casadi.sz_w()),
                                        device=self._device, dtype=self.dtype).contiguous()
        self.inputs_sparse = self._input_tensors
        self.outputs_sparse = self._output_tensors
        self.outputs_dense = self._output_tensors_dense

    def _prepareInputTensor(self, inputs):
        for i in range(self.fn_casadi.n_in()):
            self._input_tensors[i] = inputs[i]
        self.inputs_sparse = self._input_tensors

    def _clearTensors(self):
        # for i in range(self.fn_casadi.n_in()):
        #     self._input_tensors[i].zero_()
        for i in range(self.fn_casadi.n_out()):
            self._output_tensors[i].zero_()
        self._work_tensor.zero_()