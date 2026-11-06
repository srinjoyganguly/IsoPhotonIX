"""Time a native solver stage. All step scopes must synchronize the returned JAX state."""
from contextlib import contextmanager
import ctypes as ct
from importlib.metadata import distribution, PackageNotFoundError
from math import prod
from time import perf_counter
from uuid import uuid4


class _CudaEventClock:
    """Bracket completed batches on the GPU clock without registering profiler callbacks.

    The start event completes before dispatch. The end event is submitted only after
    JAX synchronizes the native solver state. These are elapsed intervals, including
    host launch gaps and GPU idle intervals inside each batch.
    """
    def __init__(self, target):
        package = distribution('nvidia-cuda-runtime-cu12')
        self.lib = ct.CDLL(str(package.locate_file('nvidia/cuda_runtime/lib/libcudart.so.12')))
        self.lib.cudaGetErrorString.restype = ct.c_char_p
        device = ct.c_int()
        self.check(self.lib.cudaGetDevice(ct.byref(device)))
        if device.value != target.local_hardware_id:
            raise ValueError('CUDA clock and JAX use different device ordinals.')
        self.stream = ct.c_void_p()
        self.start, self.end = ct.c_void_p(), ct.c_void_p()
        try:
            self.check(self.lib.cudaStreamCreateWithFlags(ct.byref(self.stream),1))
            self.check(self.lib.cudaEventCreate(ct.byref(self.start)))
            self.check(self.lib.cudaEventCreate(ct.byref(self.end)))
        except Exception:
            self.close()
            raise

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.cudaGetErrorString(code).decode())

    def begin(self):
        self.check(self.lib.cudaEventRecord(self.start,self.stream))
        self.check(self.lib.cudaEventSynchronize(self.start))

    def elapsed(self):
        self.check(self.lib.cudaEventRecord(self.end,self.stream))
        self.check(self.lib.cudaEventSynchronize(self.end))
        ms = ct.c_float()
        self.check(self.lib.cudaEventElapsedTime(ct.byref(ms),self.start,self.end))
        return ms.value / 1000

    def close(self):
        for event in (self.end,self.start):
            if event.value:
                self.lib.cudaEventDestroy(event)
        if self.stream.value:
            self.lib.cudaStreamDestroy(self.stream)


_optional_errors = (OSError, ValueError, RuntimeError, PackageNotFoundError, AttributeError)


class StageTiming:
    def __init__(self, target):
        self.target = target
        self.started = perf_counter()
        self.compile_s = self.step_s = 0.
        self.step_calls = 0
        self.events = None
        self.event_s = None
        self.note = 'CPU execution' if target.platform != 'gpu' else None
        self.stage_id = uuid4().hex
        self.finished = None

    @contextmanager
    def compile(self):
        start = perf_counter()
        try:
            yield
        finally:
            self.compile_s += perf_counter() - start

    def begin_steps(self):
        if self.target.platform == 'gpu':
            try:
                self.events = _CudaEventClock(self.target)
                self.event_s = 0.
            except _optional_errors as exc:
                self.note = str(exc)

    def drop_clock(self, exc):
        self.note = str(exc)
        self.event_s = None
        if self.events:
            self.events.close()
            self.events = None

    @contextmanager
    def step(self):
        if self.events:
            try:
                self.events.begin()
            except _optional_errors as exc:
                self.drop_clock(exc)
        start = perf_counter()
        try:
            yield
        finally:
            self.step_s += perf_counter() - start
            self.step_calls += 1
            if self.events:
                try:
                    self.event_s += self.events.elapsed()
                except _optional_errors as exc:
                    self.drop_clock(exc)

    def end_steps(self):
        if self.events:
            self.events.close()
            self.events = None
        self.finished = perf_counter()

    def report(self, *, shape, steps, dt_seconds, spacing_um, pml_cells):
        if self.finished is None or self.step_calls == 0:
            raise RuntimeError('Stage timing requires completed synchronized step scopes.')
        shape = [int(n) for n in shape]
        cells = prod(shape)
        return {'schema': 1, 'acquisition_id': self.stage_id, 'platform': self.target.platform,
                'device': str(self.target), 'device_kind': self.target.device_kind,
                'grid_shape': shape, 'cells': cells, 'spacing_um': float(spacing_um),
                'pml_cells': pml_cells, 'steps': int(steps), 'dt_seconds': float(dt_seconds),
                'simulated_time_fs': steps * float(dt_seconds) * 1e15, 'compile_seconds': self.compile_s,
                'synchronized_step_seconds': self.step_s, 'step_batches': self.step_calls,
                'cuda_event_elapsed_seconds': self.event_s, 'cuda_event_note': self.note,
                'kernel_active_seconds': None, 'kernel_count': None,
                'kernel_timing_note': 'Native stepping records elapsed time. Exclusive kernel-active time was not acquired.',
                'stage_through_steps_seconds': self.finished - self.started,
                'million_cell_steps_per_second': cells * steps / (1e6 * self.step_s)}
