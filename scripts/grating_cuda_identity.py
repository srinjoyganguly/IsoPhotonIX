"""Bind reusable native stages to the exact local CUDA device without publishing its UUID."""
import ctypes
import fcntl
import hashlib
import os


def cache_salt(cache):
    """Use a private per-checkpoint salt so device tags cannot be linked across stores."""
    cache.mkdir(exist_ok=True,parents=True)
    path=cache/'.device-salt'
    with (cache/'.device-salt.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if not path.exists():
            with os.fdopen(os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'wb') as file:
                file.write(os.urandom(32))
        value=path.read_bytes()
        if len(value)!=32:
            raise ValueError('The private checkpoint device salt is incomplete.')
        return value


def driver_device():
    """Use the CUDA driver ordinal selected by the worker, before allocating FDTD memory."""
    try:
        library=ctypes.CDLL('libcuda.so.1')
    except OSError as error:
        raise RuntimeError('A local NVIDIA CUDA driver is required for 3D grating studies.') from error
    def call(name,*args):
        status=getattr(library,name)(*args)
        if status!=0:
            raise RuntimeError(f'NVIDIA driver call {name} failed with status {status}.')
    call('cuInit',0)
    ordinal=ctypes.c_int()
    call('cuDeviceGet',ctypes.byref(ordinal),0)
    uuid=ctypes.create_string_buffer(16)
    function='cuDeviceGetUuid_v2' if hasattr(library,'cuDeviceGetUuid_v2') else 'cuDeviceGetUuid'
    call(function,ctypes.byref(uuid),ordinal)
    name=ctypes.create_string_buffer(256)
    call('cuDeviceGetName',name,ctypes.sizeof(name),ordinal)
    major=ctypes.c_int();minor=ctypes.c_int();version=ctypes.c_int()
    call('cuDeviceGetAttribute',ctypes.byref(major),75,ordinal)
    call('cuDeviceGetAttribute',ctypes.byref(minor),76,ordinal)
    call('cuDriverGetVersion',ctypes.byref(version))
    return bytes(uuid),{'ordinal':0,'name':name.value.decode(),
                       'compute_capability':f'{major.value}.{minor.value}','driver_version':version.value}


def fingerprint(salt):
    if len(salt)!=32:
        raise ValueError('Expected a private 32-byte device salt.')
    uuid,metadata=driver_device()
    return {**metadata,'per_cache_device_sha256':hashlib.sha256(salt+uuid).hexdigest()}


def validate_worker(expected, actual):
    if expected!=actual:
        raise ValueError('NVIDIA device, driver or local cache identity changed during acquisition.')
