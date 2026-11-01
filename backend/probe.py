"""Executed in the chosen solver interpreter; deliberately independent of FastAPI."""
import importlib.metadata
import json
import sys


def probe(solver, target):
    if solver == "meep":
        if target != "cpu":
            raise RuntimeError("Meep GPU is unsupported.")
        import meep
        return {"version": meep.__version__, "device_name": "CPU", "actual_platform": "cpu"}
    import jax
    import jax.numpy as jnp
    import fdtdx
    expected = "gpu" if target == "gpu" else "cpu"
    devices = jax.devices(expected)
    if not devices or devices[0].platform != expected:
        raise RuntimeError(f"Requested {expected} device did not initialize.")
    with jax.default_device(devices[0]):
        value = jax.jit(lambda x: x*x+1)(jnp.array([2.0])).block_until_ready()
    if any(d.platform != expected for d in value.devices()):
        raise RuntimeError("Probe executed on a different device; fallback refused.")
    return {"version": importlib.metadata.version("fdtdx"), "jax_version": jax.__version__,
            "device_name": getattr(devices[0], "device_kind", str(devices[0])), "actual_platform": devices[0].platform}


if __name__ == "__main__":
    try:
        result = {"available": True, **probe(*sys.argv[1:3])}
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        if sys.argv[1:3] == ["fdtdx", "gpu"]:
            detail = "CUDA could not initialize. Check JAX/CUDA packages, the NVIDIA driver, and WSL GPU access. No CPU fallback was attempted. " + detail
        result = {"available": False, "reason": detail}
    print("PROBE_JSON="+json.dumps(result))
