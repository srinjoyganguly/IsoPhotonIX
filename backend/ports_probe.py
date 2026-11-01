"""Port-specific runtime readiness; no electromagnetic solve is implied."""
import json
import os
import sys


def probe_ports(solver, device):
    # FDTDX's local mode callback prepares its eigenmode on CPU. Field evolution
    # still requires the explicitly requested device and never falls back.
    if solver == 'fdtdx' and device == 'gpu':
        os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    from .probe import probe
    result = probe(solver, device)
    if solver == 'meep':
        import meep as mp
        from meep import mpb
        if not hasattr(mp, 'EigenModeSource') or not hasattr(mpb.ModeSolver, 'find_k'):
            raise RuntimeError('Meep eigenmode source and MPB support are required for port studies.')
        result['preparation'] = 'MPB mode preparation and Meep FDTD run on CPU.'
    else:
        import fdtdx
        import jax
        from tidy3d.components.mode.solver import compute_modes
        for name in ('ModePlaneSource', 'PhasorDetector', 'EnergyDetector'):
            if not hasattr(fdtdx, name):
                raise RuntimeError(f'FDTDX {name} is required for port measurements.')
        if not callable(compute_modes) or not jax.devices('cpu'):
            raise RuntimeError('Local CPU mode preparation is unavailable.')
        result['preparation'] = ('Local source-mode preparation uses CPU; Maxwell evolution and fields use the selected '
                                 + ('NVIDIA GPU.' if device == 'gpu' else 'CPU.'))
    return result


if __name__ == '__main__':
    try:
        result = {'available': True, **probe_ports(*sys.argv[1:3])}
    except Exception as exc:
        result = {'available': False, 'reason': str(exc).strip() or type(exc).__name__}
    print('PROBE_JSON='+json.dumps(result))
