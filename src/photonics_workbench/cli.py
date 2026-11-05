"""Launch the installed interface or inspect its local solver targets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog='photonics-workbench')
    commands = parser.add_subparsers(dest='command', required=True)
    web = commands.add_parser('web', help='serve the local workbench on 127.0.0.1')
    web.add_argument('--port', type=int, default=8000)
    doctor = commands.add_parser('doctor', help='probe Meep CPU and FDTDX CPU/NVIDIA targets')
    doctor.add_argument('--json', action='store_true', help='print machine-readable probe results')
    run = commands.add_parser('run', help='run an exported study configuration in its selected solver')
    run.add_argument('config', type=Path)
    run.add_argument('--output', required=True, type=Path, help='new directory for the result')
    references = commands.add_parser('compare-examples',help='reproduce the published component comparisons with local MPB')
    references.add_argument('--output',required=True,type=Path,help='new directory for measured results and refinement records')
    references.add_argument('--case',choices=('all','lumerical-coupler','tidy3d-strip-plugin'),default='all')
    references.add_argument('--resolutions',nargs='+',type=int,help='increasing grid densities, 24–256 cells/µm')
    setup = commands.add_parser('setup', help='install a managed local numerical environment')
    setup.add_argument('profile', choices=['meep'])
    setup.add_argument('--conda', help='path to the Conda executable')
    setup.add_argument('--offline', action='store_true', help='use the existing Conda package cache')
    for command in (web, doctor, setup, run, references):
        command.add_argument('--data-dir', type=Path, help='location for local runs, settings and managed environments')
    args = parser.parse_args(argv)
    if args.command == 'web' and not 1 <= args.port <= 65535:
        parser.error('Choose a port from 1 to 65535.')
    configured = os.environ.get('PHOTONICS_DATA_DIR')
    default_data = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local/share') / 'photonics-workbench'
    data = (args.data_dir or Path(configured or default_data)).expanduser().resolve()
    resources = Path(__file__).resolve().parent / '_app'
    if not (resources / 'frontend/dist/index.html').is_file():
        parser.exit(1, 'The compiled interface is missing. Reinstall the built Photonics Workbench wheel.\n')
    os.environ['PHOTONICS_DATA_DIR'] = str(data)
    data.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The application source lives in a private package resource tree. Numerical
    # subprocesses run with that tree as their cwd and resolve the same sources.
    sys.path.insert(0, str(resources))
    if args.command == 'web':
        import fcntl
        import uvicorn
        # A second server with the same run store could recover a still-active
        # worker as interrupted. Claim the store before the application starts.
        state = data / '.local'
        state.mkdir(parents=True, exist_ok=True)
        with (state / 'web.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.exit(1, f'Another workbench server is already using {data}.\n')
            print(f'Open http://127.0.0.1:{args.port} in your browser. Runs are saved under {data / "runs"}.', flush=True)
            uvicorn.run('backend.app:app', host='127.0.0.1', port=args.port)
        return 0
    if args.command == 'run':
        return subprocess.call([sys.executable, str(resources / 'scripts/run_simulation.py'),
                                str(args.config.resolve()), '--output', str(args.output.resolve())])
    if args.command == 'compare-examples':
        from backend.runtime import interpreter, worker_env
        command = [interpreter('meep'), str(resources/'scripts/benchmark_component_references.py'),
                   '--output', str(args.output.resolve()), '--case', args.case]
        if args.resolutions:
            command.extend(['--resolutions', *map(str, args.resolutions)])
        return subprocess.call(command, env=worker_env('meep','cpu'))
    if args.command == 'setup':
        from .managed_setup import setup_meep
        try:
            result = setup_meep(conda=args.conda, offline=args.offline, data_dir=data, resources=resources)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            parser.exit(1, f'{error}\n')
        print(json.dumps(result, indent=2))
        return 0
    from backend.capabilities import capabilities
    result = capabilities(refresh=True)
    if args.json:
        print(json.dumps({'data_dir': str(data), 'targets': result['targets']}, indent=2))
    else:
        print(f'Local data: {data}')
        for target in result['targets']:
            detail = target.get('device_name', '') if target['available'] else target.get('reason', '')
            status = 'ready' if target['available'] else 'unavailable'
            print(f"{target['solver']} {target['device']}: {status} {detail}")
    return 0
