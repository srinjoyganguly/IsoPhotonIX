"""Record the first-party numerical implementation used by an isolated worker.

Only Python modules in the numerical dependency closure enter this archive.
Application state, runtime settings, logs and dependency installations do not.
The format is the same in a checkout and an extracted standalone study bundle.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

FORMAT = 'photonics-workbench-numerical-source-v1'
FILENAME = 'implementation.zip'
MANIFEST = 'implementation-manifest.json'
FORBIDDEN = {'backend.app', 'backend.jobs', 'backend.runtime', 'backend.reproduce',
             'backend.reproduction_runtime'}
SOURCE_NAME = re.compile(r'source/(?:backend/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*|scripts/gpu_timing)\.py\Z')


def _root(root):
    return Path(root) if root is not None else Path(__file__).resolve().parents[1]


def _module_path(root, name):
    stem = root.joinpath(*name.split('.'))
    plain = stem.with_suffix('.py')
    return plain if plain.is_file() else stem/'__init__.py'


def source_files(root=None):
    """Read the numerical import closure, including imports inside functions."""
    root = _root(root)
    pending = ['backend.worker', 'backend.study3d']
    pending.extend('backend.adapters.'+p.stem for p in (root/'backend/adapters').glob('*.py')
                   if p.name != '__init__.py')
    seen, files = set(), {}
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        if any(name == blocked or name.startswith(blocked+'.') for blocked in FORBIDDEN):
            raise ValueError(f'Numerical implementation depends on application state: {name}.')
        seen.add(name)
        path = _module_path(root, name)
        if not path.is_file():
            raise ValueError(f'Numerical source dependency is missing: {name}.')
        relative = path.relative_to(root).as_posix()
        archive_name = 'source/'+relative
        if not SOURCE_NAME.fullmatch(archive_name):
            raise ValueError('Unexpected numerical source filename.')
        data = path.read_bytes()
        files[archive_name] = data
        # Package initialization is executable source too. Preserve it exactly
        # when present; an absent __init__.py stays an absent namespace package.
        parts = name.split('.')
        package = parts if path.name == '__init__.py' else parts[:-1]
        for depth in range(1, len(parts)):
            parent = '.'.join(parts[:depth])
            if (root.joinpath(*parts[:depth])/'__init__.py').is_file():
                pending.append(parent)
        for node in ast.walk(ast.parse(data, filename=relative)):
            imported = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise ValueError('Numerical source imports outside its package.')
                    base = package[:len(package)-node.level+1]
                    imported = (['.'.join(base+node.module.split('.'))] if node.module else
                                ['.'.join(base+[alias.name]) for alias in node.names])
                elif node.module and (node.module == 'backend' or node.module.startswith('backend.') or node.module=='scripts.gpu_timing'):
                    imported = [node.module]
                    # "from backend import helper" also executes helper when
                    # it is a local module, rather than a package attribute.
                    imported.extend(node.module+'.'+alias.name for alias in node.names
                                    if _module_path(root,node.module+'.'+alias.name).is_file())
                elif node.module=='scripts':
                    imported=['scripts.gpu_timing' for alias in node.names if alias.name=='gpu_timing']
            elif isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names
                            if alias.name == 'backend' or alias.name.startswith('backend.') or alias.name=='scripts.gpu_timing']
            pending.extend(item for item in imported if item not in seen)
    return dict(sorted(files.items()))


def numerical_source_files(root=None):
    """Public export-builder name for the same numerical source closure."""
    return source_files(root)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _archive(files):
    manifest = {'format':FORMAT, 'files':{name:{'sha256':_hash(data),'size_bytes':len(data)}
                                        for name,data in files.items()}}
    entries = {**files, MANIFEST:(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(entries.items()):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info,data)
    return stream.getvalue()


def capture_implementation(output, root=None):
    """Capture before importing a solver; never replace earlier provenance."""
    files = source_files(root)
    data = _archive(files)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/FILENAME).open('xb') as stream:
        stream.write(data)
    return {'format':FORMAT, 'filename':FILENAME, 'sha256':_hash(data),
            'file_count':len(files), 'capture':'before_solver_import',
            'verification':'Numerical source bytes must still match at successful completion.'}


def read_implementation(path, expected_sha256):
    """Read only declared, hash-verified Python source; never extract a ZIP."""
    data = Path(path).read_bytes()
    if not isinstance(expected_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_sha256):
        raise ValueError('A recorded numerical implementation SHA256 is required.')
    if _hash(data) != expected_sha256:
        raise ValueError('Numerical implementation archive checksum differs from its run record.')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or MANIFEST not in names:
            raise ValueError('Implementation archive has duplicate or missing entries.')
        manifest = json.loads(archive.read(MANIFEST))
        records = manifest.get('files')
        if manifest.get('format') != FORMAT or not isinstance(records,dict) or not records:
            raise ValueError('Unknown or empty numerical implementation archive.')
        if set(names) != set(records)|{MANIFEST}:
            raise ValueError('Implementation archive contains undeclared files.')
        files = {}
        for name,record in records.items():
            if not SOURCE_NAME.fullmatch(name):
                raise ValueError('Implementation archive contains a non-source or unsafe path.')
            module = name[len('source/'):].removesuffix('.py').replace('/','.')
            if any(module == blocked or module.startswith(blocked+'.') for blocked in FORBIDDEN):
                raise ValueError('Implementation archive contains application state.')
            value = archive.read(name)
            if not isinstance(record,dict) or record.get('sha256') != _hash(value) or record.get('size_bytes') != len(value):
                raise ValueError('Implementation source differs from its archive manifest.')
            files[name] = value
    if not {'source/backend/worker.py','source/backend/study3d.py'} <= files.keys():
        raise ValueError('Implementation archive is missing numerical entry points.')
    return dict(sorted(files.items()))


def verify_implementation(output, record, root=None):
    """Reject a result if numerical source changed while its worker executed."""
    if record.get('format') != FORMAT or record.get('filename') != FILENAME:
        raise ValueError('Unknown numerical implementation record.')
    saved = read_implementation(Path(output)/FILENAME,record.get('sha256'))
    if source_files(root) != saved:
        raise RuntimeError('Numerical implementation changed during this solve; result provenance cannot be certified. Run again with unchanged source.')
    return record
