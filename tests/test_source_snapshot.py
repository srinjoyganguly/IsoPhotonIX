"""Source provenance tests; the worker fixture is not electromagnetic evidence."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from backend.source_snapshot import (FORMAT, capture_implementation, read_implementation,
                                     source_files, verify_implementation)

ROOT = Path(__file__).resolve().parents[1]


def put(root,name,text):
    path = root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)


def fixture(root):
    put(root,'backend/__init__.py','"""Fixture package."""\n')
    put(root,'backend/worker.py','from .study3d import Config\n')
    put(root,'backend/study3d.py','class Config: pass\n')
    put(root,'backend/adapters/example.py','def solve():\n    from ..helper import VALUE\n')
    put(root,'backend/helper.py','VALUE=1\n')
    put(root,'backend/app.py','raise RuntimeError("must never import or archive app")\n')
    put(root,'.local/runtime.json','{"secret":"not source"}\n')


class NumericalSourceSnapshotTests(unittest.TestCase):
    def test_lazy_import_closure_preserves_package_and_excludes_app_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root)
            files = source_files(root)
            self.assertIn('source/backend/helper.py',files)
            self.assertEqual(files['source/backend/__init__.py'],(root/'backend/__init__.py').read_bytes())
            self.assertNotIn('source/backend/adapters/__init__.py',files)
            self.assertFalse(any('app.py' in name or 'runtime.json' in name for name in files))
            put(root,'backend/helper.py','from . import app\n')
            with self.assertRaisesRegex(ValueError,'application state'):
                source_files(root)

    def test_deterministic_archive_detects_source_changes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'project';fixture(root)
            first,second=Path(temporary)/'first',Path(temporary)/'second'
            record=capture_implementation(first,root)
            other=capture_implementation(second,root)
            self.assertEqual(record['format'],FORMAT)
            self.assertEqual(record['sha256'],other['sha256'])
            self.assertEqual(read_implementation(first/'implementation.zip',record['sha256']),source_files(root))
            verify_implementation(first,record,root)
            with self.assertRaises(FileExistsError):capture_implementation(first,root)
            put(root,'backend/helper.py','VALUE=2\n')
            with self.assertRaisesRegex(RuntimeError,'changed during this solve'):
                verify_implementation(first,record,root)

    def test_corrupt_archive_or_undeclared_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'project';fixture(root)
            output=Path(temporary)/'output';record=capture_implementation(output,root)
            path=output/'implementation.zip';original=path.read_bytes()
            path.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                read_implementation(path,record['sha256'])
            stream=io.BytesIO(original)
            with zipfile.ZipFile(stream,'a') as archive:
                archive.writestr('../private.txt','not numerical source')
            changed=stream.getvalue();path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError,'undeclared'):
                read_implementation(path,hashlib.sha256(changed).hexdigest())

    def test_real_numerical_closure_recaptures_inside_standalone_source(self):
        files=source_files()
        self.assertIn('source/backend/source_snapshot.py',files)
        self.assertIn('source/backend/sparameters.py',files)
        self.assertIn('source/backend/ports3d.py',files)
        self.assertNotIn('source/backend/runtime.py',files)
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            for name,data in files.items():
                path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            spec=importlib.util.spec_from_file_location('isolated_snapshot',base/'source/backend/source_snapshot.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertEqual(module.source_files(),files)
            record=module.capture_implementation(base/'results')
            module.verify_implementation(base/'results',record)

    def test_worker_records_source_and_fails_if_adapter_changes_during_execution(self):
        # A small deterministic array fixture exercises worker provenance only.
        adapter='''import numpy as np
from pathlib import Path
def run(config,output):
    np.savez(output/'arrays.npz',field=np.ones((2,2)),epsilon=np.ones((2,2)),x_um=np.arange(2.),y_um=np.arange(2.))
    if config.get('mutate_source'):
        path=Path(__file__);path.write_text(path.read_text()+'\\n# changed during fixture execution\\n')
    return {'actual_device':'CPU','versions':{'fixture':'not a solver'}}
'''
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'source'
            for name in ('worker.py','source_snapshot.py','__init__.py'):
                put(root,'backend/'+name,(ROOT/'backend'/name).read_text())
            put(root,'backend/study3d.py','"""Fixture schema root."""\n')
            for name in ('volume','spectra','sparameters'):
                put(root,f'backend/{name}.py','"""Unused fixture worker branch."""\n')
            put(root,'backend/adapters/meep_adapter.py',adapter)
            for mutate in (False,True):
                output=Path(temporary)/('changed' if mutate else 'stable');output.mkdir()
                config=output/'config.json';config.write_text(json.dumps({'solver':'meep','device':'cpu','mutate_source':mutate}))
                result=subprocess.run([sys.executable,'-m','backend.worker',str(config),str(output)],
                    cwd=root,env={**os.environ,'PYTHONPATH':str(root)},capture_output=True,text=True,timeout=30)
                if mutate:
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('changed during this solve',result.stderr)
                    self.assertFalse((output/'metadata.json').exists())
                else:
                    self.assertEqual(result.returncode,0,result.stderr)
                    record=json.loads((output/'metadata.json').read_text())['implementation_snapshot']
                    self.assertEqual(record['capture'],'before_solver_import')
                    self.assertIn('source/backend/adapters/meep_adapter.py',read_implementation(output/'implementation.zip',record['sha256']))


if __name__ == '__main__':
    unittest.main()
