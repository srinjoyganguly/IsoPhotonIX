"""Atomic local checkpoints for measured Meep spectra and field samples."""
import errno
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np


class MeasurementCheckpoints:
    def __init__(self, directory, worker):
        import meep
        import scipy

        self.root=Path(directory).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.provenance={
            'protocol':1, 'worker_sha256':hashlib.sha256(Path(worker).read_bytes()).hexdigest(),
            'meep':meep.__version__, 'numpy':np.__version__, 'scipy':scipy.__version__,
            'python':sys.version.split()[0], 'architecture':platform.machine(),
        }

    def acquire(self, label, settings, calculate):
        if not label or any(not (letter.isalnum() or letter in '._-') for letter in label):
            raise ValueError('Invalid checkpoint measurement name.')
        spec={'label':label,'settings':settings,'provenance':self.provenance}
        signature=self.json_hash(spec)
        dest=self.root/(label+'-'+signature[:20])
        if not dest.exists():
            report,arrays=calculate()
            with tempfile.TemporaryDirectory(dir=self.root) as tmp:
                stage=Path(tmp)/'sample'
                stage.mkdir()
                np.savez_compressed(stage/'arrays.npz',**arrays)
                checksum=self.file_hash(stage/'arrays.npz')
                receipt={'signature':signature,'specification':spec,'arrays_sha256':checksum,
                         'report':report,'report_sha256':self.json_hash(report)}
                (stage/'receipt.json').write_text(json.dumps(receipt,allow_nan=False)+'\n')
                try:
                    stage.rename(dest)
                except OSError as exc:
                    # Another process may finish the same immutable measurement first.
                    if exc.errno not in (errno.ENOTEMPTY,errno.EEXIST) or not dest.is_dir():
                        raise
        receipt=json.loads((dest/'receipt.json').read_text())
        if (receipt['signature']!=signature or self.json_hash(receipt['specification'])!=signature
                or self.json_hash(receipt['report'])!=receipt['report_sha256']
                or self.file_hash(dest/'arrays.npz')!=receipt['arrays_sha256']):
            raise ValueError(f'Checkpoint failed integrity verification: {dest}')
        with np.load(dest/'arrays.npz',allow_pickle=False) as record:
            arrays={key:record[key] for key in record.files}
        return receipt['report'],arrays

    @staticmethod
    def file_hash(path):
        with path.open('rb') as data:
            return hashlib.file_digest(data,'sha256').hexdigest()

    @staticmethod
    def json_hash(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def read_locked_plan(path, lock_name, pending_name):
    """Recover initialization only when there are no data from a previous plan.

    The caller holds the native study lock. A leftover pending manifest was
    never committed, so it can be replaced before starting the first stage.
    """
    if path.is_file():
        return path.read_text().strip()
    permitted={lock_name,pending_name}
    if any(item.name not in permitted or not item.is_file() for item in path.parent.iterdir()):
        raise ValueError('Cannot resume a study containing data without its committed manifest.')
    return None
