"""Execute both programs from a local result export outside the workbench checkout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def replay(product, archive_path, destination):
    folder=destination/product
    folder.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(folder)
    env = {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
           'JUPYTER_RUNTIME_DIR':str(folder/'jupyter'),'IPYTHONDIR':str(folder/'ipython'),
           'MPLCONFIGDIR':str(folder/'mpl'),'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']}
    env.pop('PYTHONPATH',None)
    if product=='python':
        command=[sys.executable,'run.py','--output','script-result']
        result=folder/'script-result'
    else:
        script="""import nbformat
from nbclient import NotebookClient
book=nbformat.read('calibration.ipynb',as_version=4)
nbformat.validate(book)
NotebookClient(book,timeout=6*3600,kernel_name='python3').execute()
nbformat.write(book,'executed.ipynb')
"""
        command=[sys.executable,'-c',script]
        result=folder/'notebook-result'
    with (folder/'execution.log').open('w') as stream:
        subprocess.run(command,cwd=folder,env=env,check=True,timeout=6*3600,stdout=stream,stderr=subprocess.STDOUT)
    return json.loads((result/'reproduction.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--browser-report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    directory=args.output.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={p:pool.submit(replay,p,args.export.resolve(),directory) for p in ('python','notebook')}
        results={name:future.result() for name,future in futures.items()}
    record={'passed':all(r['passed'] for r in results.values()),'verified_at':datetime.now(timezone.utc).isoformat(),
            'export_sha256':hashlib.sha256(args.export.read_bytes()).hexdigest(), **results}
    if args.browser_report:
        record['browser']=json.loads(args.browser_report.read_text())
    (directory/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
