"""Interrupt one real local grating job after its incident reference, then verify checkpoint reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True)
    p.add_argument('--runs',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--browser-signal',type=Path,help='write the interrupted job ID and wait for a browser to click Resume saved study')
    args=p.parse_args()
    def api(path,data=None):
        req=Request(args.url+path,data=json.dumps(data).encode() if data is not None else None,
                    headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=300) as response:
            return json.load(response)
    def wait(check):
        until=time.monotonic()+3600
        while time.monotonic()<until:
            result=check()
            if result:return result
            time.sleep(1)
        raise TimeoutError('Timed out waiting for the native study.')
    def ready():
        try:return api('/api/health')['status']=='ok'
        except OSError:return False
    wait(ready)
    spec={'case':'grating','solver':'meep','device':'cpu','grating_scope':'tidy3d-compact','resolution':40}
    created=api('/api/calibrations/jobs',spec)
    job=created['id']
    folder=args.runs/job
    endpoint='/api/calibrations/jobs/'+job
    def first_checkpoint():
        item=next((folder/'checkpoints').rglob('receipt.json'),None)
        if item:return item
        if api('/api/jobs/'+job)['status'] in ('failed','cancelled','completed'):
            raise RuntimeError('The first native stage could not be interrupted.')
        return None
    first=wait(first_checkpoint)
    saved=first.read_bytes()
    timestamp=first.stat().st_mtime_ns
    manifest=(folder/'replay/manifest.json').read_bytes()
    cancelled=api('/api/jobs/'+job+'/cancel',{})
    assert cancelled['status']=='cancelled'
    wait(lambda: any(j['id']==job and j.get('can_resume') for j in api('/api/calibrations/jobs')))
    if args.browser_signal:
        args.browser_signal.write_text(json.dumps({'job_id':job})+'\n')
        resumed=wait(lambda: (r if (r:=api('/api/jobs/'+job)).get('attempt')==2 else None))
    else:
        resumed=api(endpoint+'/resume',{})
    assert resumed['id']==job and resumed['attempt']==2
    final=wait(lambda: (r if (r:=api('/api/jobs/'+job))['status'] in ('completed','failed','cancelled') else None))
    if final['status']!='completed':raise RuntimeError(final.get('error'))
    report=api(endpoint+'/report')
    assert report['passed'] and abs(report['result']['center_beam_to_guide']-.582341311)<1e-6
    assert first.read_bytes()==saved and first.stat().st_mtime_ns==timestamp
    assert (folder/'replay/manifest.json').read_bytes()==manifest
    result={'passed':True,'job_id':job,'attempt':final['attempt'],'resume_action':'browser' if args.browser_signal else 'API','actual_device':final['result']['actual_device'],
            'config':final['config'],'source_manifest_sha256':hashlib.sha256(manifest).hexdigest(),
            'reference_receipt_sha256':hashlib.sha256(saved).hexdigest(),'reference_timestamp_preserved':True,
            'coupling_fraction':report['result']['center_beam_to_guide']}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
