import { useState } from 'react';

export function CodeExport({ value, kind = 'study', disabled = false }: { value: unknown; kind?: 'study' | 'geometry'; disabled?: boolean }) {
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  async function download() {
    if (busy || disabled) return;
    const snapshot = JSON.stringify(value);
    setBusy(true);setError('');
    try {
      const response = await fetch(`/api/reproduce/${kind}`,{method:'POST',headers:{'Content-Type':'application/json'},body:snapshot});
      if (!response.ok) { const body=await response.json();throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail)); }
      const url=URL.createObjectURL(await response.blob());
      const link=document.createElement('a');link.href=url;link.download=`photonics-${kind}-python-notebook.zip`;link.click();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  }
  return <span className="code-export"><button className="button secondary compact" onClick={download} disabled={disabled||busy}>{busy?'Preparing code…':kind==='geometry'?'Export geometry Python + notebook':'Export Python + notebook'}</button>{error&&<span className="code-export-error" role="alert">Code export unavailable: {error}</span>}</span>;
}
