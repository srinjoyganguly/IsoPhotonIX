const readable = (v: unknown): string => typeof v === 'string' ? v : JSON.stringify(v);
function errorText(value: unknown): string {
  if (Array.isArray(value)) return value.map(item => typeof item === 'object' && item !== null && 'msg' in item ? `${(item as {loc?: string[]}).loc?.filter(x => x !== 'body').join('.') || 'Configuration'}: ${String((item as {msg: unknown}).msg)}` : readable(item)).join('; ');
  return readable(value) || 'Request failed';
}
export async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, body === undefined ? undefined : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(errorText(data.detail || data.error || data));
  return data as T;
}
