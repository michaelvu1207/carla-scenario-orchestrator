import { env } from '$env/dynamic/private';

const DEFAULT_ORCHESTRATOR_URL = 'http://127.0.0.1:8002';
const BASE = env.ORCHESTRATOR_URL || DEFAULT_ORCHESTRATOR_URL;

export async function fetchOrchestrator<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}
