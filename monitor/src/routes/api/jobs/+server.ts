import { json } from '@sveltejs/kit';
import { fetchOrchestrator } from '$lib/server/orchestrator';

export async function GET() {
  try {
    const jobs = await fetchOrchestrator('/api/jobs');
    return json(jobs);
  } catch (err) {
    return json({ error: (err as Error).message, items: [] }, { status: 500 });
  }
}
