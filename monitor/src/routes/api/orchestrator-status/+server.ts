import { json } from '@sveltejs/kit';
import { fetchOrchestrator } from '$lib/server/orchestrator';

export async function GET() {
  try {
    const [health, simulation, capacity, carlaStatus, jobs] = await Promise.all([
      fetchOrchestrator<{
        status: string;
        simulation_running: boolean;
        connections: number;
        total_slots: number;
        busy_slots: number;
        queued_jobs: number;
      }>('/api/health').catch(() => null),
      fetchOrchestrator<{
        status: string;
        message: string;
        is_running: boolean;
        is_paused: boolean;
        job_id: string | null;
        queue_position: number;
        run_id: string | null;
        camera_recordings: unknown;
      }>('/api/simulation/status').catch(() => null),
      fetchOrchestrator<{ total_slots: number; busy_slots: number; free_slots: number }>('/api/capacity').catch(() => null),
      fetchOrchestrator<{ current_map: string | null }>('/api/carla/status').catch(() => null),
      fetchOrchestrator<{ items: Array<{ job_id: string; state: string; run_id: string | null }> }>('/api/jobs').catch(() => null),
    ]);

    const latestJob = jobs && jobs.items.length > 0 ? jobs.items[jobs.items.length - 1] : null;

    return json({
      healthy: health?.status === 'healthy',
      health,
      simulation,
      capacity,
      latestJob,
      mapName: carlaStatus?.current_map ?? null,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    return json({ error: (err as Error).message, healthy: false }, { status: 500 });
  }
}
