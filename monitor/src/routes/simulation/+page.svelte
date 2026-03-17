<script lang="ts">
  import { onMount } from 'svelte';
  import Card from '$lib/components/Card.svelte';
  import StatusBadge from '$lib/components/StatusBadge.svelte';

  type OrchestratorData = {
    healthy: boolean;
    health: { status: string; simulation_running: boolean; queued_jobs: number } | null;
    capacity: { total_slots: number; busy_slots: number; free_slots: number } | null;
    latestJob: { job_id: string; state: string; run_id: string | null } | null;
    simulation: {
      status: string;
      message: string;
      is_running: boolean;
      is_paused: boolean;
      camera_recordings: unknown;
      job_id: string | null;
      queue_position: number;
      run_id: string | null;
    } | null;
    mapName: string | null;
  };

  let data: OrchestratorData | null = $state(null);
  let loading = $state(true);

  async function refresh() {
    const res = await fetch('/api/orchestrator-status');
    data = await res.json();
    loading = false;
  }

  onMount(() => {
    refresh();
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  });
</script>

<svelte:head>
  <title>Simulation — SimCloud Control Hub</title>
</svelte:head>

<h2 class="mb-6 text-2xl font-semibold">Simulation Status</h2>

{#if loading}
  <div class="flex items-center justify-center py-20 text-gray-500">
    <div class="size-6 animate-spin rounded-full border-2 border-gray-600 border-t-emerald-500"></div>
    <span class="ml-3">Loading...</span>
  </div>
{:else}
  <div class="grid grid-cols-1 gap-5 md:grid-cols-2">
    <Card title="CARLA Connection">
      <div class="space-y-4">
        <div class="flex items-center justify-between">
          <span class="text-sm text-gray-400">API Health</span>
          <StatusBadge status={data?.healthy ? 'healthy' : 'unhealthy'} />
        </div>
        {#if data?.health}
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Simulation Running</span>
            <StatusBadge status={data.health.simulation_running ? 'running' : 'stopped'} />
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Queued Jobs</span>
            <span class="text-lg font-semibold">{data.health.queued_jobs}</span>
          </div>
          {#if data.capacity}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">GPU Slots</span>
              <span class="text-sm">{data.capacity.busy_slots}/{data.capacity.total_slots} busy</span>
            </div>
          {/if}
        {:else}
          <p class="text-sm text-red-400">Cannot reach orchestrator API</p>
        {/if}
      </div>
    </Card>

    <Card title="Current Simulation">
      {#if data?.simulation}
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">State</span>
            <StatusBadge status={data.simulation.is_running ? 'running' : data.simulation.is_paused ? 'paused' : 'stopped'} />
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Message</span>
            <span class="text-sm">{data.simulation.message}</span>
          </div>
          {#if data.simulation.job_id}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Job ID</span>
              <span class="font-mono text-sm">{data.simulation.job_id}</span>
            </div>
          {/if}
          {#if data.simulation.run_id}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Run ID</span>
              <span class="font-mono text-sm">{data.simulation.run_id}</span>
            </div>
          {/if}
          {#if !data.simulation.job_id && data.latestJob}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Latest Job</span>
              <span class="font-mono text-sm">{data.latestJob.job_id}</span>
            </div>
          {/if}
          {#if data.mapName}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Loaded Map</span>
              <span class="rounded bg-gray-800 px-2 py-0.5 font-mono text-sm">{data.mapName}</span>
            </div>
          {/if}
          {#if data.simulation.camera_recordings}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Camera Recordings</span>
              <span class="text-sm text-emerald-400">Active</span>
            </div>
          {:else}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Camera Recordings</span>
              <span class="text-sm text-gray-500">None</span>
            </div>
          {/if}
        </div>
      {:else}
        <p class="text-sm text-gray-500">No simulation data available</p>
      {/if}
    </Card>
  </div>
{/if}
