<script lang="ts">
  import { onMount } from 'svelte';
  import StatusBadge from '$lib/components/StatusBadge.svelte';

  type Job = {
    job_id: string;
    state: string;
    created_at: string;
    updated_at: string;
    queue_position: number;
    run_id: string | null;
    error: string | null;
    request: {
      map_name: string;
      source_run_id?: string | null;
      actors: Array<{ id: string }>;
    };
    gpu: {
      device_id: string;
      carla_rpc_port: number;
    } | null;
  };

  let jobs: Job[] = $state([]);
  let loading = $state(true);
  let loadError: string | null = $state(null);

  async function refresh() {
    try {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      jobs = data.items ?? [];
      loadError = data.error ?? null;
    } catch (err) {
      loadError = (err as Error).message;
      jobs = [];
    }
    loading = false;
  }

  onMount(() => {
    refresh();
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  });
</script>

<svelte:head>
  <title>Jobs — SimCloud Control Hub</title>
</svelte:head>

<div class="mb-6 flex items-center justify-between">
  <h2 class="text-2xl font-semibold">Job History</h2>
  <button onclick={refresh} class="rounded-lg bg-gray-800 px-3 py-1.5 text-sm text-gray-300 transition hover:bg-gray-700">
    Refresh
  </button>
</div>

{#if loading}
  <div class="flex items-center justify-center py-20 text-gray-500">
    <div class="size-6 animate-spin rounded-full border-2 border-gray-600 border-t-emerald-500"></div>
    <span class="ml-3">Loading jobs...</span>
  </div>
{:else if loadError}
  <div class="rounded-xl border border-red-900/50 bg-red-950/30 px-5 py-4 text-sm text-red-300">
    Failed to load jobs: {loadError}
  </div>
{:else if jobs.length === 0}
  <div class="rounded-xl border border-gray-800 bg-[#161822] px-5 py-10 text-center text-sm text-gray-500">
    No orchestrator jobs yet.
  </div>
{:else}
  <div class="overflow-hidden rounded-xl border border-gray-800 bg-[#161822]">
    <table class="w-full text-left text-sm">
      <thead class="border-b border-gray-800 text-xs uppercase text-gray-500">
        <tr>
          <th class="px-5 py-3">Job</th>
          <th class="px-5 py-3">State</th>
          <th class="px-5 py-3">Map</th>
          <th class="px-5 py-3">Actors</th>
          <th class="px-5 py-3">GPU</th>
          <th class="px-5 py-3">Queue</th>
          <th class="px-5 py-3">Run</th>
          <th class="px-5 py-3">Updated</th>
        </tr>
      </thead>
      <tbody>
        {#each [...jobs].reverse() as job}
          <tr class="border-b border-gray-800/50 align-top">
            <td class="px-5 py-3">
              <div class="font-mono text-gray-200">{job.job_id}</div>
              {#if job.request.source_run_id}
                <div class="mt-1 text-xs text-gray-500">source: {job.request.source_run_id}</div>
              {/if}
              {#if job.error}
                <div class="mt-2 max-w-sm text-xs text-red-300">{job.error}</div>
              {/if}
            </td>
            <td class="px-5 py-3">
              <StatusBadge status={job.state} />
            </td>
            <td class="px-5 py-3 font-mono text-gray-300">{job.request.map_name}</td>
            <td class="px-5 py-3 text-gray-400">{job.request.actors.length}</td>
            <td class="px-5 py-3 text-gray-400">
              {#if job.gpu}
                GPU {job.gpu.device_id}<br />
                <span class="font-mono text-xs">:{job.gpu.carla_rpc_port}</span>
              {:else}
                —
              {/if}
            </td>
            <td class="px-5 py-3 text-gray-400">{job.queue_position || '—'}</td>
            <td class="px-5 py-3 font-mono text-xs text-gray-400">{job.run_id ?? '—'}</td>
            <td class="px-5 py-3 text-gray-400">{new Date(job.updated_at).toLocaleString()}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{/if}
