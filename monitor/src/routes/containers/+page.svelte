<script lang="ts">
  import { onMount } from 'svelte';
  import StatusBadge from '$lib/components/StatusBadge.svelte';

  type Container = { name: string; status: string; state: string; image: string; ports: string };

  let containers: Container[] = $state([]);
  let loading = $state(true);
  let restarting: Record<string, boolean> = $state({});
  let logModal: { container: string; logs: string } | null = $state(null);
  let logLoading = $state(false);

  async function refresh() {
    const res = await fetch('/api/server-status');
    const data = await res.json();
    containers = data.containers ?? [];
    loading = false;
  }

  async function restartContainer(name: string) {
    restarting[name] = true;
    try {
      await fetch('/api/docker/restart', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ container: name }),
      });
      await refresh();
    } finally {
      restarting[name] = false;
    }
  }

  async function viewLogs(name: string) {
    logLoading = true;
    logModal = { container: name, logs: '' };
    try {
      const res = await fetch(`/api/logs?container=${name}&lines=150`);
      const data = await res.json();
      logModal = { container: name, logs: data.logs ?? data.error ?? 'No logs' };
    } catch {
      logModal = { container: name, logs: 'Failed to fetch logs' };
    }
    logLoading = false;
  }

  onMount(() => {
    refresh();
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  });
</script>

<svelte:head>
  <title>Containers — SimCloud Control Hub</title>
</svelte:head>

<h2 class="mb-6 text-2xl font-semibold">Docker Containers</h2>

{#if loading}
  <div class="flex items-center justify-center py-20 text-gray-500">
    <div class="size-6 animate-spin rounded-full border-2 border-gray-600 border-t-emerald-500"></div>
    <span class="ml-3">Loading...</span>
  </div>
{:else}
  <div class="overflow-hidden rounded-xl border border-gray-800 bg-[#161822]">
    <table class="w-full text-left text-sm">
      <thead class="border-b border-gray-800 text-xs uppercase text-gray-500">
        <tr>
          <th class="px-5 py-3">Name</th>
          <th class="px-5 py-3">Image</th>
          <th class="px-5 py-3">Status</th>
          <th class="px-5 py-3">Ports</th>
          <th class="px-5 py-3 text-right">Actions</th>
        </tr>
      </thead>
      <tbody>
        {#each containers as c}
          <tr class="border-b border-gray-800/50">
            <td class="px-5 py-3 font-mono font-medium">{c.name}</td>
            <td class="px-5 py-3 text-gray-400">{c.image}</td>
            <td class="px-5 py-3">
              <StatusBadge status={c.state} />
              <span class="ml-2 text-xs text-gray-500">{c.status}</span>
            </td>
            <td class="px-5 py-3 font-mono text-xs text-gray-400">{c.ports || '—'}</td>
            <td class="px-5 py-3 text-right">
              <button
                onclick={() => restartContainer(c.name)}
                disabled={restarting[c.name]}
                class="mr-2 rounded-lg bg-yellow-600/20 px-3 py-1 text-xs text-yellow-400 transition hover:bg-yellow-600/30 disabled:opacity-50"
              >
                {restarting[c.name] ? 'Restarting...' : 'Restart'}
              </button>
              <button
                onclick={() => viewLogs(c.name)}
                class="rounded-lg bg-gray-700/50 px-3 py-1 text-xs text-gray-300 transition hover:bg-gray-700"
              >
                Logs
              </button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{/if}

<!-- Log Modal -->
{#if logModal}
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" role="dialog">
    <div class="flex max-h-[80vh] w-full max-w-4xl flex-col rounded-xl border border-gray-800 bg-[#161822]">
      <div class="flex items-center justify-between border-b border-gray-800 px-5 py-3">
        <h3 class="font-medium">Logs: <span class="font-mono text-emerald-400">{logModal.container}</span></h3>
        <button onclick={() => (logModal = null)} class="text-gray-400 hover:text-white">✕</button>
      </div>
      <div class="flex-1 overflow-auto p-4">
        {#if logLoading}
          <p class="text-gray-500">Loading logs...</p>
        {:else}
          <pre class="whitespace-pre-wrap font-mono text-xs leading-relaxed text-gray-300">{logModal.logs}</pre>
        {/if}
      </div>
    </div>
  </div>
{/if}
