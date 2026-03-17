<script lang="ts">
  import { onMount } from 'svelte';
  import Card from '$lib/components/Card.svelte';
  import ProgressBar from '$lib/components/ProgressBar.svelte';
  import StatusBadge from '$lib/components/StatusBadge.svelte';

  type GPU = {
    index: number;
    name: string;
    utilizationGpu: number;
    utilizationMem: number;
    memoryUsed: number;
    memoryTotal: number;
    temperature: number;
    powerDraw: number;
  };

  type GpuProcess = { pid: string; gpuUuid: string; usedMemory: string; processName: string };

  type ServerData = {
    containers: Array<{ name: string; status: string; state: string; image: string; ports: string }>;
    gpus: GPU[];
    gpuProcesses: GpuProcess[];
    memory: { totalMB: number; usedMB: number; availableMB: number };
    uptime: { text: string; load1: number; load5: number; load15: number };
    disk: { size: string; used: string; available: string; usePercent: number };
    timestamp: string;
    error?: string;
  };

  type OrchestratorData = {
    healthy: boolean;
    health: { status: string; simulation_running: boolean; queued_jobs: number } | null;
    capacity: { total_slots: number; busy_slots: number; free_slots: number } | null;
    latestJob: { job_id: string; state: string; run_id: string | null } | null;
    simulation: { status: string; message: string; is_running: boolean; is_paused: boolean; job_id?: string | null } | null;
    mapName: string | null;
    error?: string;
  };

  let server: ServerData | null = $state(null);
  let orch: OrchestratorData | null = $state(null);
  let lastUpdated: string = $state('never');
  let countdown: number = $state(10);
  let loading: boolean = $state(true);

  async function refresh() {
    try {
      const [s, o] = await Promise.all([
        fetch('/api/server-status').then((r) => r.json()),
        fetch('/api/orchestrator-status').then((r) => r.json()),
      ]);
      server = s;
      orch = o;
      lastUpdated = new Date().toLocaleTimeString();
    } catch {
      // keep previous data
    }
    loading = false;
    countdown = 10;
  }

  onMount(() => {
    refresh();
    const dataInterval = setInterval(refresh, 10000);
    const tickInterval = setInterval(() => {
      countdown = Math.max(0, countdown - 1);
    }, 1000);
    return () => {
      clearInterval(dataInterval);
      clearInterval(tickInterval);
    };
  });

  function tempColor(t: number) {
    if (t >= 85) return 'text-red-400';
    if (t >= 70) return 'text-yellow-400';
    return 'text-emerald-400';
  }

  function tempBg(t: number) {
    if (t >= 85) return 'bg-red-500';
    if (t >= 70) return 'bg-yellow-500';
    return 'bg-emerald-500';
  }
</script>

<svelte:head>
  <title>Dashboard — SimCloud Control Hub</title>
</svelte:head>

<div class="mb-6 flex items-center justify-between">
  <h2 class="text-2xl font-semibold">Dashboard</h2>
  <div class="flex items-center gap-4 text-sm text-gray-500">
    <span>Last updated: {lastUpdated}</span>
    <span class="tabular-nums">Refresh in {countdown}s</span>
    <button onclick={refresh} class="rounded-lg bg-gray-800 px-3 py-1.5 text-gray-300 transition hover:bg-gray-700">
      Refresh
    </button>
  </div>
</div>

{#if loading}
  <div class="flex items-center justify-center py-20 text-gray-500">
    <div class="size-6 animate-spin rounded-full border-2 border-gray-600 border-t-emerald-500"></div>
    <span class="ml-3">Connecting to server...</span>
  </div>
{:else}
  <!-- Top row: Server Health, Memory, Disk -->
  <div class="grid grid-cols-1 gap-5 md:grid-cols-3">
    <Card title="Server Health">
      {#if server?.error}
        <StatusBadge status="offline" />
      {:else if server}
        <div class="space-y-3">
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Status</span>
            <StatusBadge status="healthy" />
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Uptime</span>
            <span class="text-sm">{server.uptime.text}</span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Load (1/5/15m)</span>
            <span class="font-mono text-sm">{server.uptime.load1} / {server.uptime.load5} / {server.uptime.load15}</span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Containers</span>
            <span class="text-sm">{server.containers.filter((c) => c.state === 'running').length}/{server.containers.length} running</span>
          </div>
        </div>
      {/if}
    </Card>

    <Card title="System Memory">
      {#if server?.memory}
        <div class="space-y-3">
          <div class="mb-1 flex justify-between text-xs text-gray-400">
            <span>RAM</span>
            <span>{(server.memory.usedMB / 1024).toFixed(1)} / {(server.memory.totalMB / 1024).toFixed(1)} GB</span>
          </div>
          <ProgressBar value={server.memory.usedMB} max={server.memory.totalMB} color="blue" />
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Available</span>
            <span class="text-sm">{(server.memory.availableMB / 1024).toFixed(1)} GB</span>
          </div>
        </div>
      {/if}
    </Card>

    <Card title="Disk Usage">
      {#if server?.disk}
        <div class="space-y-3">
          <div class="mb-1 flex justify-between text-xs text-gray-400">
            <span>Root (/)</span>
            <span>{server.disk.used} / {server.disk.size}</span>
          </div>
          <ProgressBar value={server.disk.usePercent} max={100} />
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Available</span>
            <span class="text-sm">{server.disk.available}</span>
          </div>
        </div>
      {/if}
    </Card>
  </div>

  <!-- GPU Grid: All 8 GPUs -->
  {#if server?.gpus?.length}
    <h3 class="mb-4 mt-8 text-lg font-semibold">GPUs ({server.gpus.length}x {server.gpus[0].name})</h3>
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {#each server.gpus as gpu}
        <div class="rounded-xl border border-gray-800 bg-[#161822] p-4">
          <div class="mb-3 flex items-center justify-between">
            <span class="font-mono text-sm font-semibold text-gray-200">GPU {gpu.index}</span>
            <span class="text-sm font-medium {tempColor(gpu.temperature)}">{gpu.temperature}°C</span>
          </div>

          <!-- Compute utilization -->
          <div class="mb-2">
            <div class="mb-1 flex justify-between text-xs text-gray-500">
              <span>Compute</span>
              <span>{gpu.utilizationGpu}%</span>
            </div>
            <div class="h-1.5 overflow-hidden rounded-full bg-gray-700">
              <div class="h-full rounded-full bg-emerald-500 transition-all duration-500" style="width:{gpu.utilizationGpu}%"></div>
            </div>
          </div>

          <!-- VRAM -->
          <div class="mb-2">
            <div class="mb-1 flex justify-between text-xs text-gray-500">
              <span>VRAM</span>
              <span>{(gpu.memoryUsed / 1024).toFixed(1)}/{(gpu.memoryTotal / 1024).toFixed(0)} GB</span>
            </div>
            <div class="h-1.5 overflow-hidden rounded-full bg-gray-700">
              <div class="h-full rounded-full bg-blue-500 transition-all duration-500" style="width:{(gpu.memoryUsed / gpu.memoryTotal * 100).toFixed(0)}%"></div>
            </div>
          </div>

          <!-- Power -->
          <div class="flex justify-between text-xs text-gray-500">
            <span>Power</span>
            <span>{gpu.powerDraw} W</span>
          </div>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Docker Containers -->
  {#if server?.containers?.length}
    <h3 class="mb-4 mt-8 text-lg font-semibold">Containers</h3>
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {#each server.containers as c}
        <div class="flex items-center justify-between rounded-xl border border-gray-800 bg-[#161822] p-4">
          <div>
            <p class="font-mono text-sm font-medium">{c.name}</p>
            <p class="mt-1 text-xs text-gray-500">{c.status}</p>
          </div>
          <StatusBadge status={c.state} />
        </div>
      {/each}
    </div>
  {/if}

  <!-- Orchestrator & Simulation -->
  <h3 class="mb-4 mt-8 text-lg font-semibold">CARLA Orchestrator</h3>
  <div class="grid grid-cols-1 gap-5 md:grid-cols-2">
    <Card title="API Status">
      {#if orch}
        <div class="space-y-3">
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Health</span>
            <StatusBadge status={orch.healthy ? 'healthy' : 'unhealthy'} />
          </div>
          {#if orch.health}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Sim Running</span>
              <span class="text-sm">{orch.health.simulation_running ? 'Yes' : 'No'}</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Queued Jobs</span>
              <span class="text-sm">{orch.health.queued_jobs}</span>
            </div>
          {/if}
          {#if orch.capacity}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">GPU Slots</span>
              <span class="text-sm">{orch.capacity.busy_slots}/{orch.capacity.total_slots} busy</span>
            </div>
          {/if}
          {#if orch.mapName}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Map</span>
              <span class="rounded bg-gray-800 px-2 py-0.5 font-mono text-sm">{orch.mapName}</span>
            </div>
          {/if}
        </div>
      {/if}
    </Card>

    <Card title="Simulation">
      {#if orch?.simulation}
        <div class="space-y-3">
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">State</span>
            <StatusBadge status={orch.simulation.is_running ? 'running' : orch.simulation.is_paused ? 'paused' : 'stopped'} />
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Message</span>
            <span class="text-sm text-gray-300">{orch.simulation.message}</span>
          </div>
          {#if orch.simulation.job_id}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Active Job</span>
              <span class="font-mono text-sm text-gray-300">{orch.simulation.job_id}</span>
            </div>
          {/if}
          {#if orch.latestJob && !orch.simulation.is_running}
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-400">Latest Job</span>
              <span class="font-mono text-sm text-gray-300">{orch.latestJob.job_id}</span>
            </div>
          {/if}
        </div>
      {:else}
        <p class="text-sm text-gray-500">Unable to reach orchestrator</p>
      {/if}
    </Card>
  </div>

  <!-- GPU Processes -->
  {#if server?.gpuProcesses?.length}
    <h3 class="mb-4 mt-8 text-lg font-semibold">GPU Processes</h3>
    <div class="overflow-hidden rounded-xl border border-gray-800 bg-[#161822]">
      <table class="w-full text-left text-sm">
        <thead class="border-b border-gray-800 text-xs uppercase text-gray-500">
          <tr>
            <th class="px-5 py-3">PID</th>
            <th class="px-5 py-3">Process</th>
            <th class="px-5 py-3">GPU Memory</th>
          </tr>
        </thead>
        <tbody>
          {#each server.gpuProcesses as proc}
            <tr class="border-b border-gray-800/50">
              <td class="px-5 py-2 font-mono text-gray-300">{proc.pid}</td>
              <td class="px-5 py-2 text-gray-400">{proc.processName}</td>
              <td class="px-5 py-2 text-gray-400">{proc.usedMemory}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
{/if}
