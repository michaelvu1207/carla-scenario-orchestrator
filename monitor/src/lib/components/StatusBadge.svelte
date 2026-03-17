<script lang="ts">
  let { status }: { status: string } = $props();
  const isUp = $derived(status === 'running' || status === 'healthy' || status === 'true');
  const isWarn = $derived(status === 'paused');
</script>

<span class="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium
  {isUp ? 'bg-emerald-500/10 text-emerald-400' : isWarn ? 'bg-yellow-500/10 text-yellow-400' : 'bg-red-500/10 text-red-400'}">
  <span class="size-1.5 rounded-full {isUp ? 'bg-emerald-400' : isWarn ? 'bg-yellow-400' : 'bg-red-400'}"></span>
  {status}
</span>
