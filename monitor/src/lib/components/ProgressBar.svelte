<script lang="ts">
  let { value, max, color = 'emerald' }: { value: number; max: number; color?: string } = $props();
  const pct = $derived(Math.min(100, Math.round((value / Math.max(max, 1)) * 100)));
  const colorMap: Record<string, string> = {
    emerald: 'bg-emerald-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500',
    blue: 'bg-blue-500',
  };
  const barColor = $derived(pct > 90 ? 'bg-red-500' : pct > 70 ? 'bg-yellow-500' : (colorMap[color] ?? 'bg-emerald-500'));
</script>

<div class="flex items-center gap-3">
  <div class="h-2 flex-1 overflow-hidden rounded-full bg-gray-700">
    <div class="{barColor} h-full rounded-full transition-all duration-500" style="width:{pct}%"></div>
  </div>
  <span class="text-xs text-gray-400">{pct}%</span>
</div>
