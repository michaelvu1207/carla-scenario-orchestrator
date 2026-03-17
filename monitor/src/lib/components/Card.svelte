<script lang="ts">
  import type { Snippet } from 'svelte';
  let { title, children }: { title: string; children: Snippet } = $props();
</script>

<div class="rounded-xl border border-gray-800 bg-[#161822] p-5">
  <h3 class="mb-4 text-sm font-medium text-gray-400">{title}</h3>
  {@render children()}
</div>
