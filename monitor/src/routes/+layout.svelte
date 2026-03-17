<script lang="ts">
  import '../app.css';
  let { children } = $props();
</script>

<div class="min-h-screen bg-[#0f1117] text-gray-100">
  <header class="border-b border-gray-800 bg-[#161822]">
    <div class="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
      <div class="flex items-center gap-3">
        <div class="flex size-8 items-center justify-center rounded-lg bg-emerald-600 text-sm font-bold">S</div>
        <h1 class="text-lg font-semibold">SimCloud Control Hub</h1>
      </div>
      <nav class="flex gap-6 text-sm">
        <a href="/" class="text-gray-400 transition hover:text-white">Dashboard</a>
        <a href="/containers" class="text-gray-400 transition hover:text-white">Containers</a>
        <a href="/simulation" class="text-gray-400 transition hover:text-white">Simulation</a>
        <a href="/jobs" class="text-gray-400 transition hover:text-white">Jobs</a>
      </nav>
    </div>
  </header>

  <main class="mx-auto max-w-7xl px-6 py-8">
    {@render children()}
  </main>
</div>
