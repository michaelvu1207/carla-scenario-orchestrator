import { json } from '@sveltejs/kit';
import { sshExec } from '$lib/server/ssh';

export async function GET() {
  try {
    const [dockerRaw, gpuRaw, memRaw, uptimeRaw, diskRaw, gpuProcessRaw] = await Promise.all([
      sshExec('docker ps -a --format \'{"name":"{{.Names}}","status":"{{.Status}}","image":"{{.Image}}","ports":"{{.Ports}}","state":"{{.State}}"}\''),
      sshExec('nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader'),
      sshExec('free -m | grep Mem'),
      sshExec('uptime'),
      sshExec('df -h / | tail -1'),
      sshExec('nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,name --format=csv,noheader 2>/dev/null || echo ""'),
    ]);

    // Parse docker containers
    const containers = dockerRaw
      .split('\n')
      .filter(Boolean)
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean);

    // Parse all GPUs (one per line)
    const gpus = gpuRaw
      .split('\n')
      .filter(Boolean)
      .map((line) => {
        const parts = line.split(',').map((s) => s.trim());
        return {
          index: parseInt(parts[0]) || 0,
          name: parts[1] ?? 'Unknown',
          utilizationGpu: parseInt(parts[2]) || 0,
          utilizationMem: parseInt(parts[3]) || 0,
          memoryUsed: parseInt(parts[4]) || 0,
          memoryTotal: parseInt(parts[5]) || 0,
          temperature: parseInt(parts[6]) || 0,
          powerDraw: parseFloat(parts[7]) || 0,
        };
      });

    // Parse GPU processes
    const gpuProcesses = gpuProcessRaw
      .split('\n')
      .filter(Boolean)
      .map((line) => {
        const parts = line.split(',').map((s) => s.trim());
        return {
          pid: parts[0] ?? '',
          gpuUuid: parts[1] ?? '',
          usedMemory: parts[2] ?? '',
          processName: parts[3] ?? '',
        };
      });

    // Parse memory
    const memParts = memRaw.split(/\s+/);
    const memory = {
      totalMB: parseInt(memParts[1]) || 0,
      usedMB: parseInt(memParts[2]) || 0,
      availableMB: parseInt(memParts[6]) || 0,
    };

    // Parse uptime
    const loadMatch = uptimeRaw.match(/load average:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)/);
    const uptimeMatch = uptimeRaw.match(/up\s+(.+?),\s+\d+\s+user/);
    const uptime = {
      text: uptimeMatch?.[1]?.trim() ?? 'unknown',
      load1: parseFloat(loadMatch?.[1] ?? '0'),
      load5: parseFloat(loadMatch?.[2] ?? '0'),
      load15: parseFloat(loadMatch?.[3] ?? '0'),
    };

    // Parse disk
    const diskParts = diskRaw.split(/\s+/);
    const disk = {
      size: diskParts[1] ?? '0',
      used: diskParts[2] ?? '0',
      available: diskParts[3] ?? '0',
      usePercent: parseInt(diskParts[4]) || 0,
    };

    return json({ containers, gpus, gpuProcesses, memory, uptime, disk, timestamp: new Date().toISOString() });
  } catch (err) {
    return json({ error: (err as Error).message }, { status: 500 });
  }
}
