import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { sshExec } from '$lib/server/ssh';

export const GET: RequestHandler = async ({ url }) => {
  const container = url.searchParams.get('container');
  const lines = parseInt(url.searchParams.get('lines') ?? '100');

  if (!container || !/^[a-zA-Z0-9_-]+$/.test(container)) {
    return json({ error: 'valid container name required' }, { status: 400 });
  }

  try {
    const logs = await sshExec(`docker logs ${container} --tail ${Math.min(lines, 500)} 2>&1`);
    return json({ container, logs, lines });
  } catch (err) {
    return json({ error: (err as Error).message }, { status: 500 });
  }
};
