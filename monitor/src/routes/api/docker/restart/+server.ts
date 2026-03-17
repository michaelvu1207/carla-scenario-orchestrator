import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { sshExec } from '$lib/server/ssh';

export const POST: RequestHandler = async ({ request }) => {
  const { container } = await request.json();
  if (!container || typeof container !== 'string') {
    return json({ error: 'container name required' }, { status: 400 });
  }

  // Only allow known container names to prevent injection
  if (!/^[a-zA-Z0-9_-]+$/.test(container)) {
    return json({ error: 'invalid container name' }, { status: 400 });
  }

  try {
    await sshExec(`docker restart ${container}`);
    const status = await sshExec(`docker ps --filter name=${container} --format '{{.Status}}'`);
    return json({ ok: true, container, status });
  } catch (err) {
    return json({ error: (err as Error).message }, { status: 500 });
  }
};
