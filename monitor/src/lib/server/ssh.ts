import { exec } from 'node:child_process';
import { Client } from 'ssh2';
import { env } from '$env/dynamic/private';

const SSH_HOST = env.SSH_HOST || '127.0.0.1';
const SSH_USER = env.SSH_USER || 'ubuntu';
const SSH_PASSWORD = env.SSH_PASSWORD || '';

const isLocal = SSH_HOST === '127.0.0.1' || SSH_HOST === 'localhost';

function localExec(command: string): Promise<string> {
  return new Promise((resolve, reject) => {
    exec(command, { timeout: 15000, maxBuffer: 1024 * 1024 }, (err, stdout, stderr) => {
      if (err) return reject(err);
      resolve(stdout.trim());
    });
  });
}

function remoteExec(command: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const conn = new Client();
    let output = '';

    conn
      .on('ready', () => {
        conn.exec(command, (err, stream) => {
          if (err) {
            conn.end();
            return reject(err);
          }
          stream
            .on('close', () => {
              conn.end();
              resolve(output.trim());
            })
            .on('data', (data: Buffer) => {
              output += data.toString();
            })
            .stderr.on('data', (data: Buffer) => {
              output += data.toString();
            });
        });
      })
      .on('error', (err) => reject(err))
      .connect({
        host: SSH_HOST,
        port: 22,
        username: SSH_USER,
        password: SSH_PASSWORD,
        readyTimeout: 10000,
      });
  });
}

export const sshExec = isLocal ? localExec : remoteExec;
