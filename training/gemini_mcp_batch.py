"""通过插件的标准 MCP stdio 工具持续抽取，逐请求检查点；不直接调用 agy。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

from training.gemini_semantic_extract import DEFAULT_OUTPUT, ROOT, atomic_json, ingest, prepare, report

PLUGIN = Path('C:/Users/Administrator/.codex/plugins/cache/coordinate-agents/coordinate-agents/2.4.0')
AGENT_ROOT = ROOT/'work/gemini_semantic_probe_2026-10-05'


class McpClient:
    def __init__(self, node: str, output: Path):
        self.lock = threading.Lock()
        self.pending = {}
        self.counter = 0
        self.stderr = (output/'mcp_stderr.log').open('a', encoding='utf-8')
        self.child = subprocess.Popen([node, str(PLUGIN/'mcp/server.mjs'), '--stdio'],
                                      cwd=PLUGIN, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self.stderr, text=True, encoding='utf-8',
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        threading.Thread(target=self.receive, daemon=True).start()
        self.rpc('initialize', {'protocolVersion': '2025-06-18',
                               'capabilities': {}, 'clientInfo': {'name': 'aecopd-extractor', 'version': '1'}})
        with self.lock:
            self.child.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'})+'\n')
            self.child.stdin.flush()

    def receive(self):
        for line in self.child.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            with self.lock:
                target = self.pending.get(message.get('id'))
            if target is not None:
                target.put(message)

    def rpc(self, method: str, params: dict):
        with self.lock:
            self.counter += 1
            identity = self.counter
            target = self.pending[identity] = queue.Queue()
            self.child.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': identity,
                                               'method': method, 'params': params})+'\n')
            self.child.stdin.flush()
        try:
            response = target.get(timeout=45)
        finally:
            with self.lock:
                self.pending.pop(identity, None)
        if 'error' in response:
            raise RuntimeError('MCP JSON-RPC error: '+str(response['error'].get('code')))
        return response['result']

    def tool(self, name: str, **arguments):
        result = self.rpc('tools/call', {'name': name, 'arguments': arguments})
        data = result.get('structuredContent') or json.loads(result['content'][0]['text'])
        if result.get('isError') or not data.get('ok'):
            raise RuntimeError('MCP tool failure: '+name+' '+str(data.get('error', {}).get('code')))
        return data

    def close(self):
        self.child.stdin.close()
        try:
            self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.child.terminate()
        self.stderr.close()


class Batch:
    def __init__(self, output: Path, node: str):
        self.output = output
        self.client = McpClient(node, output)
        self.lock = threading.Lock()
        self.state_path = output/'scheduler.json'
        self.state = json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.exists() else {
            'active': {}, 'stop_reason': None, 'started_at': time.time()}
        if self.state.get('stop_reason'):
            raise ValueError('Previous stop preserved; inspect before explicit recovery')
        self.manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        assigned = {v['id'] for v in self.state['active'].values() if v}
        self.jobs = [r['id'] for r in self.manifest['requests']
                     if not (output/'results'/f"{r['id']}.json").exists() and r['id'] not in assigned]
        self.consecutive_parse_failures = 0

    def persist(self):
        self.state['updated_at'] = time.time()
        self.state['remaining_prepared_requests'] = len(self.jobs)
        atomic_json(self.state_path, self.state)

    def next_job(self):
        # 调用方持有锁；首批用完后按同一固定排序扩充，不复跑已保存请求。
        if self.state.get('stop_reason') or (self.output/'STOP').exists():
            return None
        if not self.jobs:
            old_time = self.manifest['created_at']
            limit = self.manifest['prepared_notes']+100
            inventory = prepare(self.output, limit)
            self.manifest = json.loads((self.output/'manifest.json').read_text(encoding='utf-8'))
            self.manifest['created_at'] = old_time
            atomic_json(self.output/'manifest.json', self.manifest)
            assigned = {v['id'] for v in self.state['active'].values() if v}
            self.jobs = [r['id'] for r in self.manifest['requests']
                         if not (self.output/'results'/f"{r['id']}.json").exists() and r['id'] not in assigned]
            if not self.jobs:
                self.state['stop_reason'] = 'All eligible notes processed'
                self.persist()
                return None
        return self.jobs.pop(0)

    def worker(self, worker: int):
        key = str(worker)
        with self.lock:
            active = self.state['active'].get(key)
        try:
            while True:
                if active is None:
                    with self.lock:
                        identity = self.next_job()
                    if identity is None:
                        return
                    request = json.loads((self.output/'requests'/f'{identity}.json').read_text(encoding='utf-8'))
                    session = self.client.tool('coordinate_agents_session_open', root=str(AGENT_ROOT),
                                               agent=f'gemini-extractor-{worker}',
                                               initialPrompt=request['prompt'], language='en')['session']
                    active = {'id': identity, 'sessionId': session['id'], 'start': time.time()}
                    with self.lock:
                        self.state['active'][key] = active
                        self.persist()
                data = self.client.tool('coordinate_agents_session_inspect', root=str(AGENT_ROOT),
                                        sessionId=active['sessionId'], maxBytes=131072, maxLines=5000)
                if data['session']['state'] in {'exited', 'failed'}:
                    transport = {'output': data['output']['output'], 'truncated': data['output']['truncated'],
                                 'session_id': active['sessionId'], 'state': data['session']['state'],
                                 'exit_code': data['session'].get('exitCode')}
                    saved = ingest(self.output, active['id'], transport)
                    try:
                        envelope = json.loads(transport['output'])
                    except ValueError:
                        envelope = {}
                    with self.lock:
                        if transport['state'] == 'failed' or envelope.get('status') != 'SUCCESS':
                            self.state['stop_reason'] = 'CLI/Runtime failure; inspect saved response; no automatic retry'
                        # 少量无效原句不终止；连续整请求无法解析才停止，不能视为缺失。
                        self.consecutive_parse_failures = (
                            self.consecutive_parse_failures+1 if saved['status'] != 'complete'
                            and saved['fragments'] == 0 and saved['errors'] else 0)
                        if self.consecutive_parse_failures >= 3:
                            self.state['stop_reason'] = 'Three consecutive empty failed extractions'
                        self.state['active'][key] = None
                        self.persist()
                        report(self.output)
                    print(json.dumps(saved), flush=True)
                    active = None
                    continue
                if time.time()-active['start'] > 240:
                    self.client.tool('coordinate_agents_session_close', root=str(AGENT_ROOT),
                                     sessionId=active['sessionId'], graceful=True, timeoutMs=1500)
                    with self.lock:
                        self.state['stop_reason'] = 'Bounded wait exceeded 240s; inspect before retry'
                        self.persist()
                    return
                time.sleep(12)
        except Exception as error:
            with self.lock:
                # 不记录患者内容或带提示词的异常；只保留异常类型与受控请求ID。
                self.state['stop_reason'] = 'Coordinator failure: '+type(error).__name__
                self.state['failed_worker'] = worker
                self.persist()
            raise

    def run(self):
        try:
            # 用户要求串行：不建立并发抽取线程池，仅完成一个请求后取下一个。
            self.worker(0)
        finally:
            with self.lock:
                if (self.output/'STOP').exists() and not self.state.get('stop_reason'):
                    self.state['stop_reason'] = 'User/coordinator stop marker; results preserved'
                self.persist()
                report(self.output)
            self.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--node', default='node')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        client = McpClient(args.node, args.output)
        try:
            listing = client.rpc('tools/list', {})
            names = {t['name'] for t in listing['tools']}
            assert 'coordinate_agents_session_open' in names
            data = client.tool('coordinate_agents_session_status', root=str(AGENT_ROOT),
                               sessionId='session_c0af558877ae43c7bc5cb7f5a53cb0d9')
            print(json.dumps({'mcp': 'ok', 'known_synthetic_session_state': data['session']['state']}))
        finally:
            client.close()
    else:
        Batch(args.output, args.node).run()


if __name__ == '__main__':
    main()
