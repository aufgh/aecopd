"""本地全量宽主题抽取→GPU双编码；不接续任何预测训练。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from training.bonsai_broad_fragments import atomic_json
from training.bonsai_clinical_evidence_pilot import local_api


def run(root: Path, output: Path, *, encode_only: bool, encoding_name: str, extraction_name: str,
        selection_file: Path | None = None, allow_full: bool = False) -> None:
    if not encode_only and selection_file is None and not allow_full:
        raise ValueError('Full run was cancelled: supply --selection-file for a bounded batch; --allow-full requires an explicit new full-run request')
    root, output = root.resolve(), output.resolve()
    snapshot = 'patient_data/mimiciv_copd/snapshot_2026-09-27'
    relative = output.resolve().relative_to(root.resolve()).as_posix()
    log = output/'pipeline.log'
    server = None
    output.mkdir(parents=True, exist_ok=True)
    began = time.time()
    def status(phase, **extra):
        atomic_json(output/'pipeline_state.json', {'phase': phase, 'started_unix': began,
            'updated_unix': time.time(), 'training_started': False, 'cpu_encoding_allowed': False,
            'encoding_directory': encoding_name, **extra})
    def command(arguments):
        with log.open('a', encoding='utf-8') as handle:
            handle.write('\nCOMMAND '+json.dumps(arguments)+'\n'); handle.flush()
            subprocess.run(arguments, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    def docker(arguments, gpu=False):
        # 数据和权重留在本机，容器禁止网络、模型只读、代码只读。
        args = ['docker', 'run', '--rm', '--network', 'none']
        if gpu:
            args += ['--gpus', 'all']
        args += ['-v', str(root/'training')+':/workspace/training:ro',
                 '-v', str(root/'patient_data/mimiciv_copd/snapshot_2026-09-27')+':/snapshot:ro',
                 '-v', str(output)+':/output', '-v', str(root/'training/model_cache')+':/model-cache:ro',
                 '-w', '/workspace', '-e', 'PYTHONPATH=/workspace',
                 '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
                 'aecopd-embedding:v1', 'python']+arguments
        command(args)
    try:
        if not encode_only:
            status('starting_local_bonsai')
            # 不接管、终止别人已占用的8080服务。
            try:
                local_api('http://127.0.0.1:8080', '/health')
            except OSError:
                pass
            else:
                raise RuntimeError('Port 8080 already has a service; refusing to take ownership')
            executable = Path('D:/ywj/Bonsai-demo/bin/cuda/llama-server.exe')
            args = [str(executable), '-m', 'D:/ywj/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf',
                    '--host', '127.0.0.1', '--port', '8080', '-ngl', '99', '-fa', 'on',
                    '-c', '32768', '-np', '4', '-cb', '--jinja', '--reasoning', 'auto',
                    '--reasoning-budget', '512', '--reasoning-format', 'deepseek', '--no-webui',
                    '--cors-origins', 'localhost', '--no-cors-credentials', '-lv', '1']
            with (output/'full_server.log').open('a', encoding='utf-8') as handle:
                server = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT,
                                          creationflags=subprocess.CREATE_NO_WINDOW)
            atomic_json(output/'owned_server.json', {'pid': server.pid, 'executable': str(executable)})
            for _ in range(90):
                if server.poll() is not None:
                    raise RuntimeError('Bonsai exited during startup; see full_server.log')
                try:
                    if local_api('http://127.0.0.1:8080', '/health')['status'] == 'ok':
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('Bonsai startup timeout')
            if selection_file is None:
                status('extracting_all_eligible_notes', owned_server_pid=server.pid, workers=4)
                command([sys.executable, '-m', 'training.bonsai_broad_fragments', 'extract',
                         '--snapshot', snapshot, '--output', relative+'/'+extraction_name])
            else:
                selection = json.loads(selection_file.read_text(encoding='utf-8'))
                status('extracting_fixed_batch', owned_server_pid=server.pid, workers=4,
                       requested_notes=selection['selected_notes'], member_sha256=selection['member_sha256'])
                command([sys.executable, '-m', 'training.broad_fragment_batch', 'extract',
                         '--snapshot', snapshot, '--output', relative+'/'+extraction_name,
                         '--selection-file', str(selection_file.resolve())])
            server.terminate(); server.wait(timeout=30); server = None
        status('building_shared_manifest')
        target = output/encoding_name
        if not (target/'manifest_summary.json').exists():
            docker(['-m', 'training.bonsai_broad_fragments', 'manifest', '--snapshot', '/snapshot',
                    '--output', '/output/'+extraction_name, '--manifest-dir', '/output/'+encoding_name])
        status('encoding_bert_gpu')
        docker(['-m', 'training.encode_rule_text_bert', '--manifest', '/output/'+encoding_name+'/shared_text.csv.gz',
                '--output-dir', '/output/'+encoding_name+'/bert', '--batch-size', '64'], gpu=True)
        status('encoding_qwen_gpu')
        docker(['-m', 'training.generate_text_embeddings', '--chunks', '/output/'+encoding_name+'/shared_text.csv.gz',
                '--output', '/output/'+encoding_name+'/qwen/qwen3_embedding.float16.npy',
                '--state', '/output/'+encoding_name+'/qwen/state.json',
                '--token-budget', '4096', '--max-batch-size', '16'], gpu=True)
        status('validating_vectors')
        docker(['-m', 'training.validate_broad_fragment_vectors', '--directory', '/output/'+encoding_name])
        status('encoding_complete', validation=json.loads((target/'vector_validation.json').read_text(encoding='utf-8')))
    except Exception as error:
        status('failed', error=str(error))
        raise
    finally:
        # 只终止这个Popen对象创建的服务，不按进程名清理其他任务。
        if server is not None and server.poll() is None:
            server.terminate(); server.wait(timeout=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encode-only', action='store_true')
    parser.add_argument('--encoding-name', default='full_encoding')
    parser.add_argument('--extraction-name', default='extraction_sentence_context')
    parser.add_argument('--selection-file', type=Path)
    parser.add_argument('--allow-full', action='store_true', help='Explicit opt-in only after a new user request for a full run')
    args = parser.parse_args()
    if '/' in args.encoding_name or '\\' in args.encoding_name or args.encoding_name in ('', '.', '..'):
        parser.error('encoding-name must be a simple directory name')
    if '/' in args.extraction_name or '\\' in args.extraction_name or args.extraction_name in ('', '.', '..'):
        parser.error('extraction-name must be a simple directory name')
    run(Path.cwd(), args.output, encode_only=args.encode_only, encoding_name=args.encoding_name,
        extraction_name=args.extraction_name, selection_file=args.selection_file, allow_full=args.allow_full)
