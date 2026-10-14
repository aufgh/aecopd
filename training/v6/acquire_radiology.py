"""只下载公开专科模型资产，固定Hub版本及LFS哈希；绝不读取患者数据。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import requests


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def acquire(repo, root, names):
    response = requests.get(f'https://huggingface.co/api/models/{repo}', timeout=30)
    response.raise_for_status()
    revision = response.json()['sha']
    response = requests.get(f'https://huggingface.co/api/models/{repo}/tree/{revision}',
                            params={'blobs': 'true'}, timeout=30)
    response.raise_for_status()
    listing = {r['path']: r for r in response.json()}
    root.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name in names:
        metadata = listing[name]
        expected = metadata.get('lfs', {}).get('oid')
        path = root / name
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
        if not path.exists():
            part = path.with_suffix(path.suffix + '.part')
            offset = part.stat().st_size if part.exists() else 0
            total = metadata['size']
            if offset > total:
                raise ValueError('partial asset exceeds expected size')
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            started = time.perf_counter()
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                if offset and (r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise ValueError('resume range not honored')
                with part.open('ab' if offset else 'xb') as f:
                    last = time.perf_counter()
                    for block in r.iter_content(1024 * 1024):
                        f.write(block)
                        offset += len(block)
                        if time.perf_counter() - last > 15:
                            print(json.dumps({'asset': name, 'bytes': offset, 'total': total}), flush=True)
                            last = time.perf_counter()
            if part.stat().st_size != total or (expected and sha(part) != expected):
                raise ValueError('download size or SHA mismatch: ' + name)
            part.rename(path)
            print(json.dumps({'asset': name, 'seconds': time.perf_counter() - started}), flush=True)
        if path.stat().st_size != metadata['size'] or (expected and sha(path) != expected):
            raise ValueError('existing asset mismatch: ' + name)
        entries[name] = {'url': url, 'sha256': sha(path), 'lfs_sha256': expected, 'bytes': path.stat().st_size}
    manifest = {'status': 'public_assets_verified_not_inference_validated', 'repo': repo,
                'revision': revision, 'files': entries, 'patient_data_accessed': False}
    target = root / 'asset_manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('asset manifest differs; use a new directory')
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'repo': repo, 'revision': revision}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('radbert', 'radgraph', 'modernbert'), required=True)
    a = p.parse_args()
    if a.kind == 'radbert':
        acquire('zzxslp/RadBERT-RoBERTa-4m', Path('training/model_cache/RadBERT_2026-10-10'),
                ('README.md', 'config.json', 'merges.txt', 'vocab.json', 'special_tokens_map.json',
                 'tokenizer_config.json', 'pytorch_model.bin'))
    elif a.kind == 'radgraph':
        acquire('StanfordAIMI/RRG_scorers', Path('training/model_cache/RadGraph_XL_2026-10-10'),
                ('modern-radgraph-xl.tar.gz',))
    else:
        acquire('answerdotai/ModernBERT-base', Path('training/model_cache/ModernBERT_base_2026-10-10'),
                ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'model.safetensors'))
