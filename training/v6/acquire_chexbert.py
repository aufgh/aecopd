"""固定官方代码和Stanford AIMI分发权重；不发送任何患者数据。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import requests
from concurrent.futures import ThreadPoolExecutor

REVISION = '6d22a96d73f18d0a7cf5b0dbebdac50cf8e4c1aa'
WEIGHT_REVISION = '6646433b3ad83a10f6e141db76d0ece44312b236'
WEIGHT_SHA = '6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1'
BERT_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def acquire(root):
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    session = requests.Session()
    def download(relative, url, expected=None):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if expected and digest(path) != expected:
                raise ValueError('existing asset hash mismatch: ' + relative)
        else:
            part = path.with_suffix(path.suffix + '.part')
            if expected:
                total=1314414442
                offset=part.stat().st_size if part.exists() else 0
                if not 0<=offset<=total: raise ValueError('partial asset size invalid')
                chunk=4*1024*1024
                starts=list(range(offset,total,chunk))
                # 先解析一次下载地址，避免每个小分段重复跨站重定向。
                with session.get(url,headers={'Range':'bytes=0-0'},timeout=(20,30),stream=True) as resolved:
                    resolved.raise_for_status()
                    download_url=resolved.url
                def ranged(start):
                    end=min(start+chunk,total)-1
                    for attempt in range(3):
                        try:
                            response=requests.get(download_url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,60))
                            response.raise_for_status()
                            if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}' or len(response.content)!=end-start+1:
                                raise ValueError('invalid download range')
                            return response.content
                        except requests.RequestException:
                            if attempt==2: raise
                started=time.perf_counter()
                # 有界并行范围请求，按顺序落盘；最后整文件SHA为身份验收。
                with part.open('ab') as handle, ThreadPoolExecutor(max_workers=16) as pool:
                    for i in range(0,len(starts),16):
                        for content in pool.map(ranged,starts[i:i+16]):
                            handle.write(content); offset+=len(content)
                        handle.flush()
                        print(json.dumps({'asset':relative,'bytes':offset,'total_bytes':total,'seconds':round(time.perf_counter()-started,1)}),flush=True)
            else:
                if part.exists(): raise FileExistsError(part)
                with session.get(url, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    with part.open('xb') as handle:
                        for block in response.iter_content(1024*1024): handle.write(block)
            if expected and digest(part) != expected:
                raise ValueError('download hash mismatch: ' + relative)
            part.rename(path)
        sources[relative] = {'url': url, 'sha256': digest(path), 'bytes': path.stat().st_size}
    for name in ('src/models/bert_labeler.py', 'src/bert_tokenizer.py', 'src/constants.py', 'src/label.py', 'README.md', 'LICENSE.pdf'):
        download('official/' + name, f'https://raw.githubusercontent.com/stanfordmlgroup/CheXbert/{REVISION}/{name}')
    for name in ('config.json', 'vocab.txt', 'tokenizer_config.json'):
        download('bert-base-uncased/' + name, f'https://huggingface.co/google-bert/bert-base-uncased/resolve/{BERT_REVISION}/{name}')
    download('chexbert.pth', f'https://huggingface.co/StanfordAIMI/RRG_scorers/resolve/{WEIGHT_REVISION}/chexbert.pth', WEIGHT_SHA)
    manifest = {'status': 'downloaded_hash_verified_not_inference_validated', 'official_revision': REVISION,
                'weight_repository': 'StanfordAIMI/RRG_scorers', 'weight_revision': WEIGHT_REVISION,
                'official_box_returned_404': True, 'official_box_hash_equivalence_not_independently_verified': True,
                'bert_revision': BERT_REVISION, 'sources': sources,
                'limits': ['Official README says released checkpoint trained on MIMIC-CXR, different from paper',
                           'Distribution card says MIT; preserve official academic license separately; no commercial permission inferred']}
    destination = root / 'asset_manifest.json'
    if destination.exists():
        if json.loads(destination.read_text()) != manifest: raise ValueError('asset manifest changed')
    else:
        destination.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'root': str(root)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('training/model_cache/CheXbert_stanford_2026-10-09'))
    try:
        acquire(parser.parse_args().output)
    except requests.RequestException as error:
        # 不把重定向的临时签名URL写入共享日志。
        raise RuntimeError('public asset download transport failure: '+type(error).__name__) from None
