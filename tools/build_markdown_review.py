"""按已审查源码清单构建或核验 Markdown 阅读包；不自动发现私有数据。"""
import argparse
import hashlib
import json
from pathlib import Path
import re

MAX_BYTES = 800_000
FOLDER = 'review_bundle'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def chunks(raw):
    lines = raw.splitlines(keepends=True)
    if not lines:
        yield 1, 0, b''
        return
    start = 1; pending = []; size = 0
    for number, line in enumerate(lines, 1):
        if len(line) > MAX_BYTES-4096:
            raise ValueError('single source line exceeds fragment limit')
        if pending and size+len(line) > MAX_BYTES-4096:
            yield start, number-1, b''.join(pending)
            start = number; pending = []; size = 0
        pending.append(line); size += len(line)
    yield start, len(lines), b''.join(pending)


def render(root, catalog):
    groups = {}; locations = {}
    for item in catalog['sources']:
        path = root/item['path']
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('catalog path escapes public checkout')
        raw = path.read_bytes()
        if digest(raw) != item['sha256'] or len(raw) != item['bytes']:
            raise ValueError('reviewed source changed: '+item['path'])
        raw.decode('utf-8')
        locations[item['path']] = []
        for part, (start, end, fragment) in enumerate(chunks(raw), 1):
            text = fragment.decode('utf-8')
            runs = [len(v) for v in re.findall(r'`+', text)]
            fence = '`'*max(4, max(runs, default=0)+1)
            language = {'.py':'python', '.sql':'sql', '.json':'json', '.md':'markdown',
                        '.yaml':'yaml', '.yml':'yaml', '.ps1':'powershell', '.csv':'csv'}.get(path.suffix, 'text')
            added = not fragment.endswith(b'\n')
            meta = dict(path=item['path'], part=part, start_line=start, end_line=end,
                        source_sha256=item['sha256'], fragment_sha256=digest(fragment),
                        added_terminal_newline=added)
            block = ('\n## '+item['path']+f' · lines {start}–{end}\n\n'
                     +'<!-- REVIEW_FRAGMENT '+json.dumps(meta, ensure_ascii=False, separators=(',',':'))+' -->\n'
                     +fence+language+'\n'+text+('\n' if added else '')+fence+'\n'
                     +'<!-- END REVIEW_FRAGMENT -->\n')
            groups.setdefault(item['group'], []).append((item['path'], block))
    result = {}
    for group, blocks in sorted(groups.items()):
        part = 1; current = f'# {group} · {part}\n\n源文件正文或经过字段白名单提取的汇总数据，来源身份见每段元数据。\n'
        used = []
        for path, block in blocks:
            if used and len((current+block).encode('utf-8')) > MAX_BYTES:
                name = f'{group}_{part:02d}.md'; result[name] = current.encode('utf-8')
                for source in set(used):locations[source].append(name)
                part += 1; current = f'# {group} · {part}\n\n续篇；文件碎片按part序号拼接可还原原文。\n'; used = []
            current += block; used.append(path)
        if used:
            name = f'{group}_{part:02d}.md'; result[name] = current.encode('utf-8')
            for source in set(used):locations[source].append(name)
    index = ['# AECOPD 完整公开源码、汇总结果与数据结构快照', '',
        '这是替代旧00–07 TXT摘录的Markdown阅读包。每段都是可直接读取的正文；不需要解压ZIP，也不把文件目录当成已读源码。', '',
        '版本父提交：`'+catalog['base_public_commit']+'`。本次新增/更新内容以source_catalog.json中的SHA256绑定；最终发布身份以当前Git提交为准。', '',
        f"收录{len(catalog['sources'])}份文本文件，分成{len(result)}个正文分卷；所有分卷合计正文可逐字节重建。", '',
        '**阅读顺序**', '',
        '先读数据快照和结果，再按任务读取V7/V6及其依赖；历史V5/基线/SQL/测试全部保留。若会话上下文不够，可分批上传主题分卷。只有确实读取的正文才能作为源码证据。', '',
        '**边界与口径**', '',
        '“数据快照”包含表规模、划分统计、字段、数组shape/dtype、文件级hash；不包含患者行、ID取值、逐人标签/划分/预测、文本、向量或权重。'
        '历史汇总JSON经指标字段白名单提取，明确不是原始完整报告；不同任务/划分/标签的结果不可混排。合成测试不能证明独立临床有效性。', '',
        '当前任务总计19,401次合格出院/5,351患者：训练13,592/3,766，调参1,531/402，确认1,418/395，历史测试2,860/788。'
        '同患者不跨划分；历史确认/测试有项目早期曝光。本次发布不重训、不重新评估其结局。', '',
        '**分卷**', '', '| 文件 | 字节数 |', '|---|---:|']
    for name, raw in result.items():index.append(f'| [{name}]({name}) | {len(raw):,} |')
    index += ['', '**完整文件定位**', '', '| 原路径 | 正文分卷 |', '|---|---|']
    for path, names in sorted(locations.items()):
        index.append('| `'+path+'` | '+'、'.join(f'[{n}]({n})' for n in names)+' |')
    index += ['', '**不在公开范围内的源码**', '',
              '此前审查中以下9份原型含历史病例引用，保留本地，不把它们伪装成完整公开代码：', '']
    index.extend('- `'+name+'`' for name in catalog['withheld_code'])
    index += ['', '可重装依赖、缓存、重复发布现场、私有数据目录和本地旧Git历史不属于此公开源码包。'
        'PNG等既有汇总图仍在仓库原路径，本包索引列为二进制附件，不将其编码为伪Markdown数据。', '',
        '**仓库中保留的二进制汇总附件**', '']
    index.extend('- `'+item['path']+'` · '+str(item['bytes'])+' bytes · SHA256 `'+item['sha256']+'`' for item in catalog['binary_assets'])
    index += ['', '核验：`python tools/build_markdown_review.py --verify`。该工具仅读取已审查的清单路径，不递归收集本地数据。', '']
    result['00_READ_FIRST.md'] = '\n'.join(index).encode('utf-8')
    manifest = dict(base_public_commit=catalog['base_public_commit'], source_catalog_sha256=digest(json.dumps(catalog,ensure_ascii=False,indent=2).encode('utf-8')+b'\n'),
                    source_count=len(catalog['sources']), markdown_files=len(result),
                    outputs={name:dict(bytes=len(raw),sha256=digest(raw)) for name,raw in result.items()},
                    payload_bytes=sum(item['bytes'] for item in catalog['sources']),
                    all_source_bytes_reconstructible=True, contains_patient_values=False)
    result['manifest.json'] = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    return result


def reconstruct(root, catalog):
    collected = {}
    pattern = re.compile(r'<!-- REVIEW_FRAGMENT (\{[^\n]+\}) -->\n(?P<fence>`{4,})[^\n]*\n(?P<body>.*?)(?P=fence)\n<!-- END REVIEW_FRAGMENT -->', re.S)
    for p in sorted((root/FOLDER).glob('*.md')):
        for match in pattern.finditer(p.read_bytes().decode('utf-8')):
            meta = json.loads(match[1]); body = match['body'].encode('utf-8')
            if meta['added_terminal_newline']:
                if not body.endswith(b'\n'):raise ValueError('fragment terminator missing')
                body = body[:-1]
            if digest(body) != meta['fragment_sha256']:
                raise ValueError('fragment bytes differ: '+meta['path'])
            collected.setdefault(meta['path'], []).append((meta['part'], body))
    if set(collected) != {item['path'] for item in catalog['sources']}:
        raise ValueError('source coverage differs')
    for item in catalog['sources']:
        pieces = sorted(collected[item['path']])
        if [i for i,_ in pieces] != list(range(1,len(pieces)+1)):
            raise ValueError('missing or duplicate fragment')
        raw = b''.join(body for _,body in pieces)
        if raw != (root/item['path']).read_bytes() or digest(raw) != item['sha256']:
            raise ValueError('reconstructed source differs: '+item['path'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root/FOLDER/'source_catalog.json').read_text(encoding='utf-8'))
    files = render(root, catalog)
    for name, raw in files.items():
        target = root/FOLDER/name
        if args.verify:
            if not target.is_file() or target.read_bytes() != raw:
                raise ValueError('bundle differs: '+name)
        else:
            if target.exists():raise FileExistsError('existing bundle; use --verify')
            target.write_bytes(raw)
    reconstruct(root, catalog)
    print(json.dumps(dict(source_files=len(catalog['sources']), markdown_files=sum(n.endswith('.md') for n in files),
                          markdown_bytes=sum(len(v) for n,v in files.items() if n.endswith('.md')),
                          reconstructed_source_bytes=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
