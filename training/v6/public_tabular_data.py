"""同XGBoost信息来源的公共矩阵；只在训练折拟合额外预处理。"""
from pathlib import Path
import joblib
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.audited_experiment import load, prepare, restrict, INPUT, OUTPUT
from training.v6.audited_model import flat_features
from training.v6.train import write_json


def numeric_valid(a, h, text_present):
    """给连续列拟合尺度/分位点的有效性；原二元mask列依然入网。"""
    n = len(h)
    masks = np.ones((n, 34), bool)
    masks[:, 4] = h[:, 5] == 0  # 无既往住院时不存在历史间隔。
    labs = np.broadcast_to(a['lab_present'][:, :, None], a['labs'].shape).copy()
    for j, name in enumerate(a['lab_names']):
        x = a['labs'][:, j]
        if name in ('eosinophils_absolute', 'eosinophils_percent'):
            labs[:, j, :6] = x[:, 6:12] > 0
        elif name in ('neutrophils_percent', 'lymphocytes_percent', 'rdw', 'anion_gap', 'chloride',
                      'neutrophils_absolute', 'lymphocytes_absolute', 'monocytes_percent', 'hematocrit',
                      'mcv', 'calcium_total', 'phosphate', 'protein_total', 'fibrinogen', 'troponin_t'):
            labs[:, j, :5] = x[:, 5:10] > 0
        else:
            for offset in (0, 9) if name in ('bicarbonate', 'wbc', 'sodium', 'potassium', 'glucose') else (0,):
                labs[:, j, offset:offset+6] = (x[:, offset+6] > 0)[:, None]
                labs[:, j, offset+3] &= x[:, offset+7] > 0
                labs[:, j, offset+5] &= x[:, offset+7] > 0
    drug = np.broadcast_to(a['med_present'][:, :, None], a['medications'].shape).copy()
    drug[:, :, :2] = a['medications'][:, :, 2:4] > 0
    tail = np.ones((n, 20+8+6), bool)
    # PCA输入已经包含四字段可用标记；“全字段空”的固定PCA表示仍有定义。
    # 不能再用是否有文本把这16坐标置零，否则与XGBoost的共同输入不一致。
    text = np.ones((n, 16), bool)
    return np.concatenate((masks, labs.reshape(n, -1), np.ones_like(a['lab_present']),
        drug.reshape(n, -1), np.ones_like(a['med_present']), tail, text), 1)


def fit_preprocessing(x, valid, train):
    if not np.isfinite(x).all() or x.shape != valid.shape:
        raise ValueError('invalid feature matrix')
    keep = np.ptp(x[train], axis=0) > 1e-7
    # 恰为0/1的列用类别嵌入，其余连续；不对稀有二元阳性除以小标准差。
    binary = np.all((x[train] == 0) | (x[train] == 1), axis=0) & keep
    num = np.flatnonzero(keep & ~binary)
    cat = np.flatnonzero(binary)
    center = np.zeros(len(num), np.float32)
    scale = np.ones(len(num), np.float32)
    bins = []
    for i, column in enumerate(num):
        values = x[train & valid[:, column], column]
        if not len(values):
            raise ValueError('nonconstant numeric column has no valid training values')
        center[i] = np.median(values)
        q25, q75 = np.quantile(values, [.25, .75])
        s = q75-q25
        if s <= 1e-7:
            s = np.std(values)
        scale[i] = s if s > 1e-7 else 1.
        scaled = np.clip((values-center[i])/scale[i], -8, 8)
        edges = np.unique(np.quantile(scaled, np.linspace(0, 1, 17))).astype(np.float32)
        # 有效观测全同但“有测/缺测”使原列变化：不从缺失零拟合区间。
        if len(edges) == 1:
            edges = np.array([edges[0]-1., edges[0]+1.], np.float32)
        bins.append(edges)
    return dict(num_columns=num, binary_columns=cat, center=center, scale=scale, bins=bins,
        original_width=x.shape[1], excluded_train_constant_columns=np.flatnonzero(~keep))


def transform(x, valid, parameters):
    num = parameters['num_columns']
    xn = np.clip((x[:, num]-parameters['center'])/parameters['scale'], -8, 8)
    xn = np.where(valid[:, num], xn, 0.).astype(np.float32)
    xb = x[:, parameters['binary_columns']].astype(np.float32)
    if not np.isin(xb, (0, 1)).all() or not np.isfinite(xn).all():
        raise ValueError('invalid transformed matrix')
    return xn, xb


def build(root):
    cfg, d, c, h, a, audit = load()
    previous = OUTPUT/'simple_controls/preprocessing.joblib'
    prep = joblib.load(previous)
    if not prep['compact']:
        raise ValueError('expected compact XGBoost recipe')
    matrices=[]; histories=[]; text=[]
    # 测试输入缓存不参与任何拟合或模型选择；测试预测仅在所有选择完成后运行。
    for start in range(0, len(d.times), 256):
        pos = np.arange(start, min(start+256, len(d.times)))
        batches = restrict(prepare(d, c, a, pos), a, True)
        for _, hi, b, _, _ in batches:
            raw = flat_features(b).cpu().numpy()
            histories.append(hi.cpu().numpy())
            matrices.append(np.concatenate((hi.cpu().numpy(), raw[:, :prep['boundary']],
                prep['text_pca'].transform(raw[:, prep['boundary']:])), 1))
            text.append(b.text_present.any(1).cpu().numpy())
    x = np.concatenate(matrices).astype(np.float32)
    valid = numeric_valid(a, np.concatenate(histories), np.concatenate(text))
    params = fit_preprocessing(x, valid, d.arrays['patient_splits']=='train')
    xn, xb = transform(x, valid, params)
    # PCA及尺度落盘用于新进程重放；不在验证/测试折重拟合。
    params['text_pca'] = prep['text_pca']
    params['boundary'] = prep['boundary']
    joblib.dump(params, root/'preprocessing.joblib')
    np.savez_compressed(root/'matrix.npz', x_num=xn, x_binary=xb, x_xgb=x,
        times=d.times, events=d.events, hadm_ids=d.arrays['hadm_ids'], subject_ids=d.arrays['subject_ids'],
        patient_splits=d.arrays['patient_splits'])
    report = dict(admissions=len(x), patients=len(np.unique(d.arrays['subject_ids'])),
        counts={f:int((d.arrays['patient_splits']==f).sum()) for f in ('train','validation','test')},
        original_features=x.shape[1], numeric_features=len(params['num_columns']), binary_features=len(params['binary_columns']),
        train_constant_excluded=len(params['excluded_train_constant_columns']), train_only_fit=True,
        input_sha256=sha256(INPUT/'interface.npz'), reused_text_pca_sha256=sha256(previous), data=d.provenance,
        matrix_sha256=sha256(root/'matrix.npz'), preprocessing_sha256=sha256(root/'preprocessing.joblib'))
    write_json(root/'data_contract.json', report)
    return report


def read(root):
    with np.load(root/'matrix.npz') as z:
        data = {key:z[key] for key in z.files}
    return data, joblib.load(root/'preprocessing.joblib')
