"""统一检验与可学习历史的2×2条件；完整输入在每步重新编码，非教师隐藏缓存。"""
import copy
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.model import masked_pool
from training.v6.risk import RiskOutput

ARMS = ('current_original', 'current_unified', 'history_original', 'history_unified')


def f1_at(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    if (y.dtype != bool or y.shape != score.shape or y.ndim != 1 or not len(y)
            or not np.isfinite(score).all() or np.any((score < 0)|(score > 1)) or not np.isfinite(threshold)):
        raise ValueError('invalid annual F1 inputs')
    flag = score >= threshold
    tp, fp, fn, tn = (int(x.sum()) for x in (flag&y, flag&~y, ~flag&y, ~flag&~y))
    return dict(threshold=float(threshold), f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
                false_positive_rate=fp/(fp+tn) if fp+tn else 0., tp=tp, fp=fp, fn=fn, tn=tn, alerts=tp+fp)


def best_f1(y, score):
    f1_at(y, score, 1.)  # 同一接口验证；分数同值必须同组。
    if not y.any(): return f1_at(y, score, 1.)
    order = np.argsort(-score, kind='stable'); p = score[order]; labels = y[order]
    last = np.r_[p[:-1] != p[1:], True]
    tp = np.cumsum(labels)[last]; count = np.arange(1,len(y)+1)[last]
    values = 2*tp/(count+y.sum())
    chosen = int(np.flatnonzero(values >= values.max()-1e-12)[0])
    return f1_at(y, score, float(p[last][chosen]))


class LongitudinalModel(nn.Module):
    def __init__(self, fresh, arm):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.original = copy.deepcopy(fresh)
        self.arm = arm; self.unified = arm.endswith('unified'); self.use_history = arm.startswith('history')
        # 所有条件分配同一新模块/RNG，开关而非构造顺序决定实际信息。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.lab_ids = nn.Embedding(36,128)
            self.eos_context = nn.Sequential(nn.Linear(16,32), nn.GELU(), nn.Linear(32,128))
            self.past_context = nn.Sequential(nn.LayerNorm(34), nn.Linear(34,128), nn.GELU())
            self.gap = nn.Sequential(nn.Linear(1,128), nn.Tanh())
            layer = nn.TransformerEncoderLayer(128,4,256,.1,batch_first=True,norm_first=True)
            self.temporal = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.history_head = nn.Linear(128,4)
            nn.init.zeros_(self.history_head.weight); nn.init.zeros_(self.history_head.bias)
        with torch.no_grad(): self.lab_ids.weight[:22].copy_(self.original.encoder.base.lab_ids.weight)
        self.original.baseline.requires_grad_(False)
        core = self.original.encoder.base
        for name, p in self.original.named_parameters():
            if name.startswith(('encoder.base.current', 'encoder.base.branch_norms.0', 'encoder.extra_meds')):
                p.requires_grad_(False)
            if self.unified and name.startswith(('encoder.extra_labs', 'encoder.base.lab_ids')):
                p.requires_grad_(False)
        self.lab_ids.requires_grad_(self.unified); self.eos_context.requires_grad_(self.unified)
        for module in (self.past_context,self.gap,self.temporal,self.history_head):
            module.requires_grad_(self.use_history)

    def train(self, mode=True):
        super().train(mode); self.original.baseline.eval(); return self

    def visits(self, compact, legacy_extra, unified, eos_context):
        core = self.original.encoder.base; b = compact.clinical; n = len(b.current_values)
        if self.unified:
            observed = unified[:,:,6] > 0
            clean = torch.where(observed[:,:,None], unified, 0.)
            l = masked_pool(core.lab(clean)+self.lab_ids(torch.arange(36,device=clean.device)), observed,core.lab_score)
            eos_present = observed[:,32:].any(1)
            l = l + self.eos_context(eos_context)*eos_present[:,None]
        else:
            observed = b.lab_axis_inputs[:,:,6] > 0
            clean = torch.where(observed[:,:,None],b.lab_axis_inputs,0.)
            l = masked_pool(core.lab(clean)+core.lab_ids(torch.arange(22,device=clean.device)), observed,core.lab_score)
        med = compact.med_features; mp = med[:,:,2:].bool().any((1,2)); m = core.med(med.flatten(1))
        micro = core.microbiology(torch.cat((compact.micro_values,compact.micro_masks.float()),1))
        support = core.support(torch.cat((compact.support_values,compact.support_masks.float()),1))
        present = [torch.zeros(n,dtype=torch.bool,device=clean.device),observed.any(1),mp,
                   compact.micro_masks.any(1),compact.support_masks.any(1)]
        s = torch.zeros_like(l)
        branches = [norm(x)*p[:,None] for norm,x,p in zip(core.branch_norms,(s,l,m,micro,support),present,strict=True)]
        text, text_present = core._text(b,n,mask_pmh=False)
        h = core.fusion(torch.cat((*branches,text,torch.stack((*present,text_present),1).float()),1))
        if not self.unified:
            h = h + self.original.encoder.extra_labs(legacy_extra)*legacy_extra.ne(0).any(1)[:,None]
        return h

    def forward(self, batch):
        compact, legacy, labs, eos, context, current_index, past_index, past_present, gaps = batch
        hidden = self.visits(compact, legacy, labs, eos)
        current = hidden[current_index]
        delta = self.original.delta_head(current).reshape(-1,2,2)
        if self.use_history:
            history = hidden[past_index] + self.past_context(context[past_index])
            history = history + self.gap(torch.log1p(gaps.clamp_min(0))[:,:,None]/6.)
            history = torch.where(past_present[:,:,None],history,0.)
            seq = torch.cat((current[:,None],history),1)
            valid = torch.cat((torch.ones(len(current),1,dtype=torch.bool,device=current.device),past_present),1)
            read = self.temporal(seq,src_key_padding_mask=~valid)[:,0]
            addition = self.history_head(read).reshape(-1,2,2)*past_present.any(1)[:,None,None]
            delta = delta + addition
        with torch.no_grad(): base = self.original.baseline(context[current_index])
        return RiskOutput((base.log_amplitude+delta[:,:,0]).clamp(-10.,6.),
                          F.softplus(self.original.baseline.raw_shape.detach()+delta[:,:,1])+.1,delta)


def make_batch(c, ext, interface, positions, neighbors, offsets, gaps, note_times, admission_times, *, history, device='cuda'):
    """按当前预测点构造已可见的历史；相同过去行的文本可用状态各自明确，不依赖batch其他人。"""
    keys = [(int(p),True) for p in positions]; unique = list(dict.fromkeys(keys)); by_key = {k:i for i,k in enumerate(unique)}
    past_keys = []; past_gaps = []
    for p in positions:
        selected = neighbors[offsets[p]:offsets[p+1]] if history else np.array([],int)
        relations = []
        for v in selected:
            key = (int(v),bool(note_times[v] <= admission_times[p]))
            if key not in by_key: by_key[key]=len(unique);unique.append(key)
            relations.append(by_key[key])
        past_keys.append(relations); past_gaps.append(gaps[offsets[p]:offsets[p+1]] if history else [])
    width = max(1,max(map(len,past_keys))); n=len(positions)
    index=np.zeros((n,width),np.int64); present=np.zeros((n,width),bool); times=np.zeros((n,width),np.float32)
    for i,(v,t) in enumerate(zip(past_keys,past_gaps,strict=True)):
        index[i,:len(v)]=v;present[i,:len(v)]=True;times[i,:len(v)]=t
    # 不直接让重复hadm进入既有collate；先唯一取值，再显式展开并重映射全部片段。
    source = np.array(sorted({p for p,_ in unique}),np.int64); source_slot={p:i for i,p in enumerate(source)}
    h, compact, _, _ = c.collate(source,device=device)
    from training.v6.exploration_model import expanded_inputs
    context = expanded_inputs(h,compact)
    expand=torch.as_tensor([source_slot[p] for p,_ in unique],device=device)
    b=compact.clinical
    vectors=[]; refs=[]
    for i,(p,allowed) in enumerate(unique):
        if not allowed: continue
        mask=b.snippet_index[:,0]==source_slot[p]
        vectors.append(b.snippet_vectors[mask]); r=b.snippet_index[mask].clone();r[:,0]=i;refs.append(r)
    fields = dict(current_values=b.current_values[expand]*0, current_masks=b.current_masks[expand]&False,
                  lab_axis_inputs=b.lab_axis_inputs[expand], med_exposure=b.med_exposure[expand],
                  med_source_mask=b.med_source_mask[expand],
                  snippet_vectors=torch.cat(vectors) if vectors else b.snippet_vectors[:0],
                  snippet_index=torch.cat(refs) if refs else b.snippet_index[:0])
    compact=replace(compact,clinical=replace(b,**fields),
        **{k:getattr(compact,k)[expand] for k in ('med_features','micro_values','micro_masks','support_values','support_masks')})
    original_positions=np.array([p for p,_ in unique])
    legacy=np.concatenate((ext['new_labs'][original_positions].reshape(len(unique),-1),ext['eos'][original_positions].reshape(len(unique),-1)),1)
    tensor=lambda x:torch.as_tensor(x,device=device)
    return (compact,tensor(legacy),tensor(interface['values'][original_positions]),tensor(interface['eos_context'][original_positions]),
            context[expand],tensor(np.array([by_key[k] for k in keys])),tensor(index),tensor(present),tensor(times))
