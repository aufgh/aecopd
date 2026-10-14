"""临床概念/药物身份/片段先交互后池化；独立候选，不改变旧V6。"""
from dataclasses import dataclass, replace
import numpy as np
import torch
from torch import nn
from training.v6.model import AnchoredModel, masked_pool


@dataclass(frozen=True)
class TokenBatch:
    labs: torch.Tensor         # [N,29,20] 保留原来源特征
    numeric_valid: torch.Tensor
    lab_present: torch.Tensor
    coarse: torch.Tensor       # [N,7,4]
    named: torch.Tensor        # [N,21,23]
    named_present: torch.Tensor
    other: torch.Tensor        # [N,8,2] 微生物/呼吸支持
    text: torch.Tensor         # [N,L,768] 不先平均
    text_fields: torch.Tensor
    text_present: torch.Tensor
    masked_labs: torch.Tensor  # 自监督遮盖标记，不是缺失=正常


class TokenEncoder(nn.Module):
    def __init__(self, *, interaction=False, flat=False, width=128):
        super().__init__(); self.width=width; self.interaction=interaction; self.flat=flat
        self.lab=nn.Sequential(nn.Linear(21,width),nn.GELU(),nn.LayerNorm(width))
        self.coarse=nn.Linear(4,width); self.named=nn.Linear(23,width); self.other=nn.Linear(2,width)
        self.ids=nn.Embedding(65,width); self.field_ids=nn.Embedding(4,width)
        self.types=nn.Embedding(5,width)
        self.local=nn.Sequential(nn.Linear(width,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,width),nn.LayerNorm(width))
        layer=nn.TransformerEncoderLayer(width,4,256,.1,batch_first=True,norm_first=True)
        self.cross=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.score=nn.Linear(width,1); self.output_norm=nn.LayerNorm(width)
        # 最小对照不作token交互，完整表格加四字段均值；宽度与层数明确记录。
        self.flat_net=nn.Sequential(nn.Linear(29*20+7*4+21*23+8*2+4*768+4,width),nn.GELU(),
                                    nn.Dropout(.1),nn.Linear(width,width),nn.GELU(),nn.LayerNorm(width))
        self.text_projection=nn.Linear(768,width)  # 最后初始化
        self.flat_net.requires_grad_(flat)
        self.cross.requires_grad_(interaction and not flat)
        if flat:
            for name,module in self.named_children():
                if name!='flat_net': module.requires_grad_(False)

    def tokens(self,b):
        n=len(b.labs); dev=b.labs.device
        if b.labs.shape!=(n,29,20) or b.named.shape!=(n,21,23): raise ValueError('token interface mismatch')
        parts=(self.lab(torch.cat((b.labs,b.masked_labs.float().unsqueeze(-1)),2)),
               self.coarse(b.coarse),self.named(b.named),self.other(b.other))
        x=torch.cat(parts,1)+self.ids(torch.arange(65,device=dev))
        types=torch.tensor([0]*29+[1]*7+[1]*21+[2]*5+[3]*3,device=dev)
        x=x+self.types(types)
        text=self.text_projection(b.text)+self.field_ids(b.text_fields)+self.types.weight[4]
        x=torch.cat((x,text),1)
        present=torch.cat((b.lab_present,b.coarse[:,:,2:].bool().any(2),b.named_present,
                           b.other[:,:,1].bool(),b.text_present),1)
        x=torch.where(present[:,:,None],x,0)
        safe=present.clone(); safe[~safe.any(1),0]=True
        # 无顺序位置编码：交换完整token及其身份/有效性不改变住院表示。
        x=x+self.local(x)
        if self.interaction: x=self.cross(x,src_key_padding_mask=~safe)
        return x,present

    def flat_features(self,b):
        n=len(b.labs); text=b.text.new_zeros(n,4,768); flags=b.text.new_zeros(n,4)
        for field in range(4):
            mask=b.text_present & (b.text_fields==field)
            text[:,field]=(b.text*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
            flags[:,field]=mask.any(1)
        return torch.cat((b.labs.flatten(1),b.coarse.flatten(1),b.named.flatten(1),b.other.flatten(1),text.flatten(1),flags),1)

    def forward(self,b,*,mask_pmh=False):
        if mask_pmh: b=replace(b,text_present=b.text_present & (b.text_fields!=1))
        if self.flat: return self.flat_net(self.flat_features(b)),{}
        x,present=self.tokens(b); h=masked_pool(x,present,self.score)
        return self.output_norm(h)*present.any(1)[:,None],{'tokens':x,'present':present}


class TokenModel(AnchoredModel):
    def __init__(self,baseline,*,interaction=False,flat=False):
        super().__init__(baseline,768,width=128)
        self.encoder=TokenEncoder(interaction=interaction,flat=flat)


def fast_cif(a,k,*,device='cpu'):
    """与既有.25天积分相同的向量化实现；独立测试核对后用于频繁验证。"""
    x=torch.as_tensor(a,dtype=torch.float64,device=device); s=torch.as_tensor(k,dtype=torch.float64,device=device)
    if x.ndim!=2 or x.shape[1]!=2 or s.shape!=x.shape or not torch.isfinite(x).all() or not torch.isfinite(s).all() or (s<=0).any():
        raise ValueError('invalid risk parameters')
    grid=np.unique(np.r_[np.arange(0,365.0001,.25),np.geomspace(1e-10,.25,80)])
    g=torch.as_tensor(grid,dtype=torch.float64,device=device)
    indices=torch.as_tensor(np.searchsorted(grid,np.arange(366)),device=device)
    chunks=[]
    for start in range(0,len(x),512):
        cumulative=x[start:start+512,None,:].exp()*(g[None,:,None]/365).pow(s[start:start+512,None,:])
        increments=torch.diff(cumulative,dim=1).clamp_min(0); total=increments.sum(2)
        survival=(-cumulative.sum(2)).exp()
        mass=survival[:,:-1,None]*(-torch.expm1(-total))[:,:,None]*increments/total[:,:,None].clamp_min(1e-300)
        cif=torch.cat((torch.zeros_like(mass[:,:1]),mass.cumsum(1)),1)
        chunks.append(torch.cat((survival[:,:,None],cif),2)[:,indices].cpu().numpy())
    return np.concatenate(chunks)


def daily_probabilities(curve):
    if curve.ndim!=3 or curve.shape[1:]!=(366,3) or not np.isfinite(curve).all(): raise ValueError('invalid daily curve')
    if not np.allclose(curve.sum(2),1,atol=1e-8) or np.any(np.diff(curve[:,:,1:],axis=1)<-1e-9):
        raise ValueError('nonconserving/nonmonotone curve')
    daily=np.diff(curve[:,:,1:],axis=1)
    conditional=np.divide(daily,curve[:,:-1,:1],out=np.zeros_like(daily),where=curve[:,:-1,:1]>1e-12)
    return daily,conditional


def masked_reconstruction_batch(b,seed,rate=.15):
    """成分相关轴同时屏蔽，避免只遮低值轴却从同一项目高值轴抄答案。"""
    generator=torch.Generator(device=b.labs.device).manual_seed(seed)
    # 22轴真实命名是 arterial_low_ph / glucose_low；不能只去除前缀。
    from patient_data.v6_lab_state import AXES
    names=list(AXES)+['neutrophils','lymphocytes','rdw','anion_gap','chloride','eosinophils','eosinophils']
    families=[]
    for name in names:
        if name.startswith('arterial_') or name.startswith('bicarbonate_'): key='blood_gas'
        else: key=name.removeprefix('low_').removeprefix('high_').removesuffix('_low').removesuffix('_high')
        families.append(key)
    keys=list(dict.fromkeys(families)); chosen=torch.rand(len(b.labs),len(keys),device=b.labs.device,generator=generator)<rate
    selected=chosen[:,torch.tensor([keys.index(f) for f in families],device=b.labs.device)] & b.lab_present
    values=torch.where(selected[:,:,None],0,b.labs)
    target_mask=b.numeric_valid & selected[:,:,None]
    return replace(b,labs=values,masked_labs=selected),target_mask
