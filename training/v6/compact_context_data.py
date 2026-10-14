"""显式紧凑多来源输入桥；不改变旧ClinicalBatch、配方或训练入口。"""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from patient_data.compact_multisource import MED_CHANNELS, MICRO_NAMES, SUPPORT_NAMES
from patient_data.v6_lab_state import MED_GROUPS, AXES
from patient_data.export_mimiciv import sha256
from training.v6.source_context_data import checked_arrays


@dataclass(frozen=True)
class CompactContextBatch:
    clinical: object          # 原本次诊断/22轴/文本；旧med字段仍仅执行证据
    med_features: object      # [N,7,4] 执行、仅医嘱、两种来源mask
    micro_values: object      # [N,5]
    micro_masks: object
    support_values: object    # [N,3]
    support_masks: object


@dataclass(frozen=True)
class CompactContextData:
    base: object
    arrays: dict
    provenance: dict

    def collate(self, positions, device='cpu'):
        import torch
        h, clinical, t, e = self.base.collate(positions, device=device)
        positions = np.asarray(positions, dtype=np.int64)
        tensor = lambda key: torch.as_tensor(self.arrays[key][positions], device=device)
        med = tensor('med_features')
        clinical = replace(clinical, med_source_mask=med[:, :, 2].bool())
        batch = CompactContextBatch(clinical, med, tensor('micro_values'), tensor('micro_masks'),
                                    tensor('support_values'), tensor('support_masks'))
        return h, batch, t, e


def attach(base, directory: Path):
    arrays, manifest = checked_arrays(directory, 'compact_multisource_validated_not_trained', 'compact_context.npz', base)
    names = {'med_group_names': tuple(MED_GROUPS), 'med_channel_names': MED_CHANNELS,
             'micro_names': MICRO_NAMES, 'support_names': SUPPORT_NAMES, 'axis_names': AXES}
    for key, expected in names.items():
        if tuple(arrays[key]) != expected: raise ValueError('compact vocabulary mismatch')
    n = len(base.arrays['hadm_ids'])
    for key, shape in {'med_features': (n, 7, 4), 'micro_values': (n, 5), 'micro_masks': (n, 5),
                       'support_values': (n, 3), 'support_masks': (n, 3)}.items():
        value = arrays[key]
        if value.shape != shape or not np.isfinite(value).all() or not np.isin(value, (0, 1)).all():
            raise ValueError('invalid compact tensor')
        if 'mask' in key and value.dtype != bool: raise ValueError('nonboolean compact mask')
    med = arrays['med_features']
    if not np.array_equal(med[:, :, 0], base.arrays['med_exposure']): raise ValueError('execution exposure changed')
    if ((med[:, :, 0] > 0) & (med[:, :, 2] == 0)).any() or ((med[:, :, 1] > 0) & (med[:, :, 3] == 0)).any():
        raise ValueError('positive medication without source')
    if (med[:, :, 0] * med[:, :, 1]).any(): raise ValueError('overlapping order-only evidence')
    for prefix in ('micro', 'support'):
        if ((arrays[prefix+'_values'] > 0) & ~arrays[prefix+'_masks']).any(): raise ValueError('positive without observed source')
    return CompactContextData(base, arrays, {'compact_manifest_sha256': sha256(directory/'manifest.json'),
                                             'training_performed': False})
