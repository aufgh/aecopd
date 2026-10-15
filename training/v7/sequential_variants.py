"""有限单机制变体；原V7训练源码与已完成实验不修改。"""
from dataclasses import dataclass
import torch
from training.v6.risk import competing_nll
from training.v6.curve_score import daily_cif
from training.v7.model import V7Model
from training.v7.losses import HORIZONS, soft_focal


@dataclass(frozen=True)
class Variant:
    name: str
    nll: bool = True
    focal: bool = False
    cross: bool = True
    decay: bool = True
    lab_keys: bool = False
    brier: bool = False


VARIANTS = (
    Variant('focal_no_smoothing', nll=False, focal=True),
    Variant('nll_focal_no_smoothing', focal=True),
    Variant('without_cross_query', cross=False),
    Variant('without_explicit_decay', decay=False),
    Variant('drug_query_labs', lab_keys=True),
    Variant('nll_brier', brier=True),
)


class SequentialV7(V7Model):
    def __init__(self, baseline, drug_parent, variant, seed=42):
        super().__init__(baseline, drug_parent, seed=seed)
        self.variant = variant
        self.contrast_projection.requires_grad_(False)
        if not variant.cross:
            self.drug_cross.requires_grad_(False)
            self.rad_cross.requires_grad_(False)
        if not variant.decay:
            self.history_pool.decay = False
            self.history_pool.lambda_raw.requires_grad_(False)

    def visits(self, b):
        if self.variant.cross and not self.variant.lab_keys:
            return super().visits(b)
        n = len(b.context)
        lab_mask = b.labs[:, :, 6] > 0
        lab = self.lab_projection(torch.where(lab_mask[:, :, None], b.labs, 0.)) + self.lab_ids.weight
        eos_mask = lab_mask[:, 32:].any(1)
        lab = torch.cat((lab, self.eos_projection(torch.where(eos_mask[:, None], b.eos, 0.))[:, None]), 1)
        lab_mask = torch.cat((lab_mask, eos_mask[:, None]), 1)
        lab_rep = self.source_pool['lab'](lab, lab_mask)[0]
        numeric = b.medications[:, :, :4].clone()
        numeric[:, :, :2] = torch.where(numeric[:, :, 2:4].bool(), numeric[:, :, :2], 0.)
        drug = self.drug_ids.weight + self.drug_parent @ self.drug_parents.weight
        drug = self.drug_norm(drug + self.drug_numeric(numeric) + self.drug_units(b.medications[:, :, 4:]))
        text_rep, text_tokens, text_mask = self.text(b, n)
        rad = self.rad_projection(torch.where(b.rad_present[:, :, None], b.radiology, 0.)) + self.rad_ids.weight
        drug_rep = self.source_pool['drug'](drug, b.med_present)[0]
        rad_rep = self.source_pool['rad'](rad, b.rad_present)[0]
        reps = dict(text=text_rep, drug=drug_rep, lab=lab_rep, rad=rad_rep)
        masks = dict(text=text_mask.any(1), drug=b.med_present.any(1),
                     lab=lab_mask.any(1), rad=b.rad_present.any(1))
        if self.variant.cross:
            keys, present = torch.cat((text_tokens, rad, lab), 1), torch.cat((text_mask, b.rad_present, lab_mask), 1)
            drug = self.drug_cross(drug, b.med_present, keys, present)
            rad = self.rad_cross(rad, b.rad_present, text_tokens, text_mask)
        coarse_mask = b.coarse_meds[:, :, 2:].bool().any(2)
        coarse = self.coarse_projection(torch.where(coarse_mask[:, :, None], b.coarse_meds, 0.)) + self.coarse_ids.weight
        micro = self.micro_projection(torch.stack((torch.where(b.micro_present, b.micro, 0.), b.micro_present.float()), 2)) + self.micro_ids.weight
        support = self.support_projection(torch.stack((torch.where(b.support_present, b.support, 0.), b.support_present.float()), 2)) + self.support_ids.weight
        branches = [self.context_projection(b.context), lab_rep,
                    self.source_pool['drug'](drug, b.med_present)[0],
                    self.source_pool['coarse'](coarse, coarse_mask)[0],
                    self.source_pool['micro'](micro, b.micro_present)[0],
                    self.source_pool['support'](support, b.support_present)[0], text_rep,
                    self.source_pool['rad'](rad, b.rad_present)[0]]
        present = torch.stack((torch.ones(n, dtype=torch.bool, device=b.context.device), masks['lab'],
            masks['drug'], coarse_mask.any(1), b.micro_present.any(1), b.support_present.any(1),
            masks['text'], masks['rad']), 1)
        return self.fusion(torch.cat((*branches, present.float()), 1)), reps, masks


def loss_terms(out, times, events, variant):
    nll = competing_nll(out, times, events)
    penalty = 1e-4 * out.delta.square().mean()
    focal = brier = nll.detach() * 0
    if variant.focal or variant.brier:
        curve = daily_cif(out)
        days = torch.as_tensor(HORIZONS if variant.focal else (30, 90, 180, 365), device=times.device)
        causes = torch.tensor([1, 2], device=times.device)
        target = (events[:, None, None] == causes[None, None, :]) & (times[:, None, None] <= days[None, :, None])
        if variant.focal:
            focal = soft_focal(curve[:, days, 1:], target, smoothing=0.)
        if variant.brier:
            brier = (curve[:, days, 1:] - target.to(curve.dtype)).square().mean()
    total = penalty + focal + brier
    if variant.nll:
        total = total + nll
    return total, dict(nll=nll, penalty=penalty, focal=focal, brier=brier)
