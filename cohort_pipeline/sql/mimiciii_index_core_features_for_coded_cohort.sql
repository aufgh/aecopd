-- 非覆盖性索引：支持按 hadm_id 抽取新队列的原始特征。
-- 官方完整索引脚本包含 DROP INDEX；这里仅创建当前所需索引。
\set ON_ERROR_STOP on

CREATE INDEX IF NOT EXISTS mimiciii_labevents_hadm_chart_idx
  ON public.labevents(hadm_id,charttime);
CREATE INDEX IF NOT EXISTS mimiciii_prescriptions_hadm_start_idx
  ON public.prescriptions(hadm_id,startdate);
CREATE INDEX IF NOT EXISTS mimiciii_microbiologyevents_hadm_chart_idx
  ON public.microbiologyevents(hadm_id,charttime);
CREATE INDEX IF NOT EXISTS mimiciii_icustays_hadm_idx
  ON public.icustays(hadm_id);
CREATE INDEX IF NOT EXISTS mimiciii_procedures_icd_hadm_idx
  ON public.procedures_icd(hadm_id);
CREATE INDEX IF NOT EXISTS mimiciii_transfers_hadm_idx
  ON public.transfers(hadm_id);
