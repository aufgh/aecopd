-- MIMIC-III 补齐新版纵向数据集所需的基础来源表；原始表须由官方建表脚本创建且保持空表。
-- 仅加载当前需要的字典、实验室、处方、ICU、微生物和住院过程数据；
-- 33GB 的 CHARTEVENTS 不在本脚本中，避免未经容量评估就整表导入。
\set ON_ERROR_STOP on

DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM public.d_labitems)
     OR EXISTS (SELECT 1 FROM public.d_icd_diagnoses)
     OR EXISTS (SELECT 1 FROM public.d_items)
     OR EXISTS (SELECT 1 FROM public.labevents)
     OR EXISTS (SELECT 1 FROM public.prescriptions)
     OR EXISTS (SELECT 1 FROM public.icustays)
     OR EXISTS (SELECT 1 FROM public.microbiologyevents)
     OR EXISTS (SELECT 1 FROM public.procedures_icd)
     OR EXISTS (SELECT 1 FROM public.transfers) THEN
    RAISE EXCEPTION 'Core-feature import expects empty target tables; inspect before retrying';
  END IF;
END $$;

\copy public.D_LABITEMS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/D_LABITEMS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.D_ICD_DIAGNOSES FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/D_ICD_DIAGNOSES.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.D_ITEMS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/D_ITEMS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.ICUSTAYS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/ICUSTAYS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.PROCEDURES_ICD FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/PROCEDURES_ICD.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.TRANSFERS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/TRANSFERS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.MICROBIOLOGYEVENTS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/MICROBIOLOGYEVENTS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.PRESCRIPTIONS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/PRESCRIPTIONS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.LABEVENTS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/LABEVENTS.csv' DELIMITER ',' CSV HEADER NULL ''
