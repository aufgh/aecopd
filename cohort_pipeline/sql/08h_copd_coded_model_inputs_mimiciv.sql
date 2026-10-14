-- MIMIC-IV：新队列低维结构化训练输入。只使用索引出院前已可用的检验结果。
-- 处方、诊断细分类及文本留给独立增量实验，避免将可用性清单直接当作特征。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_model_base_v1 AS
SELECT l.subject_id,l.hadm_id,l.source_dataset,l.event_type,l.time_days,
       LEAST(90,GREATEST(0,p.anchor_age+extract(year FROM l.admittime)-p.anchor_year))::double precision
         AS age_at_admission,
       CASE p.gender WHEN 'M' THEN 1 WHEN 'F' THEN 0 END AS male,
       c.prior_coded_admission_count,
       (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate h
        WHERE h.subject_id=l.subject_id AND h.is_strict_aecopd
          AND h.dischtime<=l.admittime) AS prior_strict_aecopd_count,
       m.has_timed_discharge_summary,m.has_timed_other_note
FROM project_aecopd.copd_coded_strict_index_labels_v1 l
JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
JOIN mimiciv_hosp.patients p ON p.subject_id=l.subject_id
JOIN project_aecopd.copd_coded_dataset_manifest m USING(hadm_id);

ALTER TABLE project_aecopd.copd_coded_model_base_v1 ADD PRIMARY KEY(hadm_id);

CREATE TABLE project_aecopd.copd_coded_model_labs_v1 AS
WITH selected AS MATERIALIZED (
  SELECT l.hadm_id,x.itemid,x.charttime,x.labevent_id,x.valuenum
  FROM project_aecopd.copd_coded_strict_index_labels_v1 l
  JOIN mimiciv_hosp.labevents x ON x.hadm_id=l.hadm_id
  WHERE x.itemid IN (50818,50821,51301,51222,50912)
    AND x.charttime BETWEEN l.admittime AND l.dischtime
    AND x.storetime<=l.dischtime
    AND x.valuenum IS NOT NULL
    AND x.valuenum>=0
), ranked AS (
  SELECT *,
         row_number() OVER (PARTITION BY hadm_id,itemid ORDER BY charttime,labevent_id)
           AS first_rank,
         row_number() OVER (PARTITION BY hadm_id,itemid ORDER BY charttime DESC,labevent_id DESC)
           AS last_rank
  FROM selected
)
SELECT hadm_id,itemid,count(*) AS measurement_count,
       max(valuenum) FILTER (WHERE first_rank=1) AS first_value,
       max(valuenum) FILTER (WHERE last_rank=1) AS last_value,
       min(valuenum) AS min_value,max(valuenum) AS max_value
FROM ranked GROUP BY hadm_id,itemid;

ALTER TABLE project_aecopd.copd_coded_model_labs_v1 ADD PRIMARY KEY(hadm_id,itemid);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_model_base_v1)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_strict_index_labels_v1)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_model_base_v1
                WHERE age_at_admission IS NULL OR male IS NULL
                   OR prior_coded_admission_count<0 OR prior_strict_aecopd_count<0)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_model_labs_v1
                WHERE measurement_count<1 OR first_value IS NULL OR last_value IS NULL)
  THEN RAISE EXCEPTION 'MIMIC-IV model input consistency check failed'; END IF;
END $$;
COMMIT;
