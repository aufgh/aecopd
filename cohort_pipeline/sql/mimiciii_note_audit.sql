-- MIMIC-III 在当前 v3 索引住院上的出院小结覆盖；只输出汇总。
WITH dx AS (
  SELECT subject_id, hadm_id, MIN(seq_num) AS best_seq_num
  FROM public.diagnoses_icd WHERE icd9_code = '49121'
  GROUP BY subject_id, hadm_id
),
seq3_patients AS (
  SELECT DISTINCT subject_id FROM dx WHERE best_seq_num <= 3
),
cohort AS (
  SELECT a.subject_id, a.hadm_id, a.dischtime
  FROM public.admissions a
  JOIN dx d USING (subject_id, hadm_id)
  JOIN seq3_patients p USING (subject_id)
  WHERE a.hospital_expire_flag = 0 AND a.dischtime > a.admittime
),
per_admission AS (
  SELECT c.hadm_id,
         COUNT(n.row_id) AS discharge_notes,
         COUNT(n.row_id) FILTER (WHERE n.charttime IS NOT NULL AND n.storetime IS NOT NULL) AS notes_with_both_times,
         COUNT(n.row_id) FILTER (WHERE n.charttime <= c.dischtime AND n.storetime <= c.dischtime) AS notes_available_by_discharge,
         COUNT(n.row_id) FILTER (WHERE n.chartdate <= c.dischtime::date) AS notes_chartdated_by_discharge
  FROM cohort c
  LEFT JOIN public.noteevents n ON n.subject_id = c.subject_id AND n.hadm_id = c.hadm_id
                                 AND n.category = 'Discharge summary' AND COALESCE(n.iserror, '0') <> '1'
  GROUP BY c.hadm_id
)
SELECT COUNT(*) AS index_admissions,
       COUNT(*) FILTER (WHERE discharge_notes > 0) AS with_discharge_note,
       COUNT(*) FILTER (WHERE notes_with_both_times > 0) AS with_note_both_times,
       COUNT(*) FILTER (WHERE notes_available_by_discharge > 0) AS with_note_available_by_discharge,
       COUNT(*) FILTER (WHERE notes_chartdated_by_discharge > 0) AS with_note_chartdated_by_discharge,
       SUM(discharge_notes) AS discharge_note_rows
FROM per_admission;
