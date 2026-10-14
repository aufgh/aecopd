-- V5 章节对齐的只读来源盘点；不得修改现有队列或标签。
BEGIN TRANSACTION READ ONLY;

SELECT 'cohort' AS source, count(*) AS rows,
       count(DISTINCT subject_id) AS patients,
       count(DISTINCT hadm_id) AS admissions
FROM project_aecopd.copd_coded_all_index_labels_v2;

SELECT 'discharge_notes' AS source, count(*) AS rows,
       count(DISTINCT n.subject_id) AS patients,
       count(DISTINCT n.hadm_id) AS admissions,
       count(*) FILTER (WHERE n.storetime > l.dischtime) AS stored_after_discharge
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_note.discharge n USING (subject_id, hadm_id)
WHERE n.text IS NOT NULL AND btrim(n.text) <> '';

SELECT 'selected_labevents' AS source, count(*) AS rows,
       count(DISTINCT x.hadm_id) AS admissions,
       count(DISTINCT x.itemid) AS concepts
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_hosp.labevents x USING (hadm_id)
WHERE x.itemid IN (50818,50820,50821,50862,50882,50912,51222,51301)
  AND x.charttime BETWEEN l.admittime AND l.dischtime
  AND x.storetime <= l.dischtime AND x.valuenum IS NOT NULL AND x.valuenum >= 0;

SELECT 'diagnoses_icd' AS source, count(*) AS rows,
       count(DISTINCT d.hadm_id) AS admissions,
       count(DISTINCT (d.icd_version, d.icd_code)) AS concepts
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_hosp.diagnoses_icd d USING (hadm_id);

SELECT 'emar_administered' AS source, count(*) AS rows,
       count(DISTINCT e.hadm_id) AS admissions,
       count(DISTINCT lower(btrim(e.medication))) AS medication_names
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_hosp.emar e USING (hadm_id)
WHERE e.charttime BETWEEN l.admittime AND l.dischtime
  AND e.storetime <= l.dischtime
  AND e.event_txt IN ('Administered','Started','Applied','Delayed Administered')
  AND e.medication IS NOT NULL AND btrim(e.medication) <> '';

SELECT 'prescriptions' AS source, count(*) AS rows,
       count(DISTINCT p.hadm_id) AS admissions,
       count(*) FILTER (WHERE p.starttime >= l.dischtime - interval '24 hours'
                        AND p.starttime <= l.dischtime) AS started_last_24h,
       count(*) FILTER (WHERE p.stoptime IS NULL OR p.stoptime >= l.dischtime)
         AS not_stopped_before_discharge
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_hosp.prescriptions p USING (hadm_id);

SELECT 'pharmacy' AS source, count(*) AS rows,
       count(DISTINCT p.hadm_id) AS admissions,
       count(*) FILTER (WHERE p.entertime <= l.dischtime) AS entered_by_discharge,
       count(*) FILTER (WHERE p.starttime >= l.dischtime - interval '24 hours'
                        AND p.starttime <= l.dischtime) AS started_last_24h
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_hosp.pharmacy p USING (hadm_id);

ROLLBACK;
