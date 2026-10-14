-- =====================================================================
-- 诊断脚本 D3：超短间隔再入院 / 转院拆分（复核问题 3）
-- 目的：量化"再入院"事件中间隔过短(0~2 天)的比例，这些更可能是转院或同一次
--   临床事件被拆成多条住院记录，而非真正的新一次 AECOPD 发作。若占比高，
--   0~30 天区间的 AECOPD 阳性标签会被污染。
-- 只读：仅 SELECT。运行方式同 D1。
-- 前置表：project_aecopd.aecopd_longitudinal_history（含 days_to_next_observed_aecopd、
--   qc_nonpositive_next_gap、qc_short_gap、qc_overlap、qc_transfer、next_observed_aecopd_hadm_id）。
--   这些 QC 标志在 02/03 阶段已算好，这里只做汇总。
-- =====================================================================
\pset pager off
SET client_encoding = 'UTF8';

-- -------- 1) 相邻 AECOPD 间隔的短间隔占比（全部已观察到下一次者）--------
SELECT '1_next_gap_short_interval' AS section,
  count(*) FILTER (WHERE next_observed_aecopd_hadm_id IS NOT NULL)  AS observed_next_n,
  count(*) FILTER (WHERE qc_nonpositive_next_gap)                   AS nonpositive_gap_n,       -- <=0 天，需人工复核
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 1) AS gap_le_1d,
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 2) AS gap_le_2d,
  count(*) FILTER (WHERE qc_short_gap)                              AS gap_le_7d,               -- (0,7] 天
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 30) AS gap_le_30d,
  round(100.0*count(*) FILTER (WHERE qc_short_gap)
        /nullif(count(*) FILTER (WHERE days_to_next_observed_aecopd > 0),0),2) AS pct_le_7d_among_positive
FROM project_aecopd.aecopd_longitudinal_history;

-- -------- 2) 转院与重叠标志：可能的同一事件拆分 --------
SELECT '2_transfer_and_overlap_flags' AS section,
  count(*)                                     AS all_index_admissions,
  count(*) FILTER (WHERE qc_transfer)          AS admitted_via_transfer_from_hospital,
  count(*) FILTER (WHERE qc_overlap)           AS overlap_with_previous_strict,
  round(100.0*count(*) FILTER (WHERE qc_transfer)/nullif(count(*),0),2) AS pct_transfer,
  round(100.0*count(*) FILTER (WHERE qc_overlap)/nullif(count(*),0),2)  AS pct_overlap
FROM project_aecopd.aecopd_longitudinal_history;

-- -------- 3) 短间隔如何落在 0~30 天区间标签上（对 0-30 区间阳性的污染估计）--------
-- 把"下一次 AECOPD 在 30 天内"的阳性拆成 <=2 天 / 3-7 天 / 8-30 天，
-- <=2 天部分最可能是转院/拆分而非真正新发作。
SELECT '3_contamination_of_0_30_positive' AS section,
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 30) AS aecopd_0_30_positive,
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 2)  AS of_which_le_2d,
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 2 AND days_to_next_observed_aecopd <= 7)  AS of_which_3_7d,
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 7 AND days_to_next_observed_aecopd <= 30) AS of_which_8_30d,
  round(100.0*count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 2)
        /nullif(count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 30),0),2)
        AS pct_le_2d_of_0_30
FROM project_aecopd.aecopd_longitudinal_history;

-- -------- 4) 转院来源的短间隔交叉（转院 且 <=2 天，最强的拆分嫌疑）--------
SELECT '4_transfer_and_short_gap_cross' AS section,
  count(*) FILTER (WHERE qc_transfer AND days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 2) AS transfer_and_le_2d,
  count(*) FILTER (WHERE qc_transfer AND qc_short_gap)                                                          AS transfer_and_le_7d
FROM project_aecopd.aecopd_longitudinal_history;
