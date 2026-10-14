"""对已逐条阅读的两组局部回归保存显式审阅；不是自动临床判真。"""
import hashlib
import json
from training.report_bonsai_catalog16_guarded import read_arm
from training.bonsai_catalog16_guarded import ROOT


def run():
    entries = {}
    for arm, expected in [('four', 24), ('two', 21)]:
        results, _ = read_arm(arm)
        if sum(len(v) for r in results for v in r['returned_quotes'].values()) != expected:
            raise ValueError('New outputs require fresh original-note review')
        for r in results:
            for key, values in r['returned_quotes'].items():
                for i, quote in enumerate(values):
                    grade, reason = 'correct', '已逐条比对完整原文；时间、适应证、字段和片段范围符合此局部回归定义。'
                    if r['review_id'] == 'R014' and key == 'respiratory_antibiotic_course' and quote.startswith('___ the ED, initial vitals'):
                        grade, reason = 'incorrect', '急诊整段混杂生命体征及检验，未聚焦抗感染方案。'
                    if r['review_id'] == 'R014' and key == 'respiratory_antibiotic_course' and 'Follow up with your PCP' in quote:
                        grade, reason = 'incorrect', '抗生素停药日期被粘到下一条无关随访嘱托。'
                    if r['review_id'] == 'R018' and key == 'systemic_steroid_taper':
                        grade, reason = 'incorrect', '原文激素属于脊髓病变术后疗程，不能仅因COPD史填作呼吸激素计划。'
                    entries[f'{arm}:{r["review_id"]}:{key}:{i}'] = {'sha256': hashlib.sha256(quote.encode()).hexdigest(), 'grade': grade, 'reason': reason}
    output = ROOT/'catalog16_guarded_review_2026-10-04.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({'reviewer': 'assistant_not_independent_physician', 'all_returned_quotes_reviewed_against_full_notes': True, 'quotes': entries}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run()
