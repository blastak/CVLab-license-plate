"""
OCR 결과 간단 분석 스크립트 (프롬프트 출력용)
"""

import csv
from collections import defaultdict

def calculate_char_distance(str1, str2):
    """두 문자열 간의 차이나는 문자 개수"""
    if str1 == str2:
        return 0
    len_diff = abs(len(str1) - len(str2))
    min_len = min(len(str1), len(str2))
    char_diff = sum(1 for i in range(min_len) if str1[i] != str2[i])
    return char_diff + len_diff

# CSV 읽기
csv_path = './output2/ocr_results_all.csv'
results = []

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)

print(f"총 {len(results)}개 OCR 결과 로드됨\n")

# 검출기별 통계
stats = defaultdict(lambda: {
    'total': 0, 'gt_exists': 0, 'exact': 0, 'tol1': 0, 'tol2': 0, 'no_det': 0,
    'by_class': defaultdict(lambda: {'total': 0, 'exact': 0, 'tol1': 0, 'tol2': 0})
})

for row in results:
    detector = row['detector']
    gt_text = row['gt_text']
    pred_text = row['pred_text']
    gt_class = row['gt_class']

    stats[detector]['total'] += 1

    if gt_text:
        stats[detector]['gt_exists'] += 1

        if not pred_text:
            stats[detector]['no_det'] += 1
        else:
            dist = calculate_char_distance(gt_text, pred_text)

            if dist == 0:
                stats[detector]['exact'] += 1
                stats[detector]['tol1'] += 1
                stats[detector]['tol2'] += 1
                stats[detector]['by_class'][gt_class]['exact'] += 1
                stats[detector]['by_class'][gt_class]['tol1'] += 1
                stats[detector]['by_class'][gt_class]['tol2'] += 1
            elif dist <= 1:
                stats[detector]['tol1'] += 1
                stats[detector]['tol2'] += 1
                stats[detector]['by_class'][gt_class]['tol1'] += 1
                stats[detector]['by_class'][gt_class]['tol2'] += 1
            elif dist <= 2:
                stats[detector]['tol2'] += 1
                stats[detector]['by_class'][gt_class]['tol2'] += 1

        stats[detector]['by_class'][gt_class]['total'] += 1

# 전체 요약 출력
print("="*100)
print("OCR 성능 비교 - 전체 요약")
print("="*100)
print(f"{'검출기':<12} {'총개수':>8} {'GT존재':>8} {'완전일치':>12} {'±1글자':>12} {'±2글자':>12} {'미검출':>12}")
print("-"*100)

for detector in sorted(stats.keys()):
    s = stats[detector]
    gt = s['gt_exists']

    exact_pct = (s['exact'] / gt * 100) if gt > 0 else 0
    tol1_pct = (s['tol1'] / gt * 100) if gt > 0 else 0
    tol2_pct = (s['tol2'] / gt * 100) if gt > 0 else 0
    no_det_pct = (s['no_det'] / gt * 100) if gt > 0 else 0

    print(f"{detector:<12} {s['total']:>8} {gt:>8} {s['exact']:>5} ({exact_pct:>5.1f}%) {s['tol1']:>5} ({tol1_pct:>5.1f}%) {s['tol2']:>5} ({tol2_pct:>5.1f}%) {s['no_det']:>5} ({no_det_pct:>5.1f}%)")

# 클래스별 상세
print("\n" + "="*100)
print("OCR 성능 비교 - 클래스별 상세")
print("="*100)

for detector in sorted(stats.keys()):
    print(f"\n[{detector}]")
    print(f"{'클래스':<10} {'총개수':>8} {'완전일치':>15} {'±1글자':>15} {'±2글자':>15}")
    print("-"*80)

    for cls in sorted(stats[detector]['by_class'].keys()):
        cs = stats[detector]['by_class'][cls]
        total = cs['total']

        exact_pct = (cs['exact'] / total * 100) if total > 0 else 0
        tol1_pct = (cs['tol1'] / total * 100) if total > 0 else 0
        tol2_pct = (cs['tol2'] / total * 100) if total > 0 else 0

        print(f"{cls:<10} {total:>8} {cs['exact']:>6} ({exact_pct:>5.1f}%) {cs['tol1']:>6} ({tol1_pct:>5.1f}%) {cs['tol2']:>6} ({tol2_pct:>5.1f}%)")

# 오류 사례
print("\n" + "="*100)
print("오류 사례 (최대 20개)")
print("="*100)

errors = []
for row in results:
    gt_text = row['gt_text']
    pred_text = row['pred_text']
    if gt_text and pred_text and gt_text != pred_text:
        dist = calculate_char_distance(gt_text, pred_text)
        errors.append((row['detector'], row['filename'], row['gt_class'], gt_text, pred_text, dist))

errors.sort(key=lambda x: x[5], reverse=True)

print(f"{'검출기':<12} {'파일명':<25} {'클래스':<8} {'GT':<12} {'예측':<12} {'차이':>4}")
print("-"*100)
for err in errors[:20]:
    print(f"{err[0]:<12} {err[1]:<25} {err[2]:<8} {err[3]:<12} {err[4]:<12} {err[5]:>4}")

print(f"\n총 오류 개수: {len(errors)}개")
