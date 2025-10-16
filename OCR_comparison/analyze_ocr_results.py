"""
OCR 결과 분석 스크립트
CSV 파일을 읽어서 다양한 tolerance로 정확도 계산 및 표 출력
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import pandas as pd


def calculate_char_distance(str1, str2):
    """
    두 문자열 간의 차이나는 문자 개수 계산 (간단한 버전)
    """
    if str1 == str2:
        return 0

    # 길이가 다른 경우
    len_diff = abs(len(str1) - len(str2))

    # 같은 위치의 다른 문자 개수
    min_len = min(len(str1), len(str2))
    char_diff = sum(1 for i in range(min_len) if str1[i] != str2[i])

    return char_diff + len_diff


def is_match_with_tolerance(gt_text, pred_text, tolerance):
    """
    tolerance 이내로 일치하는지 확인
    tolerance=0: 완전 일치
    tolerance=1: 1글자 틀려도 OK
    tolerance=2: 2글자 틀려도 OK
    """
    distance = calculate_char_distance(gt_text, pred_text)
    return distance <= tolerance


def analyze_results(csv_path):
    """
    CSV 파일을 읽어서 결과 분석
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"❌ CSV 파일이 존재하지 않습니다: {csv_path}")
        return None

    # CSV 읽기
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    print(f"총 {len(results)}개 OCR 결과 로드됨\n")

    # 검출기별 통계
    detector_stats = defaultdict(lambda: {
        'total': 0,
        'gt_exists': 0,
        'exact_match': 0,
        'tolerance_1': 0,
        'tolerance_2': 0,
        'no_detection': 0,
        'by_class': defaultdict(lambda: {
            'total': 0,
            'exact_match': 0,
            'tolerance_1': 0,
            'tolerance_2': 0
        })
    })

    for row in results:
        detector = row['detector']
        gt_text = row['gt_text']
        pred_text = row['pred_text']
        gt_class = row['gt_class']

        detector_stats[detector]['total'] += 1

        # GT가 있는 경우만 평가
        if gt_text:
            detector_stats[detector]['gt_exists'] += 1

            # 예측이 없는 경우
            if not pred_text:
                detector_stats[detector]['no_detection'] += 1
            else:
                # Tolerance별 매칭
                if is_match_with_tolerance(gt_text, pred_text, 0):
                    detector_stats[detector]['exact_match'] += 1
                    detector_stats[detector]['tolerance_1'] += 1
                    detector_stats[detector]['tolerance_2'] += 1

                    detector_stats[detector]['by_class'][gt_class]['exact_match'] += 1
                    detector_stats[detector]['by_class'][gt_class]['tolerance_1'] += 1
                    detector_stats[detector]['by_class'][gt_class]['tolerance_2'] += 1

                elif is_match_with_tolerance(gt_text, pred_text, 1):
                    detector_stats[detector]['tolerance_1'] += 1
                    detector_stats[detector]['tolerance_2'] += 1

                    detector_stats[detector]['by_class'][gt_class]['tolerance_1'] += 1
                    detector_stats[detector]['by_class'][gt_class]['tolerance_2'] += 1

                elif is_match_with_tolerance(gt_text, pred_text, 2):
                    detector_stats[detector]['tolerance_2'] += 1

                    detector_stats[detector]['by_class'][gt_class]['tolerance_2'] += 1

            # 클래스별 총 개수
            detector_stats[detector]['by_class'][gt_class]['total'] += 1

    return detector_stats, results


def print_summary_table(detector_stats):
    """
    전체 요약 테이블 출력
    """
    print("=" * 100)
    print("OCR 성능 비교 - 전체 요약")
    print("=" * 100)

    # 데이터 준비
    table_data = []

    for detector, stats in sorted(detector_stats.items()):
        gt_exists = stats['gt_exists']
        exact = stats['exact_match']
        tol1 = stats['tolerance_1']
        tol2 = stats['tolerance_2']
        no_det = stats['no_detection']

        exact_pct = (exact / gt_exists * 100) if gt_exists > 0 else 0
        tol1_pct = (tol1 / gt_exists * 100) if gt_exists > 0 else 0
        tol2_pct = (tol2 / gt_exists * 100) if gt_exists > 0 else 0
        no_det_pct = (no_det / gt_exists * 100) if gt_exists > 0 else 0

        table_data.append({
            '검출기': detector,
            '총 개수': stats['total'],
            'GT 존재': gt_exists,
            '완전 일치': f"{exact} ({exact_pct:.1f}%)",
            '±1글자': f"{tol1} ({tol1_pct:.1f}%)",
            '±2글자': f"{tol2} ({tol2_pct:.1f}%)",
            '미검출': f"{no_det} ({no_det_pct:.1f}%)"
        })

    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    print()


def print_class_table(detector_stats):
    """
    클래스별 상세 테이블 출력
    """
    print("=" * 100)
    print("OCR 성능 비교 - 클래스별 상세")
    print("=" * 100)

    for detector, stats in sorted(detector_stats.items()):
        print(f"\n[{detector}]")

        table_data = []

        for plate_class, class_stats in sorted(stats['by_class'].items()):
            total = class_stats['total']
            exact = class_stats['exact_match']
            tol1 = class_stats['tolerance_1']
            tol2 = class_stats['tolerance_2']

            exact_pct = (exact / total * 100) if total > 0 else 0
            tol1_pct = (tol1 / total * 100) if total > 0 else 0
            tol2_pct = (tol2 / total * 100) if total > 0 else 0

            table_data.append({
                '클래스': plate_class,
                '총 개수': total,
                '완전 일치': f"{exact} ({exact_pct:.1f}%)",
                '±1글자': f"{tol1} ({tol1_pct:.1f}%)",
                '±2글자': f"{tol2} ({tol2_pct:.1f}%)"
            })

        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        print()


def print_error_examples(results, max_examples=10):
    """
    오류 사례 출력
    """
    print("=" * 100)
    print(f"오류 사례 (최대 {max_examples}개)")
    print("=" * 100)

    errors = []

    for row in results:
        gt_text = row['gt_text']
        pred_text = row['pred_text']

        if gt_text and pred_text:
            if gt_text != pred_text:
                distance = calculate_char_distance(gt_text, pred_text)
                errors.append({
                    '검출기': row['detector'],
                    '파일명': row['filename'],
                    '클래스': row['gt_class'],
                    'GT': gt_text,
                    '예측': pred_text,
                    '차이': distance
                })

    if errors:
        # 차이가 큰 순서로 정렬
        errors_sorted = sorted(errors, key=lambda x: x['차이'], reverse=True)[:max_examples]

        df = pd.DataFrame(errors_sorted)
        print(df.to_string(index=False))
        print(f"\n총 오류 개수: {len(errors)}개")
    else:
        print("오류 없음!")

    print()


def save_detailed_excel(detector_stats, results, output_path):
    """
    상세 결과를 Excel 파일로 저장
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 시트 1: 전체 요약
        summary_data = []
        for detector, stats in sorted(detector_stats.items()):
            gt_exists = stats['gt_exists']
            exact = stats['exact_match']
            tol1 = stats['tolerance_1']
            tol2 = stats['tolerance_2']
            no_det = stats['no_detection']

            summary_data.append({
                '검출기': detector,
                '총_개수': stats['total'],
                'GT_존재': gt_exists,
                '완전_일치': exact,
                '완전_일치_%': (exact / gt_exists * 100) if gt_exists > 0 else 0,
                '±1글자': tol1,
                '±1글자_%': (tol1 / gt_exists * 100) if gt_exists > 0 else 0,
                '±2글자': tol2,
                '±2글자_%': (tol2 / gt_exists * 100) if gt_exists > 0 else 0,
                '미검출': no_det,
                '미검출_%': (no_det / gt_exists * 100) if gt_exists > 0 else 0
            })

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='전체_요약', index=False)

        # 시트 2: 클래스별 상세
        class_data = []
        for detector, stats in sorted(detector_stats.items()):
            for plate_class, class_stats in sorted(stats['by_class'].items()):
                total = class_stats['total']
                exact = class_stats['exact_match']
                tol1 = class_stats['tolerance_1']
                tol2 = class_stats['tolerance_2']

                class_data.append({
                    '검출기': detector,
                    '클래스': plate_class,
                    '총_개수': total,
                    '완전_일치': exact,
                    '완전_일치_%': (exact / total * 100) if total > 0 else 0,
                    '±1글자': tol1,
                    '±1글자_%': (tol1 / total * 100) if total > 0 else 0,
                    '±2글자': tol2,
                    '±2글자_%': (tol2 / total * 100) if total > 0 else 0
                })

        df_class = pd.DataFrame(class_data)
        df_class.to_excel(writer, sheet_name='클래스별_상세', index=False)

        # 시트 3: 전체 결과
        df_results = pd.DataFrame(results)
        df_results.to_excel(writer, sheet_name='전체_결과', index=False)

    print(f"상세 결과 Excel 저장: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR 결과 분석')
    parser.add_argument('--csv', type=str, required=True, help='OCR 결과 CSV 파일 경로')
    parser.add_argument('--output_excel', type=str, default='./ocr_analysis.xlsx',
                        help='출력 Excel 파일 경로')
    parser.add_argument('--max_errors', type=int, default=20, help='출력할 최대 오류 개수')

    args = parser.parse_args()

    # 결과 분석
    result = analyze_results(args.csv)

    if result:
        detector_stats, results = result

        # 테이블 출력
        print_summary_table(detector_stats)
        print_class_table(detector_stats)
        print_error_examples(results, max_examples=args.max_errors)

        # Excel 저장
        save_detailed_excel(detector_stats, results, args.output_excel)
