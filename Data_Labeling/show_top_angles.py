#!/usr/bin/env python3
"""
CSV 파일에서 각도 기준으로 상위 N개 데이터를 보여주는 스크립트
"""

import argparse
import csv
import sys
from pathlib import Path


def read_csv_with_comments(csv_path):
    """
    주석이 포함된 CSV 파일 읽기

    Returns:
        tuple: (headers, rows, metadata)
    """
    metadata = []
    headers = None
    rows = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 주석 라인 수집
            if line.startswith('#'):
                metadata.append(line.strip())
                continue

            # CSV 파싱
            if headers is None:
                headers = line.strip().split(',')
            else:
                rows.append(line.strip().split(','))

    return headers, rows, metadata


def get_method_column_index(headers, method):
    """
    메소드 이름에 해당하는 컬럼 인덱스 찾기

    Args:
        headers: CSV 헤더 리스트
        method: 메소드 이름

    Returns:
        int: 컬럼 인덱스
    """
    method_map = {
        'sqrt': 'sqrt_method',
        'arccos': 'arccos_method',
        'solvepnp': 'solvepnp_normal_method'
    }

    target_column = method_map.get(method)
    if target_column is None:
        print(f"❌ 알 수 없는 메소드: {method}")
        print(f"   사용 가능한 메소드: {', '.join(method_map.keys())}")
        sys.exit(1)

    try:
        return headers.index(target_column)
    except ValueError:
        print(f"❌ CSV 파일에 '{target_column}' 컬럼이 없습니다.")
        print(f"   사용 가능한 컬럼: {', '.join(headers)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='CSV 파일에서 각도 기준으로 상위 N개 데이터 표시'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='입력 CSV 파일 경로'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['sqrt', 'arccos', 'solvepnp'],
        default='solvepnp',
        help='정렬 기준 메소드 (기본값: solvepnp)'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='상위 몇 개를 보여줄지 (기본값: 10)'
    )
    parser.add_argument(
        '--reverse',
        action='store_true',
        help='오름차순 정렬 (기본값: 내림차순)'
    )
    parser.add_argument(
        '--out_txt',
        type=str,
        default=None,
        help='결과를 저장할 txt 파일 경로 (기본값: stdout)'
    )

    args = parser.parse_args()

    # CSV 파일 존재 확인
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV 파일을 찾을 수 없습니다: {args.csv}")
        sys.exit(1)

    # 출력 파일 설정
    outfile = open(args.out_txt, 'w', encoding='utf-8') if args.out_txt else sys.stdout

    try:
        print("=" * 100, file=outfile)
        print("📊 번호판 각도 상위 데이터 조회", file=outfile)
        print("=" * 100, file=outfile)
        print(f"📁 CSV 파일: {csv_path.name}", file=outfile)
        print(f"📏 정렬 기준: {args.method}", file=outfile)
        print(f"🔢 표시 개수: {args.top}개", file=outfile)
        print(f"📈 정렬 순서: {'오름차순' if args.reverse else '내림차순'}", file=outfile)
        print("=" * 100, file=outfile)
        print(file=outfile)

        # CSV 읽기
        headers, rows, metadata = read_csv_with_comments(args.csv)

        # 메타데이터 출력
        if metadata:
            print("📋 CSV 메타데이터:", file=outfile)
            for line in metadata:
                print(f"   {line}", file=outfile)
            print(file=outfile)

        print(f"📁 전체 데이터: {len(rows)}개\n", file=outfile)

        # 정렬 기준 컬럼 인덱스
        method_idx = get_method_column_index(headers, args.method)

        # 각도 값으로 정렬 (내림차순)
        try:
            sorted_rows = sorted(
                rows,
                key=lambda row: float(row[method_idx]),
                reverse=not args.reverse  # reverse=True면 내림차순
            )
        except (ValueError, IndexError) as e:
            print(f"❌ CSV 데이터 정렬 중 오류 발생: {e}", file=outfile)
            sys.exit(1)

        # 상위 N개 선택
        top_rows = sorted_rows[:args.top]

        # 결과 출력
        print(f"🏆 상위 {len(top_rows)}개 결과:", file=outfile)
        print("-" * 130, file=outfile)

        # 헤더 출력
        header_format = f"{'순위':<5} | {'파일명':<30} | {'타입':<5} | {'크기':<10} | "
        header_format += f"{'X(°)':<7} | {'Y(°)':<7} | {'Z(°)':<7} | "
        header_format += f"{'sqrt':<7} | {'arccos':<7} | {'pnp_nv':<7}"
        print(header_format, file=outfile)
        print("-" * 130, file=outfile)

        # 데이터 출력
        filename_idx = headers.index('filename')
        plate_type_idx = headers.index('plate_type')
        dimensions_idx = headers.index('dimensions')
        x_idx = headers.index('x_deg')
        y_idx = headers.index('y_deg')
        z_idx = headers.index('z_deg')
        sqrt_idx = headers.index('sqrt_method')
        arccos_idx = headers.index('arccos_method')
        solvepnp_idx = headers.index('solvepnp_normal_method')

        for rank, row in enumerate(top_rows, 1):
            # 현재 정렬 기준인 메소드 값을 강조 표시
            sqrt_val = f"*{float(row[sqrt_idx]):6.2f}" if args.method == 'sqrt' else f"{float(row[sqrt_idx]):7.2f}"
            arccos_val = f"*{float(row[arccos_idx]):6.2f}" if args.method == 'arccos' else f"{float(row[arccos_idx]):7.2f}"
            pnp_val = f"*{float(row[solvepnp_idx]):6.2f}" if args.method == 'solvepnp' else f"{float(row[solvepnp_idx]):7.2f}"

            print(f"{rank:<5} | {row[filename_idx]:<30} | {row[plate_type_idx]:<5} | {row[dimensions_idx]:<10} | "
                  f"{float(row[x_idx]):7.2f} | {float(row[y_idx]):7.2f} | {float(row[z_idx]):7.2f} | "
                  f"{sqrt_val} | {arccos_val} | {pnp_val}", file=outfile)

        print("-" * 130, file=outfile)
        print(f"\n💡 '*' 표시는 현재 정렬 기준 컬럼입니다.\n", file=outfile)

        # 통계 정보
        all_angles = [float(row[method_idx]) for row in rows]
        print("📊 각도 통계 (정렬 기준 메소드):", file=outfile)
        print(f"   최대값: {max(all_angles):.2f}°", file=outfile)
        print(f"   최소값: {min(all_angles):.2f}°", file=outfile)
        print(f"   평균값: {sum(all_angles)/len(all_angles):.2f}°", file=outfile)
        print(file=outfile)

    finally:
        # 파일 핸들 닫기 (stdout이 아닌 경우만)
        if args.out_txt and outfile != sys.stdout:
            outfile.close()
            print(f"✅ 결과가 {args.out_txt}에 저장되었습니다.")


if __name__ == "__main__":
    main()
