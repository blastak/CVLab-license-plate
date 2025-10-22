#!/usr/bin/env python3
"""
CSV 파일에서 각도 기준으로 상위 N개 데이터를 보여주는 스크립트
- 단일 CSV 파일 조회
- 여러 CSV 파일 통합 조회 (--pattern 옵션)
- 상위 샘플 이미지 저장 (--save-images 옵션)
"""

import argparse
import csv
import sys
import shutil
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


def load_multiple_csv_files(data_dir, pattern):
    """
    여러 CSV 파일을 통합하여 로드

    Args:
        data_dir: CSV 디렉토리
        pattern: glob 패턴

    Returns:
        list: [(csv_name, filename, plate_type, dimensions, x, y, z, sqrt, arccos, solvepnp), ...]
    """
    csv_files = sorted(Path(data_dir).glob(pattern))

    if not csv_files:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {data_dir}/{pattern}")
        sys.exit(1)

    print(f"📁 발견된 CSV 파일: {len(csv_files)}개")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")
    print()

    all_data = []

    for csv_path in csv_files:
        csv_name = csv_path.stem.replace('_GoodMatches', '')

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # 주석이나 헤더가 아닌 경우만 추가
                if not row or not row[0]:
                    continue
                if row[0].startswith('#'):
                    continue
                if row[0] == 'filename':  # 헤더
                    continue

                try:
                    # CSV 형식: filename, plate_type, dimensions, x_deg, y_deg, z_deg,
                    #           sqrt_method, arccos_method, solvepnp_normal_method
                    filename = row[0]
                    plate_type = row[1]
                    dimensions = row[2]
                    x_deg = float(row[3])
                    y_deg = float(row[4])
                    z_deg = float(row[5])
                    sqrt_val = float(row[6])
                    arccos_val = float(row[7])
                    solvepnp_val = float(row[8])

                    all_data.append((csv_name, filename, plate_type, dimensions,
                                   x_deg, y_deg, z_deg, sqrt_val, arccos_val, solvepnp_val))
                except (ValueError, IndexError):
                    continue

    print(f"📊 총 {len(all_data):,}개 데이터 로드 완료\n")
    return all_data


def find_image_path(csv_name, filename):
    """
    이미지 파일 경로 찾기 (CCPD2019 및 WebPlatemania 지원)

    Args:
        csv_name: CSV 이름 (예: ccpd_weather, WebPlatemania_P1-1)
        filename: 이미지 파일명

    Returns:
        Path or None: 이미지 파일 경로
    """
    # WebPlatemania 데이터셋인 경우
    if csv_name.startswith('WebPlatemania_'):
        plate_type = csv_name.replace('WebPlatemania_', '')
        webplatemania_dir = '/workspace/DB/01_LicensePlate/55_WebPlatemania_jpg_json_20250407'
        image_path = Path(webplatemania_dir) / f'GoodMatches_{plate_type}' / filename

        if image_path.exists():
            return image_path

        if not filename.endswith('.jpg'):
            image_path = Path(webplatemania_dir) / f'GoodMatches_{plate_type}' / f"{filename}.jpg"
            if image_path.exists():
                return image_path

        return None

    # CCPD2019 데이터셋인 경우
    ccpd_dir = '/workspace/DB/01_LicensePlate/CCPD2019'
    image_path = Path(ccpd_dir) / csv_name / 'GoodMatches_H22' / filename

    if image_path.exists():
        return image_path

    if not filename.endswith('.jpg'):
        image_path = Path(ccpd_dir) / csv_name / 'GoodMatches_H22' / f"{filename}.jpg"
        if image_path.exists():
            return image_path

    return None


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
        description='CSV 파일에서 각도 기준으로 상위 N개 데이터 표시 및 이미지 추출'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='입력 CSV 파일 경로 (단일 파일 모드)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='CSV 파일 디렉토리 (통합 모드, --pattern과 함께 사용)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        help='CSV 파일 패턴 (예: "ccpd_*_GoodMatches.csv", "WebPlatemania_*.csv")'
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
    parser.add_argument(
        '--save-images',
        type=str,
        default=None,
        help='상위 샘플 이미지를 저장할 디렉토리 (미지정 시 이미지 저장 안 함)'
    )

    args = parser.parse_args()

    # 모드 검증
    if not args.csv and not (args.data_dir and args.pattern):
        print("❌ --csv 또는 (--data_dir + --pattern) 중 하나를 지정해야 합니다.")
        print()
        print("예시:")
        print("  # 단일 파일 모드")
        print("  python show_top_angles.py --csv ccpd_weather_GoodMatches.csv --top 30")
        print()
        print("  # 통합 모드 (CCPD 전체)")
        print("  python show_top_angles.py --data_dir . --pattern 'ccpd_*_GoodMatches.csv' --top 30 --save-images top_30_ccpd")
        print()
        print("  # 통합 모드 (WebPlatemania 전체)")
        print("  python show_top_angles.py --data_dir . --pattern 'WebPlatemania_*.csv' --top 30 --save-images top_30_webplatemania")
        sys.exit(1)

    if args.csv and args.pattern:
        print("❌ --csv와 --pattern은 동시에 사용할 수 없습니다.")
        sys.exit(1)

    # 통합 모드
    if args.pattern:
        run_integrated_mode(args)
    # 단일 파일 모드
    else:
        run_single_file_mode(args)


def run_integrated_mode(args):
    """통합 모드: 여러 CSV 파일을 통합하여 상위 샘플 추출"""
    method_idx_map = {
        'sqrt': 7,
        'arccos': 8,
        'solvepnp': 9
    }
    method_idx = method_idx_map[args.method]

    # 출력 파일 설정
    outfile = open(args.out_txt, 'w', encoding='utf-8') if args.out_txt else sys.stdout

    try:
        print("=" * 120, file=outfile)
        print("📊 통합 모드: 여러 CSV 파일에서 상위 데이터 조회", file=outfile)
        print("=" * 120, file=outfile)
        print(f"📁 디렉토리: {args.data_dir}", file=outfile)
        print(f"🔍 패턴: {args.pattern}", file=outfile)
        print(f"📏 정렬 기준: {args.method}", file=outfile)
        print(f"🔢 표시 개수: {args.top}개", file=outfile)
        print(f"📈 정렬 순서: {'오름차순' if args.reverse else '내림차순'}", file=outfile)
        if args.save_images:
            print(f"💾 이미지 저장: {args.save_images}", file=outfile)
        print("=" * 120, file=outfile)
        print(file=outfile)

        # 여러 CSV 파일 로드
        all_data = load_multiple_csv_files(args.data_dir, args.pattern)

        # 정렬
        sorted_data = sorted(all_data, key=lambda x: x[method_idx], reverse=not args.reverse)

        # 상위 N개 추출
        top_samples = sorted_data[:args.top]

        # 결과 출력
        print(f"🏆 상위 {len(top_samples)}개 결과:", file=outfile)
        print("-" * 150, file=outfile)

        header_format = f"{'순위':<5} | {'데이터셋':<25} | {'파일명':<35} | {'타입':<6} | "
        header_format += f"{'X(°)':<7} | {'Y(°)':<7} | {'Z(°)':<7} | "
        header_format += f"{'sqrt':<7} | {'arccos':<7} | {'pnp_nv':<7}"
        print(header_format, file=outfile)
        print("-" * 150, file=outfile)

        for rank, (csv_name, filename, plate_type, dimensions, x_deg, y_deg, z_deg,
                   sqrt_val, arccos_val, solvepnp_val) in enumerate(top_samples, 1):
            # 현재 정렬 기준인 메소드 값 강조
            sqrt_str = f"*{sqrt_val:6.2f}" if args.method == 'sqrt' else f"{sqrt_val:7.2f}"
            arccos_str = f"*{arccos_val:6.2f}" if args.method == 'arccos' else f"{arccos_val:7.2f}"
            pnp_str = f"*{solvepnp_val:6.2f}" if args.method == 'solvepnp' else f"{solvepnp_val:7.2f}"

            print(f"{rank:<5} | {csv_name:<25} | {filename:<35} | {plate_type:<6} | "
                  f"{x_deg:7.2f} | {y_deg:7.2f} | {z_deg:7.2f} | "
                  f"{sqrt_str} | {arccos_str} | {pnp_str}", file=outfile)

        print("-" * 150, file=outfile)
        print(f"\n💡 '*' 표시는 현재 정렬 기준 컬럼입니다.\n", file=outfile)

        # 이미지 저장
        if args.save_images:
            output_path = Path(args.save_images)
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"\n📁 이미지 저장 디렉토리: {output_path}", file=outfile)
            print("🖼️  이미지 복사 중...\n", file=outfile)

            copied_count = 0

            for rank, (csv_name, filename, plate_type, dimensions, x_deg, y_deg, z_deg,
                       sqrt_val, arccos_val, solvepnp_val) in enumerate(top_samples, 1):
                angle_val = [sqrt_val, arccos_val, solvepnp_val][method_idx - 7]

                # 원본 이미지 경로 찾기
                src_path = find_image_path(csv_name, filename)

                if src_path is None:
                    print(f"   ⚠️  [{rank:2d}] 이미지 파일 없음: {csv_name}/{filename}", file=outfile)
                    continue

                # 목적지 파일명 (순위_각도_데이터셋_파일명)
                dst_filename = f"{rank:02d}_{angle_val:.2f}deg_{csv_name}_{src_path.name}"
                dst_path = output_path / dst_filename

                # 복사
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                    print(f"   ✅ [{rank:2d}] {angle_val:6.2f}° - {csv_name}/{src_path.name}", file=outfile)
                except Exception as e:
                    print(f"   ❌ [{rank:2d}] 복사 실패: {src_path} -> {e}", file=outfile)

            print(f"\n✅ {copied_count}개 이미지 저장 완료", file=outfile)

    finally:
        if args.out_txt and outfile != sys.stdout:
            outfile.close()
            print(f"✅ 결과가 {args.out_txt}에 저장되었습니다.")


def run_single_file_mode(args):
    """단일 파일 모드: 기존 로직"""
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
