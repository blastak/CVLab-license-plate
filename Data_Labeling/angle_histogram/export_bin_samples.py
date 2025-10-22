#!/usr/bin/env python3
"""
히스토그램 bin별 샘플 이미지 추출 스크립트

각 각도 bin에 해당하는 이미지를 샘플로 추출하여 폴더에 저장합니다.

사용 예시:
    python export_bin_samples.py --method arccos --samples 10
    python export_bin_samples.py --method solvepnp --bins 30 --samples 5
"""

import argparse
import csv
import os
import shutil
from pathlib import Path
import numpy as np
from collections import defaultdict


def read_csv_data(csv_path):
    """
    CSV 파일 읽기 (주석 제외)

    Returns:
        list: [(filename, sqrt, arccos, solvepnp), ...]
    """
    data_rows = []

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
                filename = row[0]
                sqrt_val = float(row[6])
                arccos_val = float(row[7])
                solvepnp_val = float(row[8])
                data_rows.append((filename, sqrt_val, arccos_val, solvepnp_val))
            except (ValueError, IndexError):
                continue

    return data_rows


def load_all_csv_files(data_dir):
    """
    디렉토리의 모든 CSV 파일 로드

    Returns:
        list: [(csv_name, filename, sqrt, arccos, solvepnp), ...]
    """
    csv_files = sorted(Path(data_dir).glob('*.csv'))

    if not csv_files:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {data_dir}")
        return None

    print(f"📁 발견된 CSV 파일: {len(csv_files)}개")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")
    print()

    all_data = []

    for csv_path in csv_files:
        print(f"📄 읽는 중: {csv_path.name}")
        rows = read_csv_data(csv_path)

        # CSV 이름 추출 (예: ccpd_weather_GoodMatches.csv -> ccpd_weather)
        csv_name = csv_path.stem.replace('_GoodMatches', '')

        for filename, sqrt_val, arccos_val, solvepnp_val in rows:
            all_data.append((csv_name, filename, sqrt_val, arccos_val, solvepnp_val))

        print(f"   ✅ {len(rows):,}개 행 로드")

    print(f"\n📊 총 {len(all_data):,}개 데이터 로드 완료\n")
    return all_data


def find_image_path(csv_name, filename, base_dir='/workspace/DB/01_LicensePlate/CCPD2019'):
    """
    이미지 파일 경로 찾기

    Args:
        csv_name: CSV 이름 (예: ccpd_weather)
        filename: 이미지 파일명
        base_dir: CCPD2019 기본 디렉토리

    Returns:
        Path or None: 이미지 파일 경로
    """
    # 예상 경로: /workspace/DB/01_LicensePlate/CCPD2019/ccpd_weather/GoodMatches_H22/파일명
    image_path = Path(base_dir) / csv_name / 'GoodMatches_H22' / filename

    if image_path.exists():
        return image_path

    # .jpg 확장자 추가 시도
    if not filename.endswith('.jpg'):
        image_path = Path(base_dir) / csv_name / 'GoodMatches_H22' / f"{filename}.jpg"
        if image_path.exists():
            return image_path

    return None


def extract_bin_samples(data, method='arccos', bins=30, samples_per_bin=10, output_dir='bin_samples', base_dir='/workspace/DB/01_LicensePlate/CCPD2019'):
    """
    bin별 샘플 이미지 추출

    Args:
        data: [(csv_name, filename, sqrt, arccos, solvepnp), ...]
        method: 정렬 기준 메소드
        bins: bin 개수
        samples_per_bin: bin당 샘플 개수
        output_dir: 출력 디렉토리
        base_dir: CCPD2019 기본 디렉토리
    """
    method_map = {
        'sqrt': 2,
        'arccos': 3,
        'solvepnp': 4
    }

    method_idx = method_map.get(method)
    if method_idx is None:
        print(f"❌ 알 수 없는 메소드: {method}")
        return

    print(f"📊 방법: {method}")
    print(f"📊 Bin 개수: {bins}")
    print(f"🖼️  Bin당 샘플 개수: {samples_per_bin}")
    print()

    # bin 범위: 0-90도, bins개로 분할
    bin_edges = np.linspace(0, 90, bins + 1)
    bin_data = defaultdict(list)

    # 데이터를 bin별로 분류
    for row in data:
        csv_name = row[0]
        filename = row[1]
        angle_val = row[method_idx]

        # 해당 bin 찾기
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= angle_val < bin_edges[i + 1]:
                bin_data[i].append((csv_name, filename, angle_val))
                break
        else:
            # 마지막 bin (90도)
            if angle_val >= bin_edges[-2]:
                bin_data[len(bin_edges) - 2].append((csv_name, filename, angle_val))

    # 출력 디렉토리 생성
    output_path = Path(output_dir) / f'{method}_bins'
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📁 출력 디렉토리: {output_path}\n")

    # bin별로 샘플 추출 및 복사
    total_copied = 0
    total_bins_with_data = 0

    for bin_idx in sorted(bin_data.keys()):
        bin_start = bin_edges[bin_idx]
        bin_end = bin_edges[bin_idx + 1]
        bin_items = bin_data[bin_idx]

        if not bin_items:
            continue

        total_bins_with_data += 1

        # bin 폴더 생성
        bin_folder = output_path / f'bin_{bin_idx:02d}_{bin_start:.1f}-{bin_end:.1f}deg'
        bin_folder.mkdir(exist_ok=True)

        # 샘플 선택 (랜덤)
        np.random.shuffle(bin_items)
        selected_samples = bin_items[:samples_per_bin]

        print(f"📂 Bin {bin_idx:02d} ({bin_start:.1f}° - {bin_end:.1f}°): {len(bin_items):,}개 중 {len(selected_samples)}개 샘플")

        copied_count = 0
        for csv_name, filename, angle_val in selected_samples:
            # 원본 이미지 경로 찾기
            src_path = find_image_path(csv_name, filename, base_dir)

            if src_path is None:
                print(f"   ⚠️  이미지 파일 없음: {csv_name}/{filename}")
                continue

            # 목적지 파일명 (각도 정보 포함)
            dst_filename = f"{angle_val:.2f}deg_{csv_name}_{src_path.name}"
            dst_path = bin_folder / dst_filename

            # 복사
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"   ❌ 복사 실패: {src_path} -> {e}")

        total_copied += copied_count
        print(f"   ✅ {copied_count}개 이미지 복사 완료\n")

    print("=" * 80)
    print(f"✅ 전체 요약")
    print(f"   총 bin 수: {bins}개")
    print(f"   데이터가 있는 bin: {total_bins_with_data}개")
    print(f"   복사된 이미지: {total_copied}개")
    print(f"   저장 위치: {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='히스토그램 bin별 샘플 이미지 추출'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../',
        help='CSV 파일이 있는 디렉토리 (기본값: ../)'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/workspace/DB/01_LicensePlate/CCPD2019',
        help='CCPD2019 이미지 기본 디렉토리'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['sqrt', 'arccos', 'solvepnp'],
        default='arccos',
        help='각도 계산 방법 (기본값: arccos)'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=30,
        help='bin 개수 (기본값: 30)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='bin당 샘플 개수 (기본값: 10)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='bin_samples',
        help='출력 디렉토리 (기본값: bin_samples)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🖼️  히스토그램 bin별 샘플 이미지 추출")
    print("=" * 80)
    print(f"📁 CSV 디렉토리: {args.data_dir}")
    print(f"📁 이미지 디렉토리: {args.base_dir}")
    print(f"📊 방법: {args.method}")
    print(f"📊 Bin 개수: {args.bins}")
    print(f"🖼️  Bin당 샘플: {args.samples}개")
    print(f"💾 출력 디렉토리: {args.output_dir}")
    print("=" * 80)
    print()

    # CSV 데이터 로드
    data = load_all_csv_files(args.data_dir)

    if data is None:
        return

    # bin별 샘플 추출
    extract_bin_samples(
        data,
        method=args.method,
        bins=args.bins,
        samples_per_bin=args.samples,
        output_dir=args.output_dir,
        base_dir=args.base_dir
    )


if __name__ == "__main__":
    main()
