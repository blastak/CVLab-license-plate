#!/usr/bin/env python3
"""
CCPD2019 전체 데이터셋 각도 분포 히스토그램 생성 스크립트

Data_Labeling 디렉토리의 모든 CSV 파일을 통합하여
각 각도 계산 방법별 히스토그램을 생성합니다.

생성되는 히스토그램:
1. sqrt_method: sqrt(x² + y² + z²)
2. arccos_method: arccos(cos(x) * cos(y) * cos(z))
3. solvepnp_normal_method: 번호판 법선과 카메라 광축 사이 각도

사용 예시:
    python plot_angle_histograms.py
    python plot_angle_histograms.py --bins 50 --output_dir histograms
"""

import argparse
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False


def read_csv_data(csv_path):
    """
    CSV 파일 읽기 (주석 제외)

    Returns:
        list: 데이터 행 리스트
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
            data_rows.append(row)

    return data_rows


def load_all_csv_files(data_dir):
    """
    디렉토리의 모든 CSV 파일 로드

    Returns:
        dict: {
            'sqrt_method': [],
            'arccos_method': [],
            'solvepnp_normal_method': []
        }
    """
    csv_files = sorted(Path(data_dir).glob('*.csv'))

    if not csv_files:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {data_dir}")
        return None

    print(f"📁 발견된 CSV 파일: {len(csv_files)}개")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")
    print()

    # 데이터 저장용 딕셔너리
    all_data = {
        'sqrt_method': [],
        'arccos_method': [],
        'solvepnp_normal_method': []
    }

    total_rows = 0

    for csv_path in csv_files:
        print(f"📄 읽는 중: {csv_path.name}")
        rows = read_csv_data(csv_path)

        for row in rows:
            # CSV 형식: filename, plate_type, dimensions, x_deg, y_deg, z_deg,
            #           sqrt_method, arccos_method, solvepnp_normal_method
            try:
                sqrt_val = float(row[6])
                arccos_val = float(row[7])
                solvepnp_val = float(row[8])

                all_data['sqrt_method'].append(sqrt_val)
                all_data['arccos_method'].append(arccos_val)
                all_data['solvepnp_normal_method'].append(solvepnp_val)
            except (ValueError, IndexError) as e:
                # 잘못된 행 건너뛰기
                continue

        total_rows += len(rows)
        print(f"   ✅ {len(rows):,}개 행 로드")

    print(f"\n📊 총 {total_rows:,}개 데이터 로드 완료\n")

    return all_data


def plot_histograms(data, bins=30, output_dir='.', file_format='pdf'):
    """
    3가지 method별 히스토그램 생성

    Args:
        data: 각도 데이터 딕셔너리
        bins: 히스토그램 bin 개수
        output_dir: 출력 디렉토리
        file_format: 출력 파일 형식 (pdf, png, svg)
    """
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    methods = [
        ('sqrt_method', 'sqrt(x² + y² + z²)', 'royalblue'),
        ('arccos_method', 'arccos(cos(x) * cos(y) * cos(z))', 'darkorange'),
        ('solvepnp_normal_method', 'Normal Vector vs Camera Axis', 'green')
    ]

    # 각 method별 개별 히스토그램
    for method_key, method_name, color in methods:
        angles = data[method_key]

        if not angles:
            print(f"⚠️  {method_key} 데이터가 없습니다.")
            continue

        # 통계 계산
        mean_val = np.mean(angles)
        median_val = np.median(angles)
        std_val = np.std(angles)
        min_val = np.min(angles)
        max_val = np.max(angles)

        print(f"📊 {method_key} 통계:")
        print(f"   데이터 수: {len(angles):,}개")
        print(f"   평균: {mean_val:.2f}°")
        print(f"   중앙값: {median_val:.2f}°")
        print(f"   표준편차: {std_val:.2f}°")
        print(f"   범위: {min_val:.2f}° ~ {max_val:.2f}°")
        print()

        # 히스토그램 그리기
        fig, ax = plt.subplots(figsize=(12, 6))

        n, bins_edges, patches = ax.hist(
            angles,
            bins=bins,
            range=(0, 90),
            color=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )

        # 통계선 추가
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}°')
        ax.axvline(median_val, color='purple', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}°')

        # 라벨 및 제목
        ax.set_xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Angle Distribution: {method_name}\n(N={len(angles):,}, Mean={mean_val:.2f}°, Std={std_val:.2f}°)',
                     fontsize=14, fontweight='bold')
        ax.set_xlim(0, 90)  # x축 범위 0-90도로 고정
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 저장
        output_file = output_path / f'{method_key}_histogram.{file_format}'
        plt.tight_layout()
        if file_format == 'pdf':
            plt.savefig(output_file, format='pdf', bbox_inches='tight')
        else:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 저장 완료: {output_file}\n")

    # 3개 method 통합 비교 히스토그램
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for idx, (method_key, method_name, color) in enumerate(methods):
        angles = data[method_key]

        if not angles:
            continue

        mean_val = np.mean(angles)
        median_val = np.median(angles)

        axes[idx].hist(
            angles,
            bins=bins,
            range=(0, 90),
            color=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}°')
        axes[idx].axvline(median_val, color='purple', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}°')

        axes[idx].set_xlabel('Angle (degrees)', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{method_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlim(0, 90)  # x축 범위 0-90도로 고정
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('CCPD2019 License Plate Angle Distribution Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # 통합 히스토그램 저장
    combined_file = output_path / f'all_methods_comparison.{file_format}'
    if file_format == 'pdf':
        plt.savefig(combined_file, format='pdf', bbox_inches='tight')
    else:
        plt.savefig(combined_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 통합 비교 히스토그램 저장: {combined_file}")


def main():
    parser = argparse.ArgumentParser(
        description='CCPD2019 각도 분포 히스토그램 생성'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../',
        help='CSV 파일이 있는 디렉토리 (기본값: ../)'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=30,
        help='히스토그램 bin 개수 (기본값: 30)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='히스토그램 저장 디렉토리 (기본값: 현재 디렉토리)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='pdf',
        choices=['pdf', 'png', 'svg'],
        help='출력 파일 형식 (기본값: pdf)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("📊 CCPD2019 각도 분포 히스토그램 생성")
    print("=" * 80)
    print(f"📁 데이터 디렉토리: {args.data_dir}")
    print(f"📊 Bin 개수: {args.bins}")
    print(f"💾 출력 디렉토리: {args.output_dir}")
    print(f"📄 출력 형식: {args.format}")
    print("=" * 80)
    print()

    # CSV 데이터 로드
    data = load_all_csv_files(args.data_dir)

    if data is None:
        return

    # 히스토그램 생성
    plot_histograms(data, bins=args.bins, output_dir=args.output_dir, file_format=args.format)

    print()
    print("=" * 80)
    print("✅ 모든 히스토그램 생성 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
