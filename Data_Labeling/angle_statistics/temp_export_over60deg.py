#!/usr/bin/env python3
"""
60도 이상 샘플의 jpg와 json 파일을 복사하는 임시 스크립트
데이터셋별로 분리하여 저장
"""

import csv
import shutil
from pathlib import Path


def read_csv_data(csv_path):
    """CSV 파일 읽기"""
    data_rows = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0]:
                continue
            if row[0].startswith('#'):
                continue
            if row[0] == 'filename':
                continue

            try:
                filename = row[0]
                solvepnp_val = float(row[8])
                data_rows.append((filename, solvepnp_val))
            except (ValueError, IndexError):
                continue

    return data_rows


def find_files(csv_name, filename):
    """jpg와 json 파일 경로 찾기"""
    # WebPlatemania
    if csv_name.startswith('WebPlatemania_'):
        plate_type = csv_name.replace('WebPlatemania_', '')
        base_dir = Path('/workspace/DB/01_LicensePlate/55_WebPlatemania_jpg_json_20250407')
        folder = base_dir / f'GoodMatches_{plate_type}'

        jpg_path = folder / filename
        json_path = folder / filename.replace('.jpg', '.json')

        if jpg_path.exists() and json_path.exists():
            return jpg_path, json_path, 'webplatemania'

    # CCPD2019
    else:
        base_dir = Path('/workspace/DB/01_LicensePlate/CCPD2019')
        folder = base_dir / csv_name / 'GoodMatches_H22'

        jpg_path = folder / filename
        json_path = folder / filename.replace('.jpg', '.json')

        if jpg_path.exists() and json_path.exists():
            return jpg_path, json_path, 'ccpd'

    return None, None, None


def main():
    print("=" * 80)
    print("📊 60도 이상 샘플 추출 (jpg + json) - 데이터셋별 분리")
    print("=" * 80)
    print()

    # CSV 파일 로드
    data_dir = Path('.')
    all_csvs = list(data_dir.glob('ccpd_*_GoodMatches.csv')) + list(data_dir.glob('WebPlatemania_*.csv'))

    print(f"📁 발견된 CSV 파일: {len(all_csvs)}개")
    for csv_file in sorted(all_csvs):
        print(f"   - {csv_file.name}")
    print()

    # 60도 이상 샘플 수집
    over_60_samples = []

    for csv_path in all_csvs:
        csv_name = csv_path.stem.replace('_GoodMatches', '')
        rows = read_csv_data(csv_path)

        for filename, angle in rows:
            if angle >= 60.0:
                over_60_samples.append((csv_name, filename, angle))

        count_over_60 = len([r for r in rows if r[1] >= 60.0])
        print(f"✅ {csv_name}: {count_over_60}개 (60도 이상)")

    print(f"\n📊 총 {len(over_60_samples):,}개 샘플 발견 (60도 이상)\n")

    # 출력 디렉토리 생성
    ccpd_output_dir = Path('/workspace/DB/01_LicensePlate/CCPD2019/over_60deg_samples')
    webplatemania_output_dir = Path('/workspace/DB/01_LicensePlate/55_WebPlatemania_jpg_json_20250407/over_60deg_samples')

    ccpd_output_dir.mkdir(exist_ok=True)
    webplatemania_output_dir.mkdir(exist_ok=True)

    print(f"📁 CCPD 출력: {ccpd_output_dir}")
    print(f"📁 WebPlatemania 출력: {webplatemania_output_dir}\n")
    print("🖼️  파일 복사 중...\n")

    ccpd_copied = 0
    webplatemania_copied = 0
    failed_count = 0

    for csv_name, filename, angle in sorted(over_60_samples, key=lambda x: x[2], reverse=True):
        jpg_path, json_path, dataset_type = find_files(csv_name, filename)

        if jpg_path is None or json_path is None:
            print(f"   ⚠️  파일 없음: {csv_name}/{filename}")
            failed_count += 1
            continue

        # 데이터셋별 출력 디렉토리 선택
        if dataset_type == 'ccpd':
            output_dir = ccpd_output_dir
        else:
            output_dir = webplatemania_output_dir

        # 파일명: 각도_데이터셋_원본파일명
        dst_jpg = output_dir / f"{angle:.2f}deg_{csv_name}_{jpg_path.name}"
        dst_json = output_dir / f"{angle:.2f}deg_{csv_name}_{json_path.name}"

        try:
            shutil.copy2(jpg_path, dst_jpg)
            shutil.copy2(json_path, dst_json)
            
            if dataset_type == 'ccpd':
                ccpd_copied += 1
            else:
                webplatemania_copied += 1

            total_copied = ccpd_copied + webplatemania_copied
            if total_copied % 100 == 0:
                print(f"   ✅ {total_copied}개 복사 완료...")
        except Exception as e:
            print(f"   ❌ 복사 실패: {jpg_path.name} -> {e}")
            failed_count += 1

    print()
    print("=" * 80)
    print(f"✅ 완료")
    print(f"   CCPD: {ccpd_copied}개 (jpg + json 쌍)")
    print(f"   WebPlatemania: {webplatemania_copied}개 (jpg + json 쌍)")
    print(f"   실패: {failed_count}개")
    print(f"   총 파일 수: {(ccpd_copied + webplatemania_copied) * 2}개 (jpg + json)")
    print()
    print(f"📁 CCPD 저장 위치: {ccpd_output_dir}")
    print(f"📁 WebPlatemania 저장 위치: {webplatemania_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
