"""
QBB CSV에서 P1-1, P1-2, P1-3, P1-4를 P1으로 병합
"""

import csv
from pathlib import Path

input_folder = Path("/workspace/repo/ultralytics/runs/qbb/train0902_2/inference_csv")
output_folder = Path("./output2/csv/QBB_P1merged")
output_folder.mkdir(parents=True, exist_ok=True)

csv_files = sorted([f for f in input_folder.iterdir() if f.suffix == '.csv'])

print(f"총 {len(csv_files)}개 CSV 파일 처리 시작...")

for idx, csv_file in enumerate(csv_files):
    # CSV 읽기
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # P1 계열 병합
    merged_rows = []
    for row in rows:
        if len(row) != 10:
            continue

        plate_class = row[0]

        # P1-1, P1-2, P1-3, P1-4를 P1으로 변경
        if plate_class.startswith('P1-'):
            plate_class = 'P1'

        merged_rows.append([plate_class] + row[1:])

    # 저장
    output_file = output_folder / csv_file.name
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in merged_rows:
            writer.writerow(row)

    if (idx + 1) % 100 == 0:
        print(f"  진행률: {idx + 1}/{len(csv_files)}")

print(f"완료! 출력 폴더: {output_folder}")
