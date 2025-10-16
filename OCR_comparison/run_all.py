"""
통합 실행 스크립트
전체 파이프라인을 한 번에 실행: CSV 생성 → Frontalization → OCR 평가
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """
    쉘 명령어 실행 및 출력
    """
    print(f"\n{'='*80}")
    print(f"[실행 중] {description}")
    print(f"{'='*80}")
    print(f"명령어: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n[오류] {description} 실패")
        sys.exit(1)

    print(f"\n[완료] {description}")


def main():
    parser = argparse.ArgumentParser(description='번호판 검출기 OCR 성능 비교 전체 파이프라인 실행')

    # 필수 인자
    parser.add_argument('--image_folder', type=str, required=True, help='입력 이미지 폴더 경로')
    parser.add_argument('--json_folder', type=str, required=True, help='GT JSON 폴더 경로')

    # 모델 경로
    parser.add_argument('--vinlpd_model', type=str, default='../LP_Detection/VIN_LPD/weight',
                        help='VIN_LPD 모델 경로')
    parser.add_argument('--iwpod_model', type=str, default='../LP_Detection/IWPOD_tf/weights/iwpod_net',
                        help='IWPOD-tf 모델 경로')
    parser.add_argument('--ocr_model', type=str, default='../LP_Recognition/VIN_OCR/weight',
                        help='VIN_OCR 모델 경로')

    # 출력 경로
    parser.add_argument('--output_dir', type=str, default='./output', help='출력 디렉토리')

    # 실행 옵션
    parser.add_argument('--detectors', type=str, nargs='+', default=['VIN_LPD', 'IWPOD_tf'],
                        help='실행할 검출기 목록 (VIN_LPD, IWPOD_tf)')
    parser.add_argument('--skip_csv', action='store_true', help='CSV 생성 단계 스킵')
    parser.add_argument('--skip_frontalization', action='store_true', help='Frontalization 단계 스킵')

    args = parser.parse_args()

    # 경로 설정
    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = output_dir / 'csv'
    frontalized_dir = output_dir / 'frontalized'
    results_csv = output_dir / 'ocr_results.csv'

    print(f"\n{'='*80}")
    print(f"번호판 검출기 OCR 성능 비교 파이프라인")
    print(f"{'='*80}")
    print(f"입력 이미지: {args.image_folder}")
    print(f"GT JSON: {args.json_folder}")
    print(f"검출기: {', '.join(args.detectors)}")
    print(f"출력 디렉토리: {output_dir}")

    # Step 1: CSV 생성
    if not args.skip_csv:
        for detector in args.detectors:
            if detector == 'VIN_LPD':
                model_path = args.vinlpd_model
            elif detector == 'IWPOD_tf':
                model_path = args.iwpod_model
            else:
                print(f"경고: 알 수 없는 검출기 {detector}, 스킵합니다.")
                continue

            csv_output = csv_dir / detector
            cmd = [
                sys.executable,
                str(script_dir / 'step1_generate_csv.py'),
                '--image_folder', args.image_folder,
                '--output_csv_folder', str(csv_output),
                '--detector', detector,
                '--model_path', model_path
            ]
            run_command(cmd, f"Step 1: {detector} CSV 생성")

    # Step 2: Frontalization
    if not args.skip_frontalization:
        for detector in args.detectors:
            csv_input = csv_dir / detector
            if not csv_input.exists():
                print(f"경고: {csv_input} 폴더가 없습니다. {detector} frontalization을 스킵합니다.")
                continue

            cmd = [
                sys.executable,
                str(script_dir / 'step2_frontalization.py'),
                '--image_folder', args.image_folder,
                '--csv_folder', str(csv_input),
                '--output_folder', str(frontalized_dir),
                '--detector_name', detector
            ]
            run_command(cmd, f"Step 2: {detector} Frontalization")

    # Step 3: OCR 평가
    cmd = [
        sys.executable,
        str(script_dir / 'step3_ocr_evaluation.py'),
        '--frontalized_folder', str(frontalized_dir),
        '--json_folder', args.json_folder,
        '--ocr_model_path', args.ocr_model,
        '--output_csv', str(results_csv),
        '--detectors'
    ] + args.detectors

    run_command(cmd, "Step 3: OCR 평가")

    print(f"\n{'='*80}")
    print(f"전체 파이프라인 완료!")
    print(f"{'='*80}")
    print(f"결과 CSV: {results_csv}")
    print(f"Frontalized 이미지: {frontalized_dir}")


if __name__ == '__main__':
    main()
