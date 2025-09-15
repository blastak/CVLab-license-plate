import csv
import json
import os


class DatasetLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.list_jpg = [f for f in os.listdir(self.base_path) if f.endswith('.jpg')]
        self.list_csv = [f for f in os.listdir(self.base_path) if f.endswith('.csv')]
        self.list_json = [f for f in os.listdir(self.base_path) if f.endswith('.json')]
        self.__valid = len(self.list_csv) == len(self.list_json) != 0

    @property
    def valid(self):
        return self.__valid

    def __len__(self):
        """데이터셋 크기 - CSV 기준"""
        return len(self.list_csv)

    def __getitem__(self, idx):
        """
        인덱스로 접근하여 (base_name, predictions, ground_truth) 반환
        """
        csv_file = self.list_csv[idx]
        json_file = self.list_json[idx]
        base_name = os.path.splitext(csv_file)[0]

        pred = self.parse_detect(csv_file)
        gt = self.parse_label(json_file)

        return base_name, pred, gt

    def parse_detect(self, csv_path):
        """CSV 파일을 파싱하여 모든 예측 번호판 정보 리스트로 반환"""
        filename = os.path.join(self.base_path, csv_path)
        detections = []  # 예측 리스트: [(plate_type, [xy1, xy2, xy3, xy4], conf), ...]

        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) != 10:
                    continue  # 잘못된 형식은 무시
                plate_type = row[0]  # 예: 'P0'
                xy1 = [float(row[1]), float(row[2])]
                xy2 = [float(row[3]), float(row[4])]
                xy3 = [float(row[5]), float(row[6])]
                xy4 = [float(row[7]), float(row[8])]
                conf = float(row[9])
                detections.append((plate_type, [xy1, xy2, xy3, xy4], conf))
        return detections

    def parse_label(self, json_path):
        """
        JSON 파일을 파싱하여 모든 GT 번호판 정보를 리스트로 반환
        반환 형식: [(plate_type, points), ...]
        """
        filename = os.path.join(self.base_path, json_path)
        results = []

        with open(filename, 'r', encoding='UTF-8') as file:
            data = json.load(file)

        for shape in data['shapes']:
            full_label = shape.get('label', '')
            plate_type = full_label.split('_')[0] if '_' in full_label else full_label
            points = shape.get('points', [])

            if len(points) == 4:  # 꼭짓점이 정확히 4개일 때만 사용
                results.append((plate_type, points))

        return results


if __name__ == '__main__':
    prefix_path = r"D:\Dataset\LicensePlate\test\test_IWPOD_\GoodMatches_P4"
    loader = DatasetLoader(base_path=prefix_path)

    if loader.valid:
        # 기존 방식 (변경 전)
        # for jpg_file in loader.list_jpg:
        #     base_name = os.path.splitext(jpg_file)[0]
        #     csv_file = f"{base_name}.csv"
        #     json_file = f"{base_name}.json"
        #
        #     if csv_file in loader.list_csv and json_file in loader.list_json:
        #         pred = loader.parse_detect(csv_file)
        #         gt = loader.parse_label(json_file)

        # 새로운 방식 (변경 후)
        for base_name, pred, gt in loader:  # 간단한 iteration
            if pred and gt:  # 둘 다 있는 경우만 처리
                for plate_type_pred, pred_coords, conf in pred:
                    for plate_type_gt, gt_coords in gt:
                        print(f"[{base_name}] 예측: {plate_type_pred} {pred_coords} conf={conf}")
                        print(f"[{base_name}] 정답: {plate_type_gt} {gt_coords}")