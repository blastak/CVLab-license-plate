import csv
import json
import os


class TestsetLoader:
    def __init__(self, csv_path, json_path):
        self.csv_path = csv_path
        self.json_path = json_path

        # 각각의 폴더에서 확장자별 파일 목록 가져오기
        self.list_csv = [f for f in os.listdir(self.csv_path) if f.endswith('.csv')]
        self.list_json = [f for f in os.listdir(self.json_path) if f.endswith('.json')]

        # csv와 json 개수가 동일하고 0이 아닐 때만 유효
        self.__valid = len(self.list_csv) == len(self.list_json) != 0

        self.mono_cls = False
        self.mc = None
        self.merge_P1 = False

        # 유효할 때만 체크 수행
        if self.__valid:
            self.mono_cls, self.mc = self._check_mono_cls()
            self.merge_P1 = self._check_P1()

    @property
    def valid(self):
        return self.__valid

    def parse_detect(self, csv_file):
        """CSV 파일을 파싱하여 모든 예측 번호판 정보 리스트로 반환"""
        filename = os.path.join(self.csv_path, csv_file)
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

    def parse_label(self, json_file):
        """
        JSON 파일을 파싱하여 모든 GT 번호판 정보를 리스트로 반환
        반환 형식: [(plate_type, points), ...]
        """
        filename = os.path.join(self.json_path, json_file)
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

    def _check_mono_cls(self):
        """예측 결과에 'P0' 또는 'License_Plate'만 있는 경우 단일 클래스 여부 확인"""
        mono_cls = False
        mc = None
        for csv_file in self.list_csv:
            pred = self.parse_detect(csv_file)
            for p in pred:
                if p[0] in ('P0', 'License_Plate'):
                    mono_cls = True
                    mc = p[0]
                    break
            if mono_cls:
                break
        return mono_cls, mc

    def _check_P1(self):
        """예측 결과에 P1이 있으면 GT에서 P1-*을 모두 P1로 병합하도록 판단."""
        for csv_file in self.list_csv:
            pred = self.parse_detect(csv_file)
            if any(p[0] == 'P1' for p in pred):
                return True
        return False


if __name__ == '__main__':
    csv_dir = r"VIN_csv"
    json_dir = r"testset"

    loader = TestsetLoader(csv_path=csv_dir, json_path=json_dir)

    if loader.valid:
        print(f"모노 클래스 여부: {loader.mono_cls}, "
              f"mc={loader.mc}, "
              f"P1 병합여부: {loader.merge_P1}")
        for csv_file in loader.list_csv:
            base_name = os.path.splitext(csv_file)[0]
            json_file = f"{base_name}.json"

            if json_file in loader.list_json:
                pred = loader.parse_detect(csv_file)
                gt = loader.parse_label(json_file)

                for plate_type_pred, pred_coords, conf in pred:
                    for plate_type_gt, gt_coords in gt:
                        print(f"[{base_name}] 예측: {plate_type_pred} {pred_coords} conf={conf}")
                        print(f"[{base_name}] 정답: {plate_type_gt} {gt_coords}")
