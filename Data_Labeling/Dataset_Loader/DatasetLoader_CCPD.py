# 기존 CCPD 문자표를 Utils로 이동
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
             "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "?"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', '?']
ads =       ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '?']

class DatasetLoader_CCPD:
    def __init__(self, base_path, subset_type='base'):
        """기존 CCPD_Loader 초기화 로직 + WebCrawl 스타일 통합"""
        # 기존 CCPD_Loader.__init__ 로직 복사
        # + self.list_jpg, self.list_json 스타일로 변경
        self.base_path = base_path
        self.subset_type = subset_type
        self.m_index = -1

    def parse_ccpd_filename(self, img_filename):
        """
        기존 read_gt() 함수를 parse_json() 스타일로 변경
        Returns: plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom
        """
        try:
            # 기존 코드 그대로 사용
            tokens = img_filename[:-4].split('-')

            # 4개 꼭지점 좌표 추출 (기존 로직 그대로)
            xy1, xy2, xy3, xy4 = (a.split('&') for a in tokens[3].split('_'))
            xy1 = list(map(float, xy1))  # float으로 변경 (WebCrawl 호환)
            xy2 = list(map(float, xy2))
            xy3 = list(map(float, xy3))
            xy4 = list(map(float, xy4))

            # Bounding box 계산 (기존 로직 활용)
            bb_lt, bb_rb = (a.split('&') for a in tokens[2].split('_'))
            bb_lt = list(map(int, bb_lt))
            bb_rb = list(map(int, bb_rb))
            left, top = bb_lt
            right, bottom = bb_rb

            # 번호판 문자 변환 (기존 로직 그대로)
            label = tokens[4].split('_')
            label[0] = provinces[int(label[0])]
            label[1] = alphabets[int(label[1])]
            label[2:] = [ads[int(l)] for l in label[2:]]
            plate_number = ''.join(label)

            plate_type = 'CHN'  # 중국 번호판 타입

            return plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom

        except Exception as e:
            print(f"Error parsing {img_filename}: {str(e)}")
            return 'CHN', '', [], [], [], [], 0, 0, 0, 0

    def create_labelme_json(self, img_filename, output_dir):
        """LabelMe JSON 파일 생성"""
        # parse_ccpd_filename()에서 정보 추출
        # JSON 구조를 WebCrawl 형식에 맞게 생성