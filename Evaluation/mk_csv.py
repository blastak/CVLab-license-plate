from pathlib import Path

from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import load_model_tf, pred_to_csv
from LP_Detection.IWPOD_torch.iwpod_plate_detection_torch import load_model_torch, IWPODtorch_to_csv
from LP_Detection.VIN_LPD.VinLPD import load_model_VinLPD, VIN_to_csv


def mk_csv(prefix_path, mod=1):  # detect 결과 -> csv 파일 생성 함수
    if mod == 1:  # 'VIN'
        d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')
        VIN_to_csv(prefix_path, d_net)
    elif mod == 2:  # 'IWPODtf'
        iwpod_tf = load_model_tf('../LP_Detection/IWPOD_tf/weights/iwpod_net')
        pred_to_csv(prefix_path, iwpod_tf)
    else:  # 'IWPODtorch'
        mymodel = load_model_torch('../LP_Detection/IWPOD_torch/src/weights/iwpodnet_retrained_epoch10000.pth')
        IWPODtorch_to_csv(prefix_path, mymodel)


if __name__ == '__main__':
    prefix_path = Path("testset")
    mk_csv(prefix_path, mod=3)  # mod : 1.VIN, 2.IWPODtf, 3.IWPODtorch
