import argparse
import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from LP_Swapping.utils import create_model, create_dataset, save_log, cvt_args2str
from Utils import get_pretty_datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name')
    parser.add_argument('--net_name', type=str, required=True, choices=['Pix2pix', 'Masked_Pix2pix'], help='Absolute or relative path of input data directory for training')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['CondReal', 'CondRealMask'], help='one of class name which in datasets.py')
    parser.add_argument('--image_dir', type=str, required=True, help='Absolute or relative path of input data directory for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs (default = xxx)')
    parser.add_argument('--display_freq', type=int, default=1, help='Frequency for saving image (in epochs) ')
    parser.add_argument('--save_freq', type=int, default=50, help='Frequency for saving checkpoints (in epochs) ')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size (default = xx)')
    parser.add_argument('--gpu_ids', type=str, default='0', help='List IDs of GPU available. ex) --gpu_ids=0,1,2,3, Use -1 for CPU mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker threads for data loading')
    args = parser.parse_args()

    gpu_ids = []
    for n in args.gpu_ids.split(','):
        if int(n) >= 0:
            gpu_ids.append(int(n))

    ########## torch environment settings
    manual_seed = 189649830
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    device = torch.device('cuda:%d' % gpu_ids[0] if (torch.cuda.is_available() and len(gpu_ids) > 0) else 'cpu')
    torch.set_default_device(device)  # working on torch>2.0.0
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        torch.multiprocessing.set_start_method('spawn')

    ########## training dataset settings
    train_dataset = create_dataset(args.dataset_name, args.image_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=device), num_workers=args.workers)

    ########## model settings
    model = create_model(args.net_name, 4, 3, gpu_ids)  # 모달리티 별로 in, out 크기를 미리 설정해두고 사용하자
    # print(model)

    ########## make saving folder
    exp_name = args.exp_name
    if len(exp_name) == 0:
        exp_name = '_'.join([args.net_name, args.dataset_name])
    save_dir = os.path.join('checkpoints', exp_name)
    cnt = 1
    while True:
        try:
            os.makedirs(save_dir + '_try%03d' % cnt)
            save_dir += '_try%03d' % cnt
            break
        except:
            cnt += 1
    args.exp_name = exp_name
    args.save_dir = save_dir
    save_log(save_dir, get_pretty_datetime())
    save_log(save_dir, cvt_args2str(vars(args)))
    save_log(save_dir, '\n' + str(model) + '\n\n')

    ########## training process
    for epoch in range(1, args.epochs + 1):
        with tqdm(train_loader, unit='batch') as tq:
            for inputs in tq:
                model.input_data(inputs)
                model.learning()

                tq.set_description(f'Epoch {epoch}/{args.epochs}')
                tq.set_postfix(model.get_current_loss())
            save_log(save_dir, str(tq))

        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(save_dir, 'ckpt_epoch%06d.pth' % epoch)
            model.save_checkpoints(ckpt_path)
        if epoch % args.display_freq == 0:
            image_path = os.path.join(save_dir, 'image_epoch%06d.jpg' % epoch)
            model.save_generated_image(image_path)

    print('Finished training the model')
    print('checkpoints are saved in "%s"' % save_dir)
    save_log(save_dir, get_pretty_datetime())
