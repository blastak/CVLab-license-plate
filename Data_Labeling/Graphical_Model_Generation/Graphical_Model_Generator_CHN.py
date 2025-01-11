import os
import random

import cv2

from Utils import bd_chn2num


class Graphical_Model_Generator_CHN:
    def __init__(self, base_path):
        self.base_path = base_path
        self.plate_wh = (440, 140)
        self.char_xywh = [
            (15.5, 25, 45, 90),
            (72.5, 25, 45, 90),
            (151.5, 25, 45, 90),
            (208.5, 25, 45, 90),
            (265.5, 25, 45, 90),
            (322.5, 25, 45, 90),
            (379.5, 25, 45, 90),
        ]

        # scaling
        self.plate_wh2 = tuple(x * 2 for x in self.plate_wh)
        self.char_xywh2 = [tuple(x * 2 for x in xywh) for xywh in self.char_xywh]

    def absoluteFilePaths(self, directory):
        retval = []
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                retval.append(os.path.abspath(os.path.join(dirpath, f)))
        return retval

    def random_file_in_dir(self, directory):
        t = self.absoluteFilePaths(directory)
        return random.choice(t)

    def overlay(self, bg, fg, xy):
        fh, fw = fg.shape[:2]
        x, y = map(int, xy)

        for c in range(3):
            bg[y:y + fh, x:x + fw, c] = cv2.add(bg[y:y + fh, x:x + fw, c], fg[:, :, c])

    def make_LP(self, demand_str):
        if len(demand_str) != 7:
            return None

        demand_str = demand_str.replace('I', '1', )
        demand_str = demand_str.replace('i', '1')
        demand_str = demand_str.replace('O', '0')
        demand_str = demand_str.replace('o', '0')

        img_template = cv2.resize(cv2.imread(self.random_file_in_dir(self.base_path + '/template/'), cv2.IMREAD_UNCHANGED), self.plate_wh2)
        # img_template[:,:,:3] = np.zeros_like(img_template[:,:,:3]) + 127 ### gray
        # img_template = cv2.resize(cv2.imread(self.absoluteFilePaths(self.base_path + '/template/')[2],cv2.IMREAD_UNCHANGED),self.plate_wh) # template3

        for i, ch in enumerate(demand_str):
            p = ''
            if not ch.isascii():  # chinese,korean,...
                p = '/chinese/' + f'/{bd_chn2num[ch]}/'
            elif ch.isdigit():  # number
                p = '/number/' + f'/{ch}/'
            else:
                p = '/alphabet/' + f'/{ch.upper()}/'
            img = cv2.resize(cv2.imread(self.random_file_in_dir(self.base_path + p)), self.char_xywh2[i][2:])
            self.overlay(img_template, img, self.char_xywh2[i][:2])

        return img_template


if __name__ == '__main__':
    generator = Graphical_Model_Generator_CHN('./BetaType/chinese_LP/')
    # img = generator.make_LP('京BYX342')
    # img = generator.make_LP('津A63060')
    # img = generator.make_LP('冀E99999')
    # img = generator.make_LP('晋H88888')
    # img = generator.make_LP('蒙T41718')
    # img = generator.make_LP('辽JL0239')
    # img = generator.make_LP('吉C68686')
    img = generator.make_LP('黑FTS521')
    cv2.imshow('img', img)
    cv2.waitKey()
