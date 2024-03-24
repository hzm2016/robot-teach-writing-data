import enum
import logging
import os
import torch
import torchvision
import cv2
import imutils

import torchvision.transforms as transforms
import numpy as np

from tools import skeletonize, stroke2img, svg2img
from tqdm import tqdm
from core.learner import Learner
from core.controller import Controller
from core.utils import load_class
from core.imgprocessor import Postprocessor
from sklearn.metrics.pairwise import pairwise_distances   


class Executor(object):
    """Class that carries out teaching process

    Args:
        object ([type]): [description]
    """

    def __init__(self, args) -> None:

        logging.info('Initialize Runner')
        self.cuda = args.get('CUDA', False)
        self.feedback = args.get('FEEDBACK')
        self.save_traj = args.get('SAVE_TRAJ', False)

        self.__init_parameters(args)
        self.__parse_feedback()

    def __parse_feedback(self,):

        if self.feedback is None:
            self.with_feedback = False
        else:
            self.with_feedback = self.feedback.get('WITH_FEEDBACK')

        self.postprocessor = Postprocessor(
            self.feedback.get('POST_PRORCESS'))

        self.learner = Learner()
        self.controller = Controller(Postprocessor(
            self.feedback.get('POST_PRORCESS')))

    def __init_parameters(self, args):

        self.generation_only = args.get('GENERATIION_ONLY', False)
        self.font_type = args.get('TTF_FILE')
        self.font_size = args.get('FONT_SIZE', 128)
        self.stylelization = args.get('STYLELIZATION',False)
        assert self.font_type is not None, 'Please provide a font file'

        if args.get('PRE_PROCESS', None):
            assert args.get('PRE_PROCESS').upper(
            ) == 'DEFAULT', '{} preprocess is not supported'.format(args.get('PRE_PROCESS'))
            pre_process = [transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))]
        else:
            pre_process = [transforms.ToTensor()
                           ]
        self.pre_process = transforms.Compose(pre_process)

        assert 'GRAPHICS_TXT' in args.keys(), 'Please provide a graphic list'
        graphic_list = args.get('GRAPHICS_TXT')
        input_lines = open(graphic_list, 'r').readlines()
        self.char_list = {}
        logging.info('Loading character list')

        for line in tqdm(input_lines):
            char_info = eval(line)
            char = char_info['character']
            self.char_list[char] = char_info

    def interact(self, traj, score=None):
        """TO DO: interaction part
        """
        if self.with_feedback:
            output_img = self.__capture_image()
            output_img = self.postprocessor.process(output_img)
            # cv2.imshow('',np.array(output_img))
            # cv2.waitKey(0)
            # self.learner.score = self.get_score(output_img)
        return output_img

    def get_score(self, image):

        return self.dis(self.pre_process(image))

    def __capture_image(self, ):
        """ Capture image with post process in order for discrinmintor to score
        """
        return cv2.imread('./example/example_feedback.png')
        # raise NotImplementedError

    def sample_character(self, stroke, written_character=None):
        """ For future development, decompose one character into several characters
        """
        if written_character is None:
            return stroke2img(self.font_type, stroke, self.font_size)
        else:
            source_image = np.array(stroke2img(
                self.font_type, stroke, self.font_size))
            return self.__generate_written_traj(written_character, source_image)

    def __reset_learner(self,):
        """ Reset learner model
        """
        self.learner.reset()

    def __quit(self):
        """ Quit all the processes
        """
        logging.info('Quittiing')

    def __save_stroke_traj(self, stroke, traj, savepath='./'):

        font_name = self.font_type.split('/')[-1].split('.')[0]
        filename = os.path.join(savepath, str(stroke)+'_'+font_name) + '.txt'
        with open(filename, 'w+') as file_stream:
            np.savetxt(file_stream,  traj[0])

        return filename

    def __filter(self, points, mask, head, tail, condition=0):
        '''
        condition explaination:
        0: head, head
        1: head, tail
        2: tail, head
        3: tail, tail   
        '''
        if head != tail:
            reverse = True

        if condition == 0:
            pos_i = (1, 0)
            pos_j = (1, 0)

        elif condition == 1:
            pos_i = (1, 0)
            pos_j = (-2, -1)

        elif condition == 2:

            pos_i = (-2, -1)
            pos_j = (1, 0)

        elif condition == 3:

            pos_i = (-2, -1)
            pos_j = (-2, -1)

        else:
            raise NotImplementedError

        for idx, i in enumerate(head):
            if mask[idx] == 0:
                continue
            # compare head with head
            for idy, j in enumerate(tail[idx+1:]):

                real_index = idx + idy + 1
                if mask[real_index] == 0:
                    continue
                if i == j:
                    x_trend_0 = points[idx][pos_i[0]][0] - \
                        points[idx][pos_i[1]][0]
                    y_trend_0 = points[idx][pos_i[0]][1] - \
                        points[idx][pos_i[1]][1]

                    x_trend_1 = points[real_index][pos_j[0]
                                                   ][0] - points[real_index][pos_j[1]][0]
                    y_trend_1 = points[real_index][pos_j[0]
                                                   ][1] - points[real_index][pos_j[1]][1]

                    if (x_trend_0 * x_trend_1 >= 0) and (y_trend_0 * y_trend_1 >= 0):

                        if len(points[idx]) > len(points[real_index]):
                            mask[real_index] = 0
                        else:
                            mask[idx] = 0

        return mask

    def __merge(self, a, b, idx, offset=1):

        if idx == 0:
            result = b + a[offset:]
        elif idx == 1:
            result = a + b[offset:]
        elif idx == 2:
            result = list(reversed(a)) + b[offset:]
        elif idx == 3:
            result = a[:-offset] + list(reversed(b))

        return result

    def fps(self, points,frac):
        P = np.array(points[0])
        num_points = int(P.shape[0] * frac)
        D = pairwise_distances(P, metric='euclidean')
        N = D.shape[0]
        # By default, takes the first point in the list to be the
        # first point in the permutation, but could be random
        perm = np.zeros(N, dtype=np.int64)
        lambdas = np.zeros(N)
        # import random
        # random.seed(0)
        # stpt = random.randint(0,N-1)
        ds = D[0, :]
        for i in range(1, N):
            idx = np.argmax(ds)
            perm[i] = idx
            lambdas[i] = ds[idx]
            ds = np.minimum(ds, D[idx, :])
        return P[perm[:num_points]].tolist()


    def merge_keypoints(self, points):

        # Find Common Element
        mask = [1] * len(points)
        head = [i[0] for i in points]
        tail = [i[-1] for i in points]

        for idx, i in enumerate(points):
            if len(i) < 3:
                mask[idx] = 0

        mask = self.__filter(points, mask, head, head, 0)
        mask = self.__filter(points, mask, head, tail, 1)
        mask = self.__filter(points, mask, tail, head, 2)
        mask = self.__filter(points, mask, tail, tail, 3)

        filtered_points = np.array(points,dtype=tuple)[list(map(bool, mask))].tolist()

        filtered_points = sorted(filtered_points, key=len)

        if len(filtered_points) == 1:
            return filtered_points[0]

        # Only take the two longest list of stroke
        if len(filtered_points) < 2:
            return filtered_points[0]

        filtered_points = filtered_points[-2:]

        a = filtered_points[0]
        b = filtered_points[1]

        if a[0] == b[-1]:
            return self.__merge(a, b, 0)
        elif a[-1] == b[0]:
            return self.__merge(a, b, 1)
        elif a[0] == b[0]:
            return self.__merge(a, b, 2)
        elif a[-1] == b[-1]:
            return self.__merge(a, b, 3)
        else:
            dist_h2h = self.__dist(a[0], b[0])
            dist_t2t = self.__dist(a[-1], b[-1])
            dist_h2t = self.__dist(a[0], b[-1])
            dist_t2h = self.__dist(a[-1], b[0])

            dists = [dist_h2t,dist_t2h,dist_h2h,dist_t2t]

            min_dist = dists.index(min(dists))  
            return self.__merge(a, b, min_dist, 0)         

    def __dist(self, a, b):

        return np.linalg.norm(np.array(a)-np.array(b))

    def stylelize(self, img):
        
        if self.cuda:
            source_image = self.pre_process(img).unsqueeze(0).cuda()
        else:
            source_image = self.pre_process(img).unsqueeze(0)

        styled_image = self.gan(source_image)

        styled_image = 0.5*(styled_image.data + 1.0)
        grid = torchvision.utils.make_grid(styled_image)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer

        styled_image = grid.mul(255).add_(0.5).clamp_(
            0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        styled_image = cv2.cvtColor(
            styled_image, cv2.COLOR_BGR2GRAY)

        return styled_image

    # Function to calculate curvature
    def curvature(self, x, y):
        # First derivatives
        # import pdb; pdb.set_trace()
        dx = np.gradient(x)
        dy = np.gradient(y)

        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature calculation
        curvature = np.abs(ddx * dy - dx * ddy) / (dx * dx + dy * dy)**1.5
        return curvature

    def resample(self, points, n):

        n = min(n, len(points))

        total_length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        distance_between_keypoints = total_length / (n - 1)

        keypoints = [points[0]] 
        current_distance = 0

        for i in range(1, len(points)):
            point1 = points[i - 1]
            point2 = points[i]

            distance = np.sqrt(np.sum((point2 - point1)**2))

            if current_distance + distance >= distance_between_keypoints:
                t = (distance_between_keypoints - current_distance) / distance
                new_point = (1 - t) * point1 + t * point2
                keypoints.append(new_point)
                points.insert(i, new_point)
                current_distance = 0
            else:
                current_distance += distance
  
        while len(keypoints) < n:
            keypoints.append(points[-1])

        return np.array(keypoints)

    def __keypoint_extraction(self, points, n=5):  
        """ Extract Keypoints from image including turning points
        """

        resampled_points = self.resample(points, 30)   #### 
        resampled_points = resampled_points[0]
        curvature = self.curvature(resampled_points[:, 0], resampled_points[:, 1])
        keypoints = np.argsort(curvature)[-n:]   
        keypoints = resampled_points[keypoints]   

        return keypoints

    def pipeline(self, save_path='', character='戈'):   
        """ Full pipeline
        Obtain target character -> generate target character's trajectory -> Interact with learner -> Get learner output
        """

        while True:  
            # character = '戈'  

            # input('Please provide a character you want to learn: ')
            written_image = None 

            if len(character) > 1:
                logging.warning('Please input once character only')
                continue 

            if character == ' ':
                break

            if character in self.char_list:
                logging.info('We find the character for you')
            else:
                logging.warning(
                    'Sorry, the character is not supported, please try another one')
                break

            char_info = self.char_list[character]
            strokes = char_info['strokes']

            # print('char_info', char_info)
            # for ch in tqdm(self.char_list):
            #     strokes = self.char_list[ch]['strokes']

            img_list = []
            # if self.stylelization:
            #     for stroke in strokes:
            #         img_list.append(self.stylelize(svg2img(stroke)))
            # else:
            for stroke in strokes:
                img_list.append(svg2img(stroke))


            while not self.learner.satisfied:

                # character_img = self.sample_character(character, written_image)
                # character_img = np.array(character_img)
                
                cnt = 0
                if self.save_traj:
                    img_ske_list = []
                    traj_list = []

                    for idx, img in enumerate(img_list):
                        traj, traj_img = skeletonize(~img)   
                        img_ske_list.append(traj_img)  
                        if len(traj) > 1:
                            # for k in traj:
                            #     print(k)
                            traj = [self.merge_keypoints(traj)]

                        # num_points_frac is the fraction of points needs to be sampled
                        starting_point = traj[0][0]
                        ending_point = traj[0][-1]
                        # num_points_frac = 0.2
                        # import pdb;pdb.set_trace()
                        # traj = self.fps(traj, num_points_frac)
                        key_points = self.__keypoint_extraction(traj)
                        # import pdb;pdb.set_trace()
                        traj_list.append(traj)  

                    for idx, traj in enumerate(traj_list):
                        save_traj_name = self.__save_stroke_traj(character+'_'+ str(idx), traj, savepath=save_path)  
                        print("save_traj_name :", save_traj_name)  
                        cv2.imwrite(save_traj_name.replace('txt', 'png'), img_ske_list[idx])

                    logging.info('{} traj stored'.format(character))

                if self.generation_only:
                    continue

        self.__quit()

        # if written_image is not None:
        #     cv2.imshow('',stroke_img)
        #     cv2.waitKey(0)
        #     cv2.imwrite('./new.png', stroke_img)
        #     cv2.imshow('',traj_img)
        #     cv2.waitKey(0)
        #     cv2.imwrite('./new_traj.png', traj_img)
