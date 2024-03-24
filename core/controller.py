from tools import skeletonize
from core.utils import hungarian_matching
import numpy as np
import cv2
import argparse
import time
import glob
import os

# task interface
from control.protocol.task_interface import TCPTask
from control.path_planning.path_generate import *
from control.path_planning.plot_path import *
# from control.vision_capture.main_functions import capture_image, show_video, record_video
import seaborn as sns
sns.set(font_scale=1.2)


def draw_points(points, canvas_size=256):

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8) + 255

    for point in points:
        cv2.circle(canvas, tuple(point*2), 2, (0, 0, 0), -1)

    return canvas


def draw_matching(points_1, points_2, matching, canvas_size=256):

    points_1 = 2 * points_1
    points_2 = 2 * points_2
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8) + 255

    for point in points_1:
        cv2.circle(canvas, tuple(point), 2, (255, 0, 0), -1)

    for point in points_2:
        cv2.circle(canvas, tuple(point), 2, (0, 255, 0), -1)

    for match in matching:
        print(tuple(points_1[match[0]]))
        cv2.line(canvas, tuple(points_1[match[0]]), tuple(
            points_2[match[0]]), (0, 0, 0))

    return canvas


class Controller(object):

    def __init__(self, args, img_processor=None, impedance_level=0) -> None:

        self.root_path = '../control/data/'
        self.show_video = True

        # initial TCP connection :::
        # self.task = TCPTask('169.254.0.99', 5005)

        self.img_processor = img_processor
        self.x_impedance_level = impedance_level
        self.y_impedance_level = impedance_level

        # impedance parameters
        self.action_dim = 2
        self.stiffness_high = np.array([10.0, 10.0])
        self.stiffness_low = np.array([0.0, 0.0])
        self.stiffness = np.zeros(self.action_dim)
        self.damping = np.zeros_like(self.stiffness)

        self.scalar_distance = False

    def guide(self,):
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def key_point_matching(self, tgt_pts, in_pts):

        matching = hungarian_matching(tgt_pts, in_pts)
        matching = np.array(matching)

        return matching

    def update_impedance(self, target, input):
        """[summary]

        Args:
            input ([image]): written image
            target ([image]): target image to be learnt
        """
        if self.img_processor is not None:
            input = self.img_processor.process(input)
        cv2.imshow("Input Image", input)
        cv2.waitKey()
        
        tgt_pts, _ = skeletonize(~target)
        in_pts, _ = skeletonize(~input)

        tgt_pts = np.array(tgt_pts)
        if len(in_pts) > 1:
            in_pts = np.array([in_pts[1]])
        else:
            in_pts = np.array(in_pts)

        tgt_pts = np.squeeze(tgt_pts, axis=0)
        in_pts = np.squeeze(in_pts, axis=0)

        # tgt_pts_vis = draw_points(in_pts)
        # cv2.imwrite('tgt_pts_vis.jpg', tgt_pts_vis)

        # tgt_pts_vis = draw_points(tgt_pts)
        # cv2.imwrite('tgt_pts_vis.jpg', tgt_pts_vis)

        # in_pts_vis = draw_points(in_pts)
        # cv2.imwrite('in_pts_vis.jpg', in_pts_vis)

        matching = self.key_point_matching(tgt_pts, in_pts)
        # print("matching :", matching)
        matching_vis = draw_matching(tgt_pts, in_pts, matching)
        cv2.imwrite('matching_vis.jpg', matching_vis)

        tgt_index = matching[:, 0]
        in_index = matching[:, 1]

        x_dis = abs(tgt_pts[tgt_index][:, 0] -
                    in_pts[in_index][:, 0])
        y_dis = abs(tgt_pts[tgt_index][:, 1] -
                    in_pts[in_index][:, 1])

        if self.scalar_distance:
            x_dis = sum(x_dis) / matching.shape[0]
            y_dis = sum(y_dis) / matching.shape[0]

        self.impedance_update_policy(x_dis, y_dis,tgt_pts[tgt_index])

        return x_dis, y_dis, tgt_pts[tgt_index]

    def impedance_update_policy(self, x_dis, y_dis, corresponding_points):
        """ Linear update based on the displacement

        Args:
            x_dis ([type]): [description]
            y_dis ([type]): [description]
        """
        self.x_impedance_level = x_dis / 128
        self.y_impedance_level = y_dis / 128

        # # send impedance params :::
        self.stiffness = [100, 100]
        self.damping = [50, 50]

        return self.stiffness, self.damping

    def get_current_path(self):

        pass

    def update_velocity(self, ):
        """ update velocity with user's input get from demonstration

            args: velocity: m/s
        """
        velocity = 0.04
        return velocity

    def interact_once(self, traj, impedance_params=[5.0, 5.0, 0.2, 0.2], velocity=0.04, mode='eval'):
        """
            interact with robot once
        """
        # check motor and encode well before experiments
        print('+' * 30, 'Hardware check', '+' * 50)
        angle_initial = self.task.wait_encoder_check()
        print("Current State (rad) :", angle_initial)

        if mode=='train':
            # check the whole path
            print('+' * 30, 'Check Path', '+' * 50)
            way_points = generate_stroke_path(traj,
                                              center_shift=np.array([0.16, -WIDTH / 2]),
                                              velocity=velocity, Ts=0.001,
                                              plot_show=False)

            print('+' * 30, 'Start Send Waypoints !', '+' * 50)
            self.task.send_way_points_request()
            self.task.send_way_points(way_points)

        # self.task.send_way_points_done()
        print('+' * 20, 'Start Send Impedance Parameters !', '+' * 30)
        self.task.send_params_request()
        print("Stiffness :", impedance_params[:2])
        print("Damping :", impedance_params[2:])
        self.task.send_params(impedance_params)

        if self.args.show_video:
            show_video()

        print('+' * 50, 'Start Move !', '+' * 50)
        
        # start_time = time.time()
        run_done = False
        index = 0
        while True:
            # video record for trail :::
            run_done = self.task.get_movement_check()
    
            if run_done:
                print('+' * 50, 'Start Capture Image !', '+' * 50)
                # print("run_done", run_done)
                written_image, _ = capture_image(
                    root_path=self.root_path + 'captured_images/', font_name='written_image_test' + '_' + str(index))
                index += 1
                run_done = False
        
        # self.task.close()
        
    def interact(self, target_img):
        written_image = None
        num_episodes = 5
        run_done = False

        for i in range(num_episodes):
            # update impedance
            if written_image is not None:
                self.update_impedance(target_img, written_image)
            params = self.stiffness + self.damping

            # update velocity
            velocity = self.update_velocity()

        return written_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runs a learning example on a registered gym environment.')
    
    parser.add_argument('--show_video',
                        default=False,
                        help='enables useful debug settings')
    
    parser.add_argument('--capture_image',
                        default=False,
                        help='enables useful debug settings')

    parser.add_argument('--generate_path',
                        default=False,
                        help='enables useful debug settings')
    
    parser.add_argument('--plot_real_path',
                        default=False,
                        help='enables useful debug settings')
     
    parser.add_argument('--record_video',
                        default=False,
                        help='enables useful debug settings')
    
    parser.add_argument('--stiff_test',
                        default=False,
                        help='enables useful debug settings')

    parser.add_argument('--folder_name',
                        default='xing',
                        help='enables useful debug settings')
    
    parser.add_argument('--font_name',
                        default='行',
                        help='enables useful debug settings')
    
    parser.add_argument('--font_type',
                        default=1,
                        type=int,
                        help='enables useful debug settings')

    parser.add_argument('--root_path',
                        default='control/data/',
                        help='enables useful debug settings')

    args = parser.parse_args()

    if args.show_video:
        show_video()
    
    if args.record_video:
        
        record_video(filepath=args.root_path + 'video/' + args.folder_name + '.svo')
     
    if args.capture_image:
        file_path = args.root_path + 'captured_images/' + args.folder_name
        if os.path.exists(file_path):
            print('%s: exist' % file_path)
        else:
            try:
                os.mkdir(file_path)
            except Exception as e:
                os.makedirs(file_path)

        word_img = capture_image(file_path=file_path, font_name=args.folder_name)
    
    if args.generate_path:
        stroke_list_file = glob.glob(args.root_path + 'font_data' + '/' + args.folder_name + '/' + args.font_name + '_*.txt')
        num_stroke = len(stroke_list_file)
        print("num_stroke :", num_stroke)
        
        traj_list = []
        for str_index in range(num_stroke):
            traj = np.loadtxt(args.root_path + 'font_data' + '/' + args.folder_name + '/' +
                                   args.font_name + '_' + str(str_index) + '_font' + str(args.font_type) + '.txt')
            traj_list.append(traj)
    
        inter_list = np.ones(len(traj_list))
        inverse_list = np.ones(len(traj_list))
    
        # ======================================
        # inverse_list[0] = False
        # inter_list[0] = 2
        # # inverse_list[1] = False
        # inter_list[1] = 2
        # inverse_list[2] = False
        # inter_list[2] = 2
        # inter_list[5] = 2
        # inverse_list[4] = False
        # inter_list[4] = 2
        # # inverse_list[5] = False
        # inter_list[5] = 2
        # =======================================
        stiffness = np.array([400, 400])
        damping = np.array([20, 20])
        generate_word_path(
            traj_list,
            stiffness,
            damping,
            inter_list,
            inverse_list,
            center_shift=np.array([0.15, -WIDTH / 2]),
            velocity=0.08,
            filter_size=13,
            plot_show=True,
            save_path=True,
            save_root=args.root_path + 'font_data',
            word_name=args.folder_name
        )

        # generate_stroke_path(traj_list[2],
        #               inter_type=1,
        #               center_shift=np.array([0.16, -WIDTH / 2]),
        #               velocity=0.04,
        #               Ts=0.001,
        #               plot_show=True,
        #               save_path=False,
        #               stroke_name=str(0)
        #               )

    if args.plot_real_path:
        stroke_list_file = glob.glob(
            args.root_path + 'font_data' + '/' + args.folder_name + '/' + 'real_angle_list*.txt')
        num_stroke = len(stroke_list_file)
        print("num_stroke :", num_stroke)
    
        word_angle_list = []
        task_point_list =[]
        torque_list = []
        period_list = []
        for str_index in range(num_stroke):
            angle_list = np.loadtxt(args.root_path + 'font_data' + '/' + args.folder_name + '/' +
                              'real_angle_list_' + str(str_index) + '.txt', delimiter=' ', skiprows=1)
            stroke_torque = np.loadtxt(args.root_path + 'font_data' + '/' + args.folder_name + '/' +
                                    'real_torque_list_' + str(str_index) + '.txt', delimiter=' ', skiprows=1)
            point_list = forward_ik_path(angle_list)
            word_angle_list.append(angle_list)
            task_point_list.append(point_list)
            torque_list.append(stroke_torque)
            period_list.append(angle_list.shape[0])

        # real_stroke_path(task_points_list=task_point_list)
        plot_torque(torque_list, period_list)

    if args.stiff_test:
        stroke_list_file = glob.glob(
            args.root_path + 'font_data' + '/' + args.folder_name + '/' + 'angle_list*.txt')

        num_stroke = len(stroke_list_file)
        print("num_stroke :", num_stroke)
    
        word_angle_list = []
        task_point_list = []
        torque_list = []
        period_list = []

        stiff_task = np.diag(np.array([300, 250]))
        damping_task = np.diag(np.array([100, 100]))
        stiff_joint = np.zeros(2)
        stiff_joint_list = []
        for str_index in range(num_stroke):
            angle_list = np.loadtxt(args.root_path + 'font_data' + '/' + args.folder_name + '/' +
                                    'angle_list_' + str(str_index) + '.txt', delimiter=' ', skiprows=1)

            period_list.append(angle_list.shape[0])

            for i in range(angle_list.shape[0]):
                stiff_joint, _ = Stiff_convert(angle_list[i, :], stiff_task, damping_task)
                stiff_joint_list.append([stiff_joint[0, 0], stiff_joint[1, 1]])

            # np.savetxt('../control/data/font_data/' + args.word_name + '/' + 'params_list_' + str(stroke_name) + '.txt',
            #            way_points, fmt='%.05f')
        
        str_index = 0
        angle_list = np.loadtxt(args.root_path + 'font_data' + '/' + args.folder_name + '/' +
                                'angle_list_' + str(str_index) + '.txt', delimiter=' ', skiprows=1)
        
        stiff_plot = np.array(stiff_joint_list)[np.newaxis, :]
        print("shape :", stiff_plot.shape)
        plot_torque(stiff_plot, period_list)

    # root_path = args.root_path + 'font_data/' + args.folder_name
    # angle_list_1 = np.loadtxt(root_path + '/angle_list_5.txt', delimiter=' ')
    # angle_list_2 = np.loadtxt(root_path + '/angle_list_6.txt', delimiter=' ')
    #
    # params_list_1 = np.loadtxt(root_path + '/params_list_5.txt', delimiter=' ')
    # params_list_2 = np.loadtxt(root_path + '/params_list_6.txt', delimiter=' ')
    #
    # angle_list, params_list = contact_two_stroke(angle_list_1, angle_list_2,
    #                    params_list_1, params_list_2, inverse=False)
    #
    # np.savetxt(root_path + '/angle_list_5.txt', angle_list.copy(), fmt='%.05f')
    # np.savetxt(root_path + '/params_list_5.txt', params_list.copy(), fmt='%.05f')
    
    # plot_real_2d_path(
    #     root_path='control/data/font_data/xing/',
    #     file_name='angle_list_',
    #     delimiter=' ',
    #     stroke_num=6,
    #     skiprows=1,
    #     # root_path='./data/font_data/' + write_name + '/',
    #     # file_name='real_angle_list_0.txt',
    #     # delimiter=' ',
    #     # skiprows=1
    # )
    
    plot_real_stroke_2d_path(
        root_path='control/data/font_data/xing/',
        file_name='angle_list_5',
        stroke_num=2,
        delimiter=' ',
        skiprows=1
    )

    # write_name = 'ren'
    # from control.server_main import load_word_path
    # word_path, word_params, real_path = load_word_path(root_path='../control/data/font_data', word_name=write_name, joint_params=np.array([45, 40, 9, 0.3]))
    # print(real_path.shape)
    #
    # plot_real_osc_2d_demo_path(
    #     real_path
    # )
    
    # from control.path_planning.plot_path import *
    # plot_real_2d_path(root_path='../control/', file_name='real_angle_list.txt')
    
    # # a = np.array([(3, 4), (7, 8)])
    # # b = np.array([(1, 2), (3, 4), (5, 6)])
    # from imgprocessor import Postprocessor
    # c = Controller(Postprocessor(
    #     {'ROTATE': 0, 'BINARIZE': 128}), img_processor=Postprocessor(
    #     {'ROTATE': 0, 'BINARIZE': 128}))
    # stroke_index = 0
    # written_stroke = cv2.imread(args.root_path + 'captured_images/' + args.folder_name + '/' + args.folder_name + '.png')
    # print("written_stroke :", written_stroke.shape)
    # sample_stroke = cv2.imread(args.root_path + 'font_data/' + args.folder_name + '/' + args.font_name + '_' + str(stroke_index) + '_font1.png',
    #                            cv2.IMREAD_GRAYSCALE)
    # print("sample_stroke :", sample_stroke.shape)
    # # cv2.waitKey(0)
    #
    # # show_video()
    #
    # x_dis, y_dis, tgt = c.update_impedance(sample_stroke, written_stroke)
    # # print(x_dis, y_dis, tgt)
    # # matching = c.key_point_matching(a, b)

    # way_points = np.loadtxt('../control/angle_list_1_1.txt', delimiter=' ')
    # N_way_points = way_points.shape[0]
    # # print("N_way_points :", N_way_points)
    # # word_path.append(way_points.copy())
    # angle_point_1 = way_points[-1, :]
    # end_point = forward_ik(angle_point_1)
    #
    # way_points_2 = np.loadtxt('../control/angle_list_1_2.txt', delimiter=' ')
    # # N_way_points = way_points_2.shape[0]
    # # print("N_way_points :", N_way_points)
    # # word_path.append(way_points.copy())
    # angle_point_2 = way_points_2[0, :]
    # start_point = forward_ik(angle_point_2)
    #
    # angle_list, N = path_planning(end_point, start_point, velocity=0.04)
    # print("angle", angle_list.shape)
    #
    # angle_list_1 = np.vstack([way_points, angle_list, way_points_2])
    # print(angle_list_1.shape)

# fig = plt.figure(figsize=(15, 4))
# plt.plot(angle_list_1[:, 0], linewidth=linewidth, label='$q_1$')
# # plt.plot(t_list[1:], angle_vel_1_list_e, linewidth=linewidth, label='$d_{q1}$')
# plt.plot(angle_list_1[:, 1], linewidth=linewidth, label='$q_2$')
# # plt.plot(t_list[1:], angle_vel_2_list_e, linewidth=linewidth, label='$d_{q2}$')
# plt.show()
# np.savetxt('../control/angle_list_1.txt', angle_list_1.copy(), fmt='%.05f')

# str_index = 0
# traj = np.loadtxt(root_path + '/' + folder_name + '/' +
#                        font_name + '_' + str(str_index) + '_font' + str(type) + '.txt')
#
# writing_controller = Controller(
#     args, img_processor=None, impedance_level=0)
#
# writing_controller.interact_once(
#     traj, impedance_params=[35.0, 25.0, 0.5, 0.1], velocity=0.04, mode='eval')

# target_img = cv2.imread(root_path + '/1_font_1.png')
# writing_controller.interact(path_data, target_img)
