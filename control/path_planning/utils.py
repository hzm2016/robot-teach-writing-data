from re import sub
import numpy as np
import math
import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2
from scipy import signal

# b, a = signal.butter(8, 0.02, 'lowpass')
# filtedData = signal.filtfilt(b, a, data)  #data为要过滤的信号

Length = [0.30, 0.150, 0.25, 0.125]
L_1 = Length[0]
L_2 = Length[2]

action_dim = 2
Ts = 0.001

# writing space
WIDTH = 0.360
HEIGHT = 0.360

linewidth = 3.0


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def initial_parameter_estimate(num_demons_each_style=30):
    """ demonstration for each styles with zero impedance """
    num_demonstrations = 30
    writting_vel = 0.0
    impedance_params = np.ones(2)

    # captured images
    images_list = []
    distance_list = []
    for i in range(num_demonstrations):
        # distance_list.appen
        # images_list.append(i.copy())
        pass

    return writting_vel, impedance_params


def IK(point):
    """
        Inverse kinematics
    """
    angle = np.zeros(action_dim)
    x1 = point[0]
    x2 = point[1]

    L = x1**2 + x2**2

    gamma = math.atan2(x2, x1)

    cos_belta = (L_1 ** 2 + L - L_2 ** 2) / (2 * L_1 * np.sqrt(L))

    if cos_belta > 1:
        angle_1 = gamma
    elif cos_belta < -1:
        angle_1 = gamma - np.pi
    else:
        angle_1 = gamma - math.acos(cos_belta)

    cos_alpha = (L_1 ** 2 - L + L_2 ** 2) / (2 * L_1 * L_2)

    if cos_alpha > 1:
        angle_2 = np.pi
    elif cos_alpha < -1:
        angle_2 = 0
    else:
        angle_2 = np.pi - math.acos(cos_alpha)

    angle[0] = np.round(angle_1, 5).copy()
    angle[1] = np.round(angle_2, 5).copy()
    return angle


def forward_ik(angle):
    """
        calculate point
    """
    point = np.zeros_like(angle)
    point[0] = L_1 * math.cos(angle[0]) + L_2 * math.cos(angle[0] + angle[1])
    point[1] = L_1 * math.sin(angle[0]) + L_2 * math.sin(angle[0] + angle[1])

    return point


def Jacobian(theta):
    """
        calculate Jacobian
    """
    J = np.zeros((action_dim, action_dim))

    J[0, 0] = -L_1 * math.sin(theta[0]) - L_2 * math.sin(theta[0] + theta[1])
    J[0, 1] = -L_2 * math.sin(theta[0] + theta[1])
    J[1, 0] = L_1 * math.cos(theta[0]) + L_2 * math.cos(theta[0] + theta[1])
    J[1, 1] = L_2 * math.cos(theta[0] + theta[1])

    return J


def Stiff_convert(theta, stiffness, damping): 
    """ 
        convert stiffness from task space to joint space
    """ 
    J = Jacobian(theta)   
    stiff_joint = J.transpose().dot(stiffness).dot(J)  
    damping_joint = J.transpose().dot(damping).dot(J) 

    return stiff_joint, damping_joint 


def path_planning(start_point, target_point, velocity=0.04):
    """
        path planning ::: linear function
    """
    dist = np.linalg.norm((start_point - target_point), ord=2)
    T = dist/velocity
    N = int(T / Ts)

    x_list = np.linspace(start_point[0], target_point[0], N)
    y_list = np.linspace(start_point[1], target_point[1], N)

    point = start_point
    angle_list = []
    for i in range(N):
        point[0] = x_list[i]
        point[1] = y_list[i]
        angle = IK(point)
        angle_list.append(angle)

    return np.array(angle_list), N


def rotate_point(angle, x_list, y_list):
    x_rotated = x_list * math.cos(angle) + y_list * math.sin(angle)
    y_rotated = y_list * math.cos(angle) - x_list * math.sin(angle)

    return x_rotated, y_rotated


def contact_two_stroke(angle_list_1, angle_list_2,
                       params_list_1, params_list_2,
                       inverse=False
                       ):
    if inverse:
        final_angle_list = np.vstack((angle_list_1, angle_list_2))
        final_params_list = np.vstack((params_list_1, params_list_2))
    else:
        final_angle_list = np.vstack((angle_list_1, np.flipud(angle_list_2)))
        final_params_list = np.vstack((params_list_1, np.flipud(params_list_2)))

    print("Final angle shape :", final_angle_list.shape)
    print("Final params shape :", final_params_list.shape)
    return final_angle_list, final_params_list


def real_stroke_path(task_points_list=None):
    fig = plt.figure(figsize=(4, 4))
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for i in range(len(task_points_list)):
        plt.plot(task_points_list[i][:, 0], task_points_list[i][:, 1], linewidth=linewidth + 2)
        # plt.scatter(task_points_list[i][0, 0], task_points_list[i][0, 1], s=100, c='b', marker='o')
        # plt.text(task_points_list[i][0, 0], task_points_list[i][0, 1], str(i + 1), rotation=90)

    plt.ylim([-WIDTH / 2, WIDTH / 2])
    plt.xlim([0.13, 0.13 + WIDTH])
    # plt.xlabel('$x_1$(m)')
    # plt.ylabel('$x_2$(m)')
    # plt.axis('off')
    plt.tight_layout()

    img_path = fig2data(fig)
    # img = img_path.transpose(Image.ROTATE_90)  # 将图片旋转90度
    # img_path.show()
    img_show = np.rot90(img_path, -1)
    # cv2.imwrite(word_folder + '/' + word_name + '/' + word_name +'.png', img_show)
    # plt.show()
    cv2.imshow("Real path :", img_show)
    cv2.waitKey(0)


def plot_stroke_path(period, traj, 
image_points, task_points, 
angle_list, fig_name='Stroke Path'
):
    """
        check the planned path
    """
    t_list = np.linspace(0.0, period, angle_list.shape[0])
    plt.rcParams['font.size'] = 8
    print("task points :", task_points)
    plt.figure(figsize=(15, 4))

    plt.title(fig_name)

    sub_fig = 4
    plt.subplot(1, sub_fig, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    plt.plot(traj[:, 1], traj[:, 0], marker='o', linewidth=linewidth)
    plt.plot(image_points[:, 0], image_points[:, 1], linewidth=linewidth - 2)

    plt.xlim([0, 128])
    plt.ylim([0, 128])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    # plt.axis('equal')
    plt.tight_layout()

    plt.subplot(1, sub_fig, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    plt.plot(traj[:, 0], linewidth=linewidth)
    plt.plot(traj[:, 1], linewidth=linewidth)
    # plt.plot(image_points[:, 0], image_points[:, 1], linewidth=linewidth - 2)

    plt.xlim([0, 128])
    plt.ylim([0, 128])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    # plt.axis('equal')
    plt.tight_layout()

    plt.subplot(1, sub_fig, 3)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    plt.plot(task_points[:, 0], task_points[:, 1], linewidth=linewidth + 2, color='r')
    # plt.plot(x_inner, y_inner, linewidth=linewidth + 2, color='r')
    plt.scatter(task_points[0, 0], task_points[0, 1], s=100, c='b', marker='o')
    # plt.scatter(x_inner[0], y_inner[0], s=100, c='b', marker='o')
    # print("distance :::", np.sqrt((x_1_list[0] - x_inner[0])**2 + (x_2_list[0] - y_inner[0])**2))
    plt.ylim([-WIDTH / 2, WIDTH / 2])
    plt.xlim([0.13, 0.13 + WIDTH])
    plt.xlabel('$x_1$(m)')
    plt.ylabel('$x_2$(m)')

    plt.subplot(1, sub_fig, 4)
    plt.plot(t_list, angle_list[:, 0], linewidth=linewidth, label='$q_1$')
    # plt.plot(t_list[1:], angle_vel_1_list_e, linewidth=linewidth, label='$d_{q1}$')
    plt.plot(t_list, angle_list[:, 1], linewidth=linewidth, label='$q_2$')
    # plt.plot(t_list[1:], angle_vel_2_list_e, linewidth=linewidth, label='$d_{q2}$')

    plt.xlabel('Time (s)')
    plt.ylabel('One-loop Angle (rad)')
    plt.legend()

    plt.show()


def plot_word_path(period_list, traj_list, image_points_list, task_points_list, word_angle_list,
                   word_folder='../control/data', word_name='Stroke Path'):
    """
        plot one word path
    """
    plt.figure(figsize=(15, 4))
    plt.title(word_name)

    plt.subplot(1, 3, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for i in range(len(traj_list)):
        # traj_list[i].transpose((0, 1)).copy()
        # plt.plot(traj_list[i][:, 1], traj_list[i][:, 0], marker='o', linewidth=linewidth)
        # plt.plot(image_points_list[i][:, 0], image_points_list[i][:, 1], linewidth=linewidth - 2)

        traj_x_rotated, traj_y_rotated = rotate_point(math.pi/2, traj_list[i][:, 1], traj_list[i][:, 0])
        # print("x_rotated :", traj_x_rotated, "y_rotated :", traj_y_rotated)
        plt.plot(traj_x_rotated, traj_y_rotated, marker='o', linewidth=linewidth)
        image_x_rotated, image_y_rotated = rotate_point(math.pi / 2, image_points_list[i][:, 0], image_points_list[i][:, 1])
        # plt.plot(traj_list[i][:, 0], traj_list[i][:, 1], marker='o', linewidth=linewidth)
        plt.plot(image_x_rotated, image_y_rotated, linewidth=linewidth - 2)

        # if i == range(len(traj_list) - 1):
        #     len(traj_list)

    plt.xlim([0, 128])
    plt.ylim([-128, 0])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plt.subplot(1, 3, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for i in range(len(traj_list)):
        # plt.plot(task_points_list[i][:, 0], task_points_list[i][:, 1], linewidth=linewidth + 2, color='r')
        # plt.scatter(task_points_list[i][0, 0], task_points_list[i][0, 1], s=100, c='b', marker='o')

        rotated_task_points_x_list, rotated_task_points_y_list = \
            rotate_point(math.pi / 2, task_points_list[i][:, 0], task_points_list[i][:, 1])
        plt.plot(rotated_task_points_x_list, rotated_task_points_y_list, linewidth=linewidth + 2, color='r')
        if i == len(traj_list) -1:
            pass
        else:
            plt.scatter(rotated_task_points_x_list[0], rotated_task_points_y_list[0], s=100, c='b', marker='o')

        # rotated_task_points_x_list, rotated_task_points_y_list = \
        #     rotate_point(math.pi / 2, task_points_list[i][:, 0], task_points_list[i][:, 1])
        # plt.scatter(x_inner[0], y_inner[0], s=100, c='b', marker='o')
        # print("distance :::", np.sqrt((x_1_list[0] - x_inner[0])**2 + (x_2_list[0] - y_inner[0])**2))

    # plt.ylim([-WIDTH / 2, WIDTH / 2])
    # plt.xlim([0.13, 0.13 + WIDTH])
    plt.xlabel('$x_1$(m)')
    plt.ylabel('$x_2$(m)')

    plt.subplot(1, 3, 3)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    total_period = sum(period_list)
    print("total_period :", total_period)
    for i in range(len(traj_list)):
        if i == 0:
            start_period = 0.0
            # t_list = np.linspace(0.0, period_list[i], word_angle_list[i].shape[0])
        else:
            start_period = sum(period_list[:i])

        t_list = np.linspace(start_period, period_list[i] + start_period, word_angle_list[i].shape[0])
        print("period :", start_period, period_list[i] + start_period)

        plt.plot(t_list, word_angle_list[i][:, 0], linewidth=linewidth, label='$q_1$')
        # plt.plot(t_list[1:], angle_vel_1_list_e, linewidth=linewidth, label='$d_{q1}$')
        plt.plot(t_list, word_angle_list[i][:, 1], linewidth=linewidth, label='$q_2$')
        # plt.plot(t_list[1:], angle_vel_2_list_e, linewidth=linewidth, label='$d_{q2}$')

    plt.xlabel('Time (s)')
    plt.ylabel('One-loop Angle (rad)')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(word_folder + '/' + word_name + '/' + word_name + '_traj.png')

    plt.show()

    fig = plt.figure(figsize=(4, 4))
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for i in range(len(traj_list)-1):
        # rotate_task_point = task_points_list[i].transpose((1, 0))
        if i == len(traj_list)-2:
            task_points_list_x_last = np.hstack((task_points_list[i][:, 0], task_points_list[i+1][:, 0]))
            task_points_list_y_last = np.hstack((task_points_list[i][:, 1], task_points_list[i + 1][:, 1]))
            plt.plot(task_points_list_x_last, task_points_list_y_last, linewidth=linewidth + 2)
            plt.scatter(task_points_list_x_last[0], task_points_list_y_last[0], s=100, c='b', marker='o')
            plt.text(task_points_list_x_last[0], task_points_list_y_last[0], str(i + 1), rotation=90)
        else:
            plt.plot(task_points_list[i][:, 0], task_points_list[i][:, 1], linewidth=linewidth + 2)
            plt.scatter(task_points_list[i][0, 0], task_points_list[i][0, 1], s=100, c='b', marker='o')
            plt.text(task_points_list[i][0, 0], task_points_list[i][0, 1], str(i + 1), rotation=90)

    plt.ylim([-WIDTH / 2, WIDTH / 2])
    plt.xlim([0.13, 0.13 + WIDTH])
    plt.yticks([-WIDTH / 2, 0, WIDTH / 2])
    plt.xticks([0.13, 0.13 + WIDTH])
    # plt.xlabel('$x_1$(m)')
    # plt.ylabel('$x_2$(m)')
    plt.tight_layout()

    img_path = fig2data(fig)
    # img = img_path.transpose(Image.ROTATE_90)  # 将图片旋转90度
    # img_path.show()
    img_show = np.rot90(img_path, -1)
    cv2.imwrite(word_folder + '/' + word_name + '/' + word_name + '.png', img_show)
    # cv2.imshow(word_folder + '/' + word_name + '/' + word_name +'.png', img_show)
    # cv2.waitKey(0)
    # plt.imshow(img_show)
    # plt.show()


def cope_real_word_path(
        root_path=None,
        file_name='mu',
        stroke_index=1,
        epi_times=1,
        delimiter=' ',
        skiprows=1,
):
    x_list = []
    y_list = []
    for epi_index in range(epi_times):
        angle_list = np.loadtxt(
            root_path + file_name + str(stroke_index) + '_' + str(epi_index) + '.txt',
            delimiter=delimiter,
            skiprows=skiprows
            )

        # angle_list : desired and real-time
        angle_list_1_e = angle_list[:, 0]
        angle_list_2_e = angle_list[:, 3]
        angle_list_1_t = angle_list[:, 1]
        angle_list_2_t = angle_list[:, 4]

        x_e = L_1 * np.cos(angle_list_1_e) + L_2 * np.cos(angle_list_1_e + angle_list_2_e)
        y_e = L_1 * np.sin(angle_list_1_e) + L_2 * np.sin(angle_list_1_e + angle_list_2_e)

        x_t = L_1 * np.cos(angle_list_1_t) + L_2 * np.cos(angle_list_1_t + angle_list_2_t)
        y_t = L_1 * np.sin(angle_list_1_t) + L_2 * np.sin(angle_list_1_t + angle_list_2_t)

        x_list.append(x_t)
        y_list.append(y_t)

    return x_list, y_list