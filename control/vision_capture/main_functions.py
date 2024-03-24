import cv2
import pyzed.sl as sl
import numpy as np

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1


def show_video():
    print("Running...")
    init = sl.InitParameters()
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")

    # init.camera_resolution = sl.RESOLUTION.HD480
    
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    print_camera_information(cam)
    print_help()

    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            cv2.imshow("ZED", mat.get_data())
            key = cv2.waitKey(5)
            settings(key, cam, runtime, mat)
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()
    print("\nFINISH")


def calibration_camera():
    
    pass


def image_precessing(img_path, img_name):
    img = cv2.imread(img_path + img_name + '.png')
    height, weight = img.shape[:2]
    print("height :::", height)
    print("weight :::", weight)
    
    # need to define according to robot position
    crop_img = img[130:height-100, 455:945]
    cv2.imshow("Processed Image", crop_img)
    resize_img = cv2.resize(crop_img, (128, 128), cv2.INTER_AREA)
    cv2.imshow("Processed Image", resize_img)
    
    cols, rows = resize_img.shape[:2]
    
    # rotate image :::
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    dst_img = cv2.warpAffine(resize_img, M, (cols, rows))
    cv2.imwrite(img_path + img_name + '_final.png', dst_img)
    cv2.imshow("Processed Image", dst_img)
    cv2.waitKey()
    return img


def capture_image(file_path='', font_name='font_1', size=(128, 128)):
    print("Capture image ...")
    
    init = sl.InitParameters()
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    
    # init.camera_resolution = sl.RESOLUTION.HD480
    
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    err = cam.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        cv2.imshow("ZED", mat.get_data())
        img = mat.get_data()

        cv2.imwrite(file_path + '/' + font_name + '_original.png', img)
        
        cols, rows = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -2, 1)
        rotated_img = cv2.warpAffine(img, M, (rows, cols))
        # cv2.imshow("rotated_img", rotated_img)
        # cv2.waitKey(0)
        
        # crop image
        crop_img = rotated_img[200:750, 350:1000]
        # feature_extraction(ori_img=crop_img)
        # cv2.imwrite(file_path + '/' + font_name + '_ori_image.png', img[200:750, 350:1000])
        
        # ori_img = img[200:700, 400:900]
        ori_img = crop_img[0:490, 50:540]
        height, weight = ori_img.shape[:2]
        print("height :", height)
        print("weight :", weight)
        
        # offset_value = 0.03
        # offset_img = int(offset_value/0.37 * 490)
        # pos_img = ori_img.copy()
        # pos_img[0:490, 0:490-offset_img] = ori_img[0:490, offset_img:490]
        # pos_img[0:490, 490-offset_img:490] = ori_img[0:490, 0:offset_img]
        # print("white img :", ori_img[0:490, 0:offset_img])

        cols, rows = ori_img.shape[:2]
        M = np.float32([[1, 0, -30], [0, 1, 0]])
        pos_img = cv2.warpAffine(ori_img, M, (rows, cols))
        
        cv2.imwrite(file_path + '/' + font_name + '_ori.png', ori_img)
        cv2.imwrite(file_path + '/' + font_name + '_pos.png', pos_img)
        
        # need to define according tob root position
        # crop_img = img[200:750, 350:900]
        # crop_img = img[200:750, ]
        # crop_img = rotate_img[350:900, 200:750]
        
        resize_img = cv2.resize(ori_img, size, cv2.INTER_AREA)
        # cv2.imshow("Processed Image", resize_img)
        
        # rotate image :::
        cols, rows = resize_img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
        final_img = cv2.warpAffine(resize_img, M, (cols, rows))
        
        cv2.imwrite(file_path + '/' + font_name + '.png', final_img)
        key = cv2.waitKey(5)
        settings(key, cam, runtime, mat)
    else:
        key = cv2.waitKey(5)
        
    cv2.destroyAllWindows()
    
    cam.close()
    print("\nFINISH ...")
    
    return final_img


def feature_extraction(ori_img=None):
    """
        extract lines
    """
    cv2.imshow('', ori_img)
    blur_img = cv2.GaussianBlur(ori_img, (3, 3), 0)
    edges = cv2.Canny(blur_img, 50, 150, apertureSize=3)
    cv2.imshow("Canny :", edges)
    lines = cv2.HoughLines(edges, 0.5, np.pi / 180, 118)

    result = blur_img.copy()
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        print('rho :', rho)
        print('theta :', theta)

        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
            pt1 = (int(rho / np.cos(theta)), 0)
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            print('horizon :')
            print('pt1 :', pt1)
            print('pt2 :', pt2)
            cv2.line(result, pt1, pt2, (255))
        else:
            print('vertical :')
            pt1 = (0, int(rho / np.sin(theta)))
            print('pt1 :', pt1)
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
            print('pt2 :', pt2)
            cv2.line(result, pt1, pt2, (255), 1)

    cv2.imshow("Result :", result)
    cv2.waitKey(0)
    

def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2),
                                         cam.get_camera_information().camera_resolution.height))
    print("Camera FPS: {0}.".format(cam.get_camera_information().camera_fps))
    print("Firmware: {0}.".format(cam.get_camera_information().camera_firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))


def print_help():
    print("Help for camera setting controls")
    print("  Increase camera settings value:     +")
    print("  Decrease camera settings value:     -")
    print("  Switch camera settings:             s")
    print("  Reset all parameters:               r")
    print("  Record a video:                     z")
    print("  Quit:                               q\n")


def settings(key, cam, runtime, mat):
    if key == 115:  # for 's' key
        switch_camera_settings()
    elif key == 43:  # for '+' key
        current_value = cam.get_camera_settings(camera_settings)
        cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:  # for '-' key
        current_value = cam.get_camera_settings(camera_settings)
        if current_value >= 1:
            cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
            print(str_camera_settings + ": " + str(current_value - step_camera_settings))
    elif key == 114:  # for 'r' key
        cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
        print("Camera settings: reset")
    elif key == 122:  # for 'z' key
        record(cam, runtime, mat)


def switch_camera_settings():
    global camera_settings
    global str_camera_settings
    if camera_settings == sl.VIDEO_SETTINGS.BRIGHTNESS:
        camera_settings = sl.VIDEO_SETTINGS.CONTRAST
        str_camera_settings = "Contrast"
        print("Camera settings: CONTRAST")
    elif camera_settings == sl.VIDEO_SETTINGS.CONTRAST:
        camera_settings = sl.VIDEO_SETTINGS.HUE
        str_camera_settings = "Hue"
        print("Camera settings: HUE")
    elif camera_settings == sl.VIDEO_SETTINGS.HUE:
        camera_settings = sl.VIDEO_SETTINGS.SATURATION
        str_camera_settings = "Saturation"
        print("Camera settings: SATURATION")
    elif camera_settings == sl.VIDEO_SETTINGS.SATURATION:
        camera_settings = sl.VIDEO_SETTINGS.SHARPNESS
        str_camera_settings = "Sharpness"
        print("Camera settings: Sharpness")
    elif camera_settings == sl.VIDEO_SETTINGS.SHARPNESS:
        camera_settings = sl.VIDEO_SETTINGS.GAIN
        str_camera_settings = "Gain"
        print("Camera settings: GAIN")
    elif camera_settings == sl.VIDEO_SETTINGS.GAIN:
        camera_settings = sl.VIDEO_SETTINGS.EXPOSURE
        str_camera_settings = "Exposure"
        print("Camera settings: EXPOSURE")
    elif camera_settings == sl.VIDEO_SETTINGS.EXPOSURE:
        camera_settings = sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE
        str_camera_settings = "White Balance"
        print("Camera settings: WHITEBALANCE")
    elif camera_settings == sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE:
        camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
        str_camera_settings = "Brightness"
        print("Camera settings: BRIGHTNESS")


def record(cam, runtime, mat, filepath=None):
    vid = sl.ERROR_CODE.FAILURE
    out = False
    while vid != sl.ERROR_CODE.SUCCESS and not out:

        record_param = sl.RecordingParameters(filepath, sl.SVO_COMPRESSION_MODE.H264)
        vid = cam.enable_recording(record_param)
        
        if vid == sl.ERROR_CODE.SUCCESS:
            print("Recording started...")
            out = True
            print("Hit spacebar to stop recording: ")
            key = False
            while key != 32:
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(mat)
                    cv2.imshow("ZED", mat.get_data())
                    key = cv2.waitKey(5)
        else:
            print("Help: you must enter the filepath + filename + SVO extension.")
            print("Recording not started.")
    
    cam.disable_recording()
    print("Recording finished.")
    cv2.destroyAllWindows()


def record_video(filepath=None):
    print("Record Video ...")
    init = sl.InitParameters()
    vid = sl.ERROR_CODE.FAILURE
    cam = sl.Camera()
    
    if not cam.is_opened():
        print("Opening ZED Camera...")
    
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    
    out = False
    while vid != sl.ERROR_CODE.SUCCESS and not out:

        record_param = sl.RecordingParameters(filepath, sl.SVO_COMPRESSION_MODE.H264)
        vid = cam.enable_recording(record_param)

        if vid == sl.ERROR_CODE.SUCCESS:
            print("Recording started...")
            out = True
            print("Hit spacebar to stop recording: ")
            key = False
            while key != 32:
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(mat)
                    cv2.imshow("ZED", mat.get_data())
                    key = cv2.waitKey(5)
        else:
            print("Help: you must enter the filepath + filename + SVO extension.")
            print("Recording not started.")

    cam.disable_recording()
    print("Recording finished.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # print("Running...")
    # init = sl.InitParameters()
    # cam = sl.Camera()
    # if not cam.is_opened():
    # 	print("Opening ZED Camera...")
    #
    # # init.camera_resolution = sl.RESOLUTION.HD480
    #
    # status = cam.open(init)
    # if status != sl.ERROR_CODE.SUCCESS:
    # 	print(repr(status))
    # 	exit()
    #
    # runtime = sl.RuntimeParameters()
    # mat = sl.Mat()
    #
    # # print_camera_information(cam)
    # # print_help()
    #
    # # record(cam, runtime, mat, filepath='font_4.svo')
    #
    # cam.close()
    # print("\nFINISH")
    
    show_video()
    
    # capture_image(font_name='capture_data/font_1')
