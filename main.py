import cv2
import numpy as np

def calibrate_camera_from_webcam(grid_size=(5, 8), captures_required=10):
    '''
    grid_size: 棋盤格的格子數，預設為(5, 8)意味著6x9的格子
    captures_required: 需要捕捉的棋盤格圖片數量，預設為10
    二个参数是棋盘格内部的角点的行列数（注意：不是棋盘格的行列数，如下图棋盘格的行列数分别为4、8，而内部角点的行列数分别是3、7，因此这里应该指定为cv::Size(3, 7)）。
    返回值: ret, mtx (相機矩陣), dist (失真參數), rvecs (旋轉向量), tvecs (平移向量)
    '''
    
    cap = cv2.VideoCapture(2)

    # 棋盤格角點在世界坐標中的位置
    objp = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    captured_images = 0
    while captured_images < captures_required:
        ret, frame = cap.read()
        if not ret:
            print("Webcam 讀取錯誤!")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
        #print(ret)

        if ret:
            cv2.drawChessboardCorners(frame, grid_size, corners, ret)
            # 繪製角點順序
            for idx, corner in enumerate(corners):
                #print(corner)
                coord = (int(corner[0][0]), int(corner[0][1]))
                cv2.circle(frame, coord, 5, (0, 0, 255), -1)  # 繪製紅色的圓點
                cv2.putText(frame, str(idx), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Calibration', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):  # 將這部分移到外面
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                captured_images += 1
                print(f"已捕捉 {captured_images} 張棋盤格圖像")

    cv2.destroyAllWindows()
    cap.release()


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def detect_qrcode_corners(image, mtx, dist):
    # 使用OpenCV的QRCodeDetector
    qrDecoder = cv2.QRCodeDetector()
    data, bbox, rectified_image = qrDecoder.detectAndDecode(image)
    if len(data) > 0:
        return np.array(bbox).reshape((4, 2)), True
    else:
        return [], False

# 假設QRCode的物理尺寸是100x100單位
object_points = np.array([[0, 0, 0],  # 左上角
                          [170, 0, 0],  # 右上角
                          [170, 170, 0],  # 右下角
                          [0, 170, 0]], dtype=np.float32)  # 左下角


def draw_axes(img, rvec, tvec, mtx, dist, corners, length=50):
    # Define the axis points
    axis_points = np.float32([[length, 0, 0],  # X-axis
                              [0, length, 0],  # Y-axis
                              [0, 0, -length]]).reshape(-1, 3)  # Z-axis (now negative)

    # Project the 3D axis points to the image plane
    imgpts, jac = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)
    corner = tuple(map(int, corners[0].ravel()))

    img = cv2.line(img, corner, tuple(map(int, imgpts[0].ravel())), (255, 0, 0), 5)  # X-axis is drawn in blue
    img = cv2.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0, 255, 0), 5)  # Y-axis is drawn in green
    img = cv2.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0, 0, 255), 5)  # Z-axis is drawn in red

    return img



def detect_and_display_qrcode(mtx, dist):
    cap = cv2.VideoCapture(2)

    # 設定光流參數
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    old_frame = None
    old_corners = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam 讀取錯誤!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 偵測QRCode的四個角
        corners, detected = detect_qrcode_corners(frame, mtx, dist)

        if detected:
            # 如果偵測到QRCode，更新角點和前一個框架
            old_corners = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
            old_frame = gray.copy()

            for i, corner in enumerate(old_corners):
                a, b = map(int, corner.ravel())
                frame = cv2.circle(frame, (a, b), 5, colors[i], -1)
                frame = cv2.putText(frame, f"({a}, {b})", (a + 5, b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        elif old_frame is not None and old_corners is not None:
            # 使用光流追蹤
            new_corners, status, err = cv2.calcOpticalFlowPyrLK(old_frame, gray, old_corners, None, **lk_params)

            # 檢查所有四個點是否都被找到
            if np.all(status):
                for i, corner in enumerate(new_corners):
                    a, b = map(int, corner.ravel())
                    frame = cv2.circle(frame, (a, b), 5, colors[i], -1)
                    frame = cv2.putText(frame, f"({a}, {b})", (a + 5, b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

                old_corners = new_corners
                old_frame = gray.copy()

                retval, rvec, tvec = cv2.solvePnP(object_points, new_corners, mtx, dist)  # 使用 new_corners 而不是 corners
                frame = draw_axes(frame, rvec, tvec, mtx, dist, new_corners)  # 添加 new_corners 作為參數

                distance_to_camera = tvec[2][0]

                text = f"Distance from QRCode center to camera: {distance_to_camera:.2f} units"
                position = (10, 30)  # x, y position on the top-left corner of the screen
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (0, 255, 255)  # yellow color
                thickness = 2
                cv2.putText(frame, text, position, font, font_scale, color, thickness)
                #print("Distance from QRCode center to camera:", distance_to_camera)
            else:
                # 如果有缺少的點，則清空前一個框架和角點
                old_frame = None
                old_corners = None

        cv2.imshow('QRCode Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate_camera_from_webcam()
    print("相機矩陣:\n", mtx)
    print("失真參數:\n", dist)
    detect_and_display_qrcode(mtx, dist)




