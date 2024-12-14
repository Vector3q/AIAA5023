import cv2

def get_last_frame(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置视频的当前帧为最后一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    # 读取最后一帧
    ret, last_frame = cap.read()

    # 释放视频捕获对象
    cap.release()

    if ret:
        return last_frame
    else:
        print("Error: Could not read the last frame.")
        return None