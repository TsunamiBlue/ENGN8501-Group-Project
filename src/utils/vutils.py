import glob
import cv2
import time
from tqdm import tqdm
import numpy as np
import os


class VideoGenerator():
    def __init__(self, video=None, dire=None):
        self.dire = dire
        self.video = video
        if self.video is not None:
            cap = cv2.VideoCapture(self.video)
            (major_ver, _, _) = (cv2.__version__).split('.')
    
            if int(major_ver)  < 3 :
                self.fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                # print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(self.fps))
            else :
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(self.fps))
            cap.release()
        else:
            self.fps = 25

    def video_gen(self):
        cap = cv2.VideoCapture(self.video)
        while(1):
            ret, img = cap.read()
            if not ret:
                break
            if isinstance(self.video, str) and 'ios' in self.video:
                img = np.rot90(img, -1)
            yield img[:, :, ::-1]
        cap.release()

    def dir_gen(self):
        imgs = sorted(glob.glob('{}/*.jpg'.format(self.dire)))
        for img_path in imgs:
            yield cv2.imread(img_path)[:, :, ::-1]

    def __call__(self):
        if self.video is not None:
            return self.video_gen(),self.fps
        elif self.dire is not None:
            return self.dir_gen(),self.fps
        else:
            return []

class VideoDumper():
    def __init__(self, file_name=None, monit=True, fps=25):
        self.file_name = file_name
        self.out = None
        self.monit = monit
        self.timer = time.time()
        self.fps = fps

    def write(self, img):
        if img.ndim==3:
            img = img[:, :, ::-1]
        if self.out is None:
            if img.ndim==3:
                h, w, c = img.shape
            elif img.ndim==2:
                h, w = img.shape
            else:
                raise Exception(f'Unexpected dim {img.shape}')
            print(f'Dumping video [{self.file_name}] {h} x {w}')


            if self.file_name:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.out = cv2.VideoWriter(self.file_name,
                                    fourcc, self.fps, (w, h))
                assert(self.out.isOpened())
            else:
                self.out = 1
        if self.monit:
            cv2.imshow("capture", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
                # break
        if self.file_name:
            self.out.write(img)
        new_timer = time.time()
        print(f'\rFrame time {new_timer - self.timer:.4f} s, {1. / (new_timer - self.timer):.2f} fps')
        self.timer = new_timer

    def __del__(self):
        pass
        # if self.monit:
            # cv2.destroyAllWindows()
        # if self.file_name:
            # self.out.release()

def show_video(path):
    gen = VideoGenerator(path)
    out = VideoDumper()
    for i in gen():
        out.write(i)
        time.sleep(1/30)


if __name__ == '__main__':
    # show_video(0)
    # exit()

    # video_list = [
    #     './ios_raw.MOV',
    #     './our_raw.MOV',
    #     './faceu_raw.MP4',
    #     './tt_raw.mp4',
    # ]
    # for v in video_list:
        # show_video(v)


    # import numpy as np
    # gen_i = VideoGenerator('MyTest/setting_room/setting_room_480p')
    # gen_t = VideoGenerator('Downloads/trans.mp4')
    # out = VideoDumper('1.avi')
    # for i, j in zip(gen_i(), gen_t()):
    #     j = cv2.resize(j, (i.shape[1], i.shape[0]), interpolation=cv2.INTER_CUBIC)
    #     print(i.shape)
    #     print(j.shape)
    #     img = np.concatenate([i, j], 0)
    #     # img = np.rot90(i)
    #     # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    #     out.write(img)


    video = '/Users/versa/PycharmProjects/MyTracking/MyTest/setting_room/setting_room_480p.mov'
    ROOT = '/E:/Intern/Versa/video/raw/天官赐福/天官赐福'
    out = '/Users/versa/PycharmProjects/MyTracking/MyTest/setting_room/setting_room_480p'
    cap = cv2.VideoCapture(video)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    skip_frame = 3
    expand_name = '.jpg'
    if not cap.isOpened():
        print("Please check the path.")
    cnt = 0
    count = 0
    while True:
        ret, frame = cap.read()
        cnt += 1
        # how many frame to cut
        if not ret:
            break
        if cnt % skip_frame == 0:
            count += 1
            cv2.imwrite(os.path.join(out, str(count) + expand_name), frame)




