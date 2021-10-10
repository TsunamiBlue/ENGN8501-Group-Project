import cv2
import os
from src.utils.vutils import VideoDumper,VideoGenerator

video_name = 'test_1'
ROOT = './results/target/test_latest/images'
out = VideoDumper(ROOT+'.avi',monit=False,fps=25)
for _,_,files in os.walk(ROOT):
    for f in files[::-1]:
        try:
            # a = int(f[6:-4])
            a = f[6:11]
            assert a=="synth"
        except:
            files.remove(f)
    files.sort(key=lambda x:int(x[:5]))
    for file in files:
        img = cv2.imread(os.path.join(ROOT,file))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (640, 480))
        out.write(img)
