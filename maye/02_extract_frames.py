import cv2
from os import mkdir

N_MOVIES = 9800
in_dir = "/media/hcsi3/EmotionImpact/2016data/data/"
out_dir = "/media/hcsi3/EmotionImpact/2016data/frames/"
for i in range(N_MOVIES):
    filename = in_dir + "ACCEDE" + str(i).zfill(5) + ".mp4"
    mkdir(out_dir + str(i))
    out = out_dir + str(i) + "/"
    video = cv2.VideoCapture(filename)
    success, image = video.read()
    count = 0
    success = True
    while success:
        success, image = video.read()
        if not success:
            print(str(i) + "Done!")
            break
        cv2.imwrite(out + "frame" + str(count) + ".png", image)
        count += 1
