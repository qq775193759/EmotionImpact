import cv2
from os import mkdir
import multiprocessing

N_MOVIES = 9800
in_dir = "/media/hcsi3/EmotionImpact/2016data/data/"
out_dir = "/media/hcsi3/EmotionImpact/2016data/frames/"


def process(i):
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
            print(str(i) + "\t\t\tDone!")
            break
        cv2.imwrite(out + "frame" + str(count) + ".png", image)
        count += 1


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=10)
    for i in range(N_MOVIES):
        pool.apply_async(process, (i, ))
    pool.close()
    pool.join()
