import subprocess
import os
from os import mkdir

import multiprocessing

N_MOVIES = 9800
in_dir = "/media/hcsi3/EmotionImpact/2016data/data/"
out_dir = "/media/hcsi3/EmotionImpact/2016data/frames/"
FNULL = open(os.devnull, 'w')


def process(i):
    filename = in_dir + "ACCEDE" + str(i).zfill(5) + ".mp4"
    mkdir(out_dir + str(i))
    out = out_dir + str(i) + "/"
    count = 0
    time = 0
    while True:
        out_name = out + "frame" + str(count) + ".png"
        subprocess.call('ffmpeg -i ' + filename + ' -ss ' + str(time) + ' -vcodec png -vframes 1 -an -f rawvideo ' + out_name,
                        shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        if os.stat(out_name).st_size == 0:
            os.remove(out_name)
            break
        time += 0.5
        count += 1

if __name__ == "__main__":
    # pool = multiprocessing.Pool(processes=10)
    for i in range(N_MOVIES):
        # pool.apply_async(process, (i, ))
        process(i)
    # pool.close()
    # pool.join()
