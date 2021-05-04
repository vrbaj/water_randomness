import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from nistrng import *


video_directory = "videos"
time_logger = {}


def time_measure(method):
    def time_elapsed(*args, **kw):
        time_start = time.time()
        result = method(*args, **kw)
        time_finish = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((time_finish - time_start))
        else:
            print('%r  %2.2f ms' % (method.__name__, (time_finish - time_start)))
        return result
    return time_elapsed


@time_measure
def odd_even_method(video_path, **kwargs):
    resulting_sequence = []
    skip_param = 5
    video_file = cv2.VideoCapture(video_path)
    video_lenght = video_file.get(cv2.CAP_PROP_FRAME_COUNT)
    while video_file.isOpened():
        current_frame_id = video_file.get(cv2.CAP_PROP_POS_FRAMES)
        if current_frame_id + skip_param < video_lenght:
            video_file.set(cv2.CAP_PROP_POS_FRAMES, current_frame_id + skip_param)
            print(current_frame_id + skip_param)
            ret, frame = video_file.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.equalizeHist(frame)
            random_bit = (frame > 250).sum() % 2
            resulting_sequence.append(random_bit)
        else:
            break
    return resulting_sequence


for file_name in os.listdir(video_directory):
    video_file_path = os.path.join(video_directory, file_name)
    result_sequence = odd_even_method(video_file_path, log_time=time_logger)

eligible_battery: dict = check_eligibility_all_battery(np.array(result_sequence), SP800_22R1A_BATTERY)
# f = open("random_file.txt", "a")
# for item in results:
#     f.write(str(item))
# f.close()
results = run_all_battery(np.array(result_sequence), eligible_battery, False)
for result, elapsed_time in results:
        if result.passed:
            print("- PASSED - score: " + str(np.round(result.score, 3)) + " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")
        else:
            print("- FAILED - score: " + str(np.round(result.score, 3)) + " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")
print(time_logger)

