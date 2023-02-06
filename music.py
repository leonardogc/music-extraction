import cv2
import os
import numpy as np
import math


def save_frame(frame_1, frame_2, curr_time, temp_path, music_sheet_coords, debug):
    frame_1 = frame_1[music_sheet_coords[1]:music_sheet_coords[3], music_sheet_coords[0]:music_sheet_coords[2]]
    frame_2 = frame_2[music_sheet_coords[1]:music_sheet_coords[3], music_sheet_coords[0]:music_sheet_coords[2]]

    final_frame = np.zeros_like(frame_1)

    x = int(final_frame.shape[1] / 2)

    final_frame[:, :x] = frame_2[:, :x]
    final_frame[:, x:] = frame_1[:, x:]
    
    save_path = os.path.join(temp_path, f'{int(curr_time)}.png')
    cv2.imwrite(save_path, final_frame)

    print(f'Saving {save_path}')

    if debug:
        cv2.imshow('frame_1_cropped', frame_1)
        cv2.imshow('frame_2_cropped', frame_2)
        cv2.imshow('final_frame', final_frame)

        cv2.waitKey(0)

def skip_ms(cap, time_step):
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_time = (cap.get(cv2.CAP_PROP_POS_FRAMES) * 1000) / fps

    while True:
        success = cap.grab()

        curr_time = (cap.get(cv2.CAP_PROP_POS_FRAMES) * 1000) / fps

        if not success or curr_time - start_time >= time_step:
            break
    
    if not success:
        return success, None, start_time + time_step

    success, image = cap.retrieve()
    
    return success, image, curr_time


def extract_frames(video_path, 
                    temp_path='temp', 
                    video_start=0,
                    video_end=np.inf,
                    time_step=1000,
                    bar_number_pixel_threshold=20,
                    bar_number_thres=75,
                    bar_number_coords=(70, 0, 70 + 80, 80),
                    music_sheet_coords=(0, 0, 1920, 350),
                    debug=False):

    os.makedirs(temp_path, exist_ok=True)

    curr_time = video_start

    # init capture
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, curr_time)
    success, frame = cap.read()

    bar_number_1 = cv2.cvtColor(frame[bar_number_coords[1]:bar_number_coords[3], bar_number_coords[0]:bar_number_coords[2]], cv2.COLOR_BGR2GRAY)
    frame_1 = frame
    frame_2 = frame

    while True:

        if not success or curr_time > video_end:
            save_frame(frame_1, frame_2, curr_time, temp_path, music_sheet_coords, debug)
            break

        bar_number = cv2.cvtColor(frame[bar_number_coords[1]:bar_number_coords[3], bar_number_coords[0]:bar_number_coords[2]], cv2.COLOR_BGR2GRAY)

        bar_number_diff = np.absolute(bar_number_1.astype(int) - bar_number.astype(int)).astype(np.uint8)
        bar_number_pixel_change = np.sum(bar_number_diff > bar_number_pixel_threshold)

        if debug:
            print(curr_time)
            print(bar_number_pixel_change)

            cv2.imshow('frame', frame)
            cv2.imshow('frame_cropped', frame[music_sheet_coords[1]:music_sheet_coords[3], music_sheet_coords[0]:music_sheet_coords[2]])

            cv2.imshow('bar_number', bar_number)
            cv2.imshow('bar_number_1', bar_number_1)

            cv2.imshow('bar_number_diff', bar_number_diff)

            cv2.waitKey(0)
        
        if bar_number_pixel_change > bar_number_thres:
            save_frame(frame_1, frame_2, curr_time, temp_path, music_sheet_coords, debug)

            bar_number_1 = bar_number
            frame_1 = frame
            frame_2 = frame
        else:
            frame_2 = frame
        
        success, frame, curr_time = skip_ms(cap, time_step)

    cap.release()
    cv2.destroyAllWindows()


def generate_music_sheet(out_path='out', frames_path='temp', buffer=0, debug=False):
    os.makedirs(out_path, exist_ok=True)

    frames = os.listdir(frames_path)
    frames.sort(key=lambda x: int(x.split('.')[0]))

    frame_shape = cv2.imread(os.path.join(frames_path, frames[0])).shape[:2]
    print(frame_shape)

    out_shape = (int(frame_shape[1] * (2**0.5)), frame_shape[1])
    print(out_shape)

    frames_per_sheet = math.floor(out_shape[0] / (frame_shape[0] + buffer))
    print(frames_per_sheet)

    total_sheets = math.ceil(len(frames) / frames_per_sheet)
    print(total_sheets)

    free_space = out_shape[0] - (frames_per_sheet * (frame_shape[0] + buffer))

    skip = math.floor(free_space / (frames_per_sheet + 1))

    print(skip)

    for i in range(total_sheets):
        sheet = np.zeros(out_shape, dtype=np.uint8) + 255

        for j in range(frames_per_sheet):
            idx = (i * frames_per_sheet) + j

            if idx >= len(frames):
                break

            frm = cv2.imread(os.path.join(frames_path, frames[idx]))
            frm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

            y = skip + j * (skip + frame_shape[0] + buffer)

            sheet[y:y+frame_shape[0]] = frm
        
        sheet[sheet == frm[1, 1]] = 255

        if debug:
            cv2.imshow('sheet', sheet)
            cv2.waitKey(0)

        cv2.imwrite(os.path.join(out_path, f'{i + 1}.png'), sheet)


if __name__ == '__main__':
    '''extract_frames('y2mate.com - How to play piano part of All Of Me by John Legend_1080pFHR.mp4',
                    temp_path='temp',
                    video_start=5.6*1000)'''

    
    extract_frames('y2mate.com - Céline Dion  My Heart Will Go On Titanic  Piano Tutorial  SHEETS_1080pFHR.mp4',
                    temp_path='temp',
                    time_step=1650,
                    video_start=6.1*1000,
                    video_end=(4*60+30)*1000,
                    bar_number_coords=(135, 0, 135 + 80, 80),
                    music_sheet_coords=(0, 0, 1920, 370))

    generate_music_sheet()

