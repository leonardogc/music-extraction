import cv2
import os
import numpy as np
import math

def get_frame(cap, time_stamp):
    cap.set(cv2.CAP_PROP_POS_MSEC, time_stamp)
    success, image = cap.read()

    return success, image


def extract_frames(video_path, 
                    temp_path='temp', 
                    video_start=0,
                    video_end=np.inf, 
                    bpm=60,
                    bpb=4,
                    bar_number_thres=0.97,
                    diff_thres=20,
                    bar_number_coords=(70, 0, 70 + 80, 80),
                    music_sheet_coords=(0, 0, 1920, 350)):

    os.makedirs(temp_path, exist_ok=True)

    # init capture
    cap = cv2.VideoCapture(video_path)

    curr_stamp = video_start

    bar_length_ms = (bpb * 60000) / bpm

    last_bar_number = None

    while True:
        success, frame = get_frame(cap, curr_stamp)

        if not success or curr_stamp > video_end:
            break

        bar_number = frame[bar_number_coords[1]:bar_number_coords[3], bar_number_coords[0]:bar_number_coords[2]]

        if last_bar_number is None:
            last_bar_number = (np.random.rand(*(bar_number.shape))*255).astype(bar_number.dtype)

        bar_number_diff = cv2.matchTemplate(last_bar_number, bar_number, cv2.TM_CCOEFF_NORMED)[0, 0]
        
        if bar_number_diff < bar_number_thres:
            frame1 = frame[music_sheet_coords[1]:music_sheet_coords[3], music_sheet_coords[0]:music_sheet_coords[2]]
            success, frame2 = get_frame(cap, curr_stamp + 2 * bar_length_ms)

            if not success:
                final_frame = frame1
            else:
                frame2 = frame2[music_sheet_coords[1]:music_sheet_coords[3], music_sheet_coords[0]:music_sheet_coords[2]]

                frame_diff = np.absolute(frame2.astype(int) - frame1.astype(int)).astype(np.uint8)
                frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

                y, x = np.mean(np.argwhere(frame_diff > diff_thres), axis=0).astype(int)

                final_frame = np.zeros_like(frame1)
                final_frame[:, :x] = frame2[:, :x]
                final_frame[:, x:] = frame1[:, x:]

                '''cv2.imshow('frame1', frame1)
                cv2.imshow('frame2', frame2)
                cv2.imshow('frame_diff', frame_diff)
                cv2.imshow('final_frame', final_frame)'''
            
            save_path = os.path.join(temp_path, f'{int(curr_stamp)}.png')
            cv2.imwrite(save_path, final_frame)

            print(f'Saving {save_path}')

            last_bar_number = bar_number

        '''cv2.imshow('frame', frame)
        cv2.imshow('bar_number', bar_number)

        print(curr_stamp)

        # Set waitKey
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break'''

        print(curr_stamp)

        curr_stamp += bar_length_ms

    cap.release()
    cv2.destroyAllWindows()
    

def generate_music_sheet(out_path='out', frames_path='temp', buffer=0):
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

            y = skip + j * (frame_shape[0] + buffer + skip)

            sheet[y:y+frame_shape[0]] = frm
        
        sheet[sheet == 249] = 255

        '''cv2.imshow('sheet', sheet)

        # Set waitKey
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break'''

        cv2.imwrite(os.path.join(out_path, f'{i + 1}.png'), sheet)



extract_frames('y2mate.com - How to play piano part of All Of Me by John Legend_1080pFHR.mp4',
                temp_path='temp2',
                video_start=3*60*1000,
                bpm=126,
                bpb=4)

# generate_music_sheet()



