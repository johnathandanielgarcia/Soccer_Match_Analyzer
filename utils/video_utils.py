# utilities to read in & save video 
import cv2

def read_video(video_path): # video @ 24fps
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame, = cap.read() # returns flag if frame exists and next frame
        if not ret: # if no frame, video ended
            break
        frames.append(frame) # add frame
    cap.release()
    return frames

def save_video(output_vid_frames, output_vid_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(output_vid_path, fourcc, 24, (output_vid_frames[0].shape[1], output_vid_frames[0].shape[0])) # takes in path, output video type, fps, frame w & h
    height, width = output_vid_frames[0].shape[:2]
    out = cv2.VideoWriter(output_vid_path, fourcc, 24, (width, height))
    for frame in output_vid_frames:
        out.write(frame)
    out.release()