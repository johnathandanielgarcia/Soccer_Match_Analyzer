from ultralytics import YOLO    # creating & training model
import supervision as sv    # for tracking objects
import pickle       # for saving object tracks
import os       # check if stub_path exists
import cv2  # to draw ellipse under players
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox, get_foot_position
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    '''def interpolate_ball_pos(self, ball_pos):
        ball_pos = [x.get(1, {}).get('bbox', []) for x in ball_pos]
        df_ball_pos = pd.DataFrame(ball_pos, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate missing values 
        df_ball_pos = df_ball_pos.interpolate()
        df_ball_pos = df_ball_pos.bfill() # for first frame if missing

        ball_pos = [{1: {"bbox": x}} for x in df_ball_pos.to_numpy().tolist()]

        return ball_pos'''
    

    def interpolate_ball_pos(self, ball_pos):
        # Extract ball positions (bounding boxes) from input
        ball_pos = [x.get(1, {}).get('bbox', []) for x in ball_pos]
        df_ball_pos = pd.DataFrame(ball_pos, columns=['x1', 'y1', 'x2', 'y2'])

        # Detect missing data (all values 0 or NaN) and mark as NaN
        df_ball_pos.replace([0, ''], np.nan, inplace=True)

        # Smoothing/filtering: Apply a smoothing filter if data is noisy
        '''window_size = 5  # This can be tuned based on your data
        poly_order = 2   # Quadratic smoothing
        df_ball_pos['x1'] = savgol_filter(df_ball_pos['x1'], window_size, poly_order)
        df_ball_pos['y1'] = savgol_filter(df_ball_pos['y1'], window_size, poly_order)'''


        # Cubic Spline Interpolation (better for ball movement tracking)
        time_points = np.arange(len(df_ball_pos))  # Time indices

        # Interpolate each column (x1, y1, x2, y2) separately
        for column in df_ball_pos.columns:
            # Check if there's enough non-missing data to interpolate
            if df_ball_pos[column].isna().sum() < len(df_ball_pos) - 1:
                cs = CubicSpline(time_points[~df_ball_pos[column].isna()], df_ball_pos[column].dropna(), bc_type='natural')
                df_ball_pos[column] = cs(time_points)

        # Fill missing values (if any remain at the edges)
        df_ball_pos = df_ball_pos.bfill().ffill()

        # Convert DataFrame back to list of dicts
        ball_pos = [{1: {"bbox": x}} for x in df_ball_pos.to_numpy().tolist()]

        return ball_pos


    def detect_frames(self, frames):
        batch_size = 20 # set batch of frame size 
        detections = [] # empty list for detection 
        for i in range(0, len(frames), batch_size): # predict for each batch in frames
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch # add detection to list
        return detections 

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            'players':[],
            'referees':[],
            'ball':[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert keepers to field player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player'] # 'players' instead of '2' so works if dataset organized differently 

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            '''for frame_num, detection in enumerate(detections):
            # Ensure tracks['players'] has enough frames
                while len(tracks['players']) <= frame_num:
                    tracks['players'].append({})  # Initialize with an empty dictionary for each frame

                while len(tracks['referees']) <= frame_num:
                    tracks['referees'].append({})  # Initialize with an empty dictionary for each frame

                while len(tracks['ball']) <= frame_num:
                    tracks['ball'].append({})  # Initialize with an empty dictionary for each frame'''


            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist() # bbox = bounding box
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                
                '''if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num].setdefault(track_id, {})["bbox"] = bbox
                if cls_id == cls_names_inv['referee']:
                    tracks["referee"][frame_num].setdefault(track_id, {})["bbox"] = bbox'''
                
            for frame_detection in detection_supervision: 
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}  # can hardcode this bc only one ball

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,      # wont draw complete circle, looks better under players
            color=color,
            thickness=2,
            lineType=cv2.LINE_4)
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width/2
        x2_rect = (x_center + rectangle_width/2) 
        y1_rect = (y2 - rectangle_height/2) + 15
        y2_rect = (y2 + rectangle_height/2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -=10

            cv2.putText(frame,
                        f'{track_id}',
                        (int(x1_text), int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2)
            
        return frame

    def draw_triangle(self, frame, bbox, color): # for ball
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4 # transparency
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # get possession numbers
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        
        if team_1_num_frames != 0 or team_2_num_frames != 0:
            team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
            team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)
        else:
            team_1 = 1 # if team_x_num_frames == 0, 0/anything = 0
            team_2 = 1
        

        cv2.putText(frame, f'Team 1 Ball Control: {team_1*100:.2f}%', (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f'Team 2 Ball Control: {team_2*100:.2f}%', (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() # dont want to mess with frames coming in

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    self.draw_triangle(frame, player["bbox"], (0, 0, 255))


            # draw ref
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255), track_id)

            # draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            
            # draw team ball controll
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)


            output_video_frames.append(frame)

        return output_video_frames


