from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assignment import TeamAssigner
from ball_possession_player import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer

def main():
    # read video
    video_frames = read_video('input_vids/08fd33_4.mp4')


    # initialize tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path = 'stubs/tracks_stubs.pkl')
    
    # get object positions
    tracker.add_position_to_tracks(tracks)

    
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_pos(tracks["ball"])
    
    '''# save cropped image of a player, going to be used to identify teams based on shirt color
        # only need to run once to get picture, uncomment if first time running
    for track_id, player in tracks['players'][0].items():
        bbox =player['bbox']
        frame = video_frames[0]

        # crop bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # save the cropped image
        cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)
        break'''
    
    # assign player team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # assign ball to player w possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(None)  
    team_ball_control = np.array(team_ball_control)
        


    # draw output 
    # draw object tracked
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    

    # save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()