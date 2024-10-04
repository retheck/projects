import cv2
import os
from utils import read_video, save_video
from trackers import Tracker
import sys
sys.path.append(r"C:\Users\Retheck\Desktop\Football_analysis\trackers\development_and_analysis\team_assigner")
from team_assigner import TeamAssigner # type: ignore
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner


def main():
    # Define directories
    input_video_path = 'input_videos/08fd33_4.mp4'
    output_dir = r"C:\Users\Retheck\Desktop\Football_analysis\output_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read video
    video_frames = read_video(input_video_path)
    
    # Initialize tracker
    tracker = Tracker('models/best.pt')

    # Get object tracks from the tracker
    object_tracks = tracker.get_object_tracks(video_frames)
    
    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], object_tracks['players'][0])

    for frame_num, player_track in enumerate(object_tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            object_tracks['players'][frame_num][player_id]['team'] = team
            object_tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Interpolate ball positions
    object_tracks["ball"] = tracker.interpolate_ball_positions(object_tracks["ball"])

    # Assign ball acquisition
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(object_tracks['players']):
        ball_bbox = object_tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            object_tracks['players'][frame_num][assigned_player]['has_ball'] = True

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, object_tracks)

    # Save video
    save_video(output_video_frames, os.path.join(output_dir, 'output_video.avi'))  

if __name__ == '__main__':
    main()
