# ==============================================================================
# MAIN TENNIS ANALYSIS PIPELINE
# ------------------------------------------------------------------------------
# This is the main starting point of the program. It reads a video of a tennis
# match, detects where the players and ball are in every frame using AI (YOLO),
# finds the court lines using a CNN keypoint model, projects everyone's 2D 
# position onto a mini-court map, calculates match statistics (like shot speed,
# player running speed, shot counts, and total distance), and draws all these
# overlays back onto the final output video!
# ==============================================================================

from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy


def main():
    # -------------------------------------------------------------------------
    # STEP 1: READ THE INPUT VIDEO
    # -------------------------------------------------------------------------
    # Load all individual image frames from the input video file.
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # -------------------------------------------------------------------------
    # STEP 2: DETECT PLAYERS AND TENNIS BALL USING AI (YOLO)
    # -------------------------------------------------------------------------
    # Initialize the YOLO AI tracker models for players and tennis ball.
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/last.pt')

    # Detect player bounding boxes for each frame (uses cached stub if enabled for speed).
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    # Detect tennis ball bounding boxes for each frame.
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )
    # Interpolate missing ball positions (fills in gaps when the ball moves too fast for YOLO).
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # -------------------------------------------------------------------------
    # STEP 3: DETECT COURT KEYPOINTS USING PYTORCH CNN MODEL
    # -------------------------------------------------------------------------
    # Predict the 14 reference corners/lines on the tennis court from the first video frame.
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Filter out people in the background (spectators, ball kids) to keep ONLY the 2 active players.
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # -------------------------------------------------------------------------
    # STEP 4: INITIALIZE MINI-COURT PROJECTION MAP
    # -------------------------------------------------------------------------
    # Set up the 2D mini-court graphic that will be drawn in the corner of the output video.
    mini_court = MiniCourt(video_frames[0]) 

    # Identify the exact frame numbers where a player strikes/hits the tennis ball.
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert pixel positions from camera view into real-world meters on the mini-court map.
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, 
        ball_detections,
        court_keypoints
    )

    # -------------------------------------------------------------------------
    # STEP 5: CALCULATE MATCH STATISTICS (SPEED, SHOTS, DISTANCE)
    # -------------------------------------------------------------------------
    # Prepare the initial statistics structure for frame 0.
    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]
    
    # Loop over every detected ball hit/shot event to compute shot speed and opponent running speed.
    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # Video runs at 24 frames per second

        # Calculate real-world distance covered by the ball during the shot (in meters)
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        ) 

        # Convert ball speed to km/h (speed = distance / time * 3.6)
        speed_of_ball_shot = (distance_covered_by_ball_meters / ball_shot_time_in_seconds) * 3.6

        # Determine which player hit the ball (the one closest to the ball position)
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(
            player_positions[player_id],
            ball_mini_court_detections[start_frame][1]
        ))

        # Calculate opponent's running speed during the shot exchange (in km/h)
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        ) 

        speed_of_opponent = (distance_covered_by_opponent_meters / ball_shot_time_in_seconds) * 3.6

        # Record and update the statistics for this frame
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    # Convert statistics list into a Pandas DataFrame and align across every video frame
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # Compute average shot and player speeds (filling zero-division NaNs with 0)
    player_stats_data_df['player_1_average_shot_speed'] = (player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']).fillna(0)
    player_stats_data_df['player_2_average_shot_speed'] = (player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']).fillna(0)
    player_stats_data_df['player_1_average_player_speed'] = (player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']).fillna(0)
    player_stats_data_df['player_2_average_player_speed'] = (player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']).fillna(0)

    # Calculate Total Cumulative Player Distance Covered (in meters) frame-by-frame
    player_1_total_distance = 0.0
    player_2_total_distance = 0.0
    player_1_distances = [0.0]
    player_2_distances = [0.0]

    for frame_idx in range(1, len(player_mini_court_detections)):
        prev_pos = player_mini_court_detections[frame_idx - 1]
        curr_pos = player_mini_court_detections[frame_idx]

        if 1 in prev_pos and 1 in curr_pos:
            dist_px = measure_distance(prev_pos[1], curr_pos[1])
            dist_m = convert_pixel_distance_to_meters(dist_px, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())
            player_1_total_distance += dist_m
        player_1_distances.append(player_1_total_distance)

        if 2 in prev_pos and 2 in curr_pos:
            dist_px = measure_distance(prev_pos[2], curr_pos[2])
            dist_m = convert_pixel_distance_to_meters(dist_px, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())
            player_2_total_distance += dist_m
        player_2_distances.append(player_2_total_distance)

    player_stats_data_df['player_1_total_distance'] = player_1_distances
    player_stats_data_df['player_2_total_distance'] = player_2_distances

    # -------------------------------------------------------------------------
    # STEP 6: DRAW OVERLAYS AND RENDER OUTPUT VIDEO
    # -------------------------------------------------------------------------
    # 1. Draw Player & Ball Bounding Boxes on video frames
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # 2. Draw court keypoints on video frames
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # 3. Draw mini-court graphic and player/ball tracking dots
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))    

    # 4. Draw match statistics overlay box
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # 5. Save a preview screenshot frame
    if len(output_video_frames) > 0:
        screenshot_frame_idx = min(150, len(output_video_frames) - 1)
        cv2.imwrite("output_videos/screenshot.jpeg", output_video_frames[screenshot_frame_idx])

    # 6. Save final output video file (.avi)
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()