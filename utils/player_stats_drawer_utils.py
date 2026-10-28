# ==============================================================================
# PLAYER STATS DRAWER OVERLAY
# ------------------------------------------------------------------------------
# This file renders the semi-transparent black statistics box in the right side
# of the output video. It reads the match stats (shot count, ball speed, player 
# speed, average speeds, and total running distance) for each video frame and 
# draws formatted text headers using OpenCV.
# ==============================================================================

import numpy as np
import cv2

def draw_player_stats(output_video_frames, player_stats):
    """
    Renders a semi-transparent statistics table overlay on each frame of the video.
    """
    for index, row in player_stats.iterrows():
        # Extract individual metric values for the current frame
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        player_1_shots = int(row.get('player_1_number_of_shots', 0))
        player_2_shots = int(row.get('player_2_number_of_shots', 0))

        player_1_dist = row.get('player_1_total_distance', 0.0)
        player_2_dist = row.get('player_2_total_distance', 0.0)

        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        # Define overlay dimensions (width = 350px, height = 290px)
        width = 350
        height = 290

        # Position box 400px from right edge, starting at Y=575 to leave 25px gap below mini-court
        start_x = frame.shape[1] - 400
        start_y = 575
        end_x = start_x + width
        end_y = start_y + height

        # Draw semi-transparent black background rectangle (50% opacity)
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        # Render Table Header ("Player 1" vs "Player 2")
        text = "     Player 1     Player 2"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 80, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Render Row 1: Shot Count (Yellow)
        text = "Shot Count"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shots} shots      {player_2_shots} shots"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Render Row 2: Shot Speed (km/h)
        text = "Shot Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Render Row 3: Player Movement Speed (km/h)
        text = "Player Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Render Row 4: Average Shot Speed (km/h)
        text = "avg. S. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Render Row 5: Average Player Speed (km/h)
        text = "avg. P. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Render Row 6: Total Cumulative Distance (Green)
        text = "Total Dist."
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_dist:.1f} m        {player_2_dist:.1f} m"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output_video_frames
