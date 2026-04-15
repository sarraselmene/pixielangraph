import os
import argparse
import pandas as pd
import numpy as np
from collections import deque
from upper_body_classifier import UpperBodyBehaviorClassifier

def process_body_data(input_csv, summary_out, raw_out, fps=30.0):
    """
    Reads the raw body geometry from YOLO/Mediapipe, runs it through
    the Upper Body Behavior Classifier, and produces a temporal summary.
    """
    print(f"\n[Body Labeling] Reading raw body geometries from {input_csv}...")
    
    # 1. Load CSV
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"[ERROR] Body input CSV not found: {input_csv}")
        return

    # Check for empty dataframe
    if df.empty:
        print(f"[WARN] Input CSV is empty. Creating empty outputs.")
        pd.DataFrame(columns=["frame_id", "track_id", "posture", "posture_confidence", "action", "action_confidence"]).to_csv(raw_out, index=False)
        pd.DataFrame(columns=["track_id", "behaviour", "start_frame", "end_frame", "duration_frames", "confidence_avg"]).to_csv(summary_out, index=False)
        return

    # Pivot the data so each row is a frame+track_id, and columns are the keypoints
    # The landmark indices mirror YOLO/MediaPipe defaults (e.g. 0=nose, 5=L shoulder, 6=R shoulder, 9=L wrist, 10=R wrist)
    kp_map = {
        0: 'nose',
        5: 'left_shoulder',
        6: 'right_shoulder',
        9: 'left_wrist',
        10: 'right_wrist'
    }
    
    # Filter only relevant upper body keypoints to speed up processing
    relevant_idx = list(kp_map.keys())
    df_filtered = df[df['landmark_idx'].isin(relevant_idx)].copy()
    
    # Sort by frame
    df_filtered.sort_values(by=['frame_id', 'track_id'], inplace=True)

    tracks = df_filtered['track_id'].unique()
    
    # We will instantiate a Tracker state machine per person
    classifiers = {tid: UpperBodyBehaviorClassifier(fps=fps, window_size_frames=30) for tid in tracks}
    
    raw_results = []
    
    print(f"[Body Labeling] Processing {len(tracks)} tracked individuals over sequence...")
    
    # Group by frame and track
    grouped = df_filtered.groupby(['frame_id', 'track_id'])
    
    for (frame_id, track_id), group in grouped:
        classifier = classifiers[track_id]
        
        # Build Keypoints Dictionary
        pts = {}
        for _, row in group.iterrows():
            lm_id = row['landmark_idx']
            if pd.notna(row['x']) and pd.notna(row['y']):
                pts[kp_map[lm_id]] = (row['x'], row['y'])
                
        # Global Desk and Shoulder Width
        # Extract global desk
        global_desk_y = group['global_desk_y'].median() if pd.notna(group['global_desk_y'].median()) else None
        
        # Calculate Shoulder width for scale normalization
        sw = 100.0 # fallback
        if 'left_shoulder' in pts and 'right_shoulder' in pts:
            # L2 distance
            sw = np.linalg.norm(np.array(pts['left_shoulder']) - np.array(pts['right_shoulder']))
        
        # Prevent division by zero
        sw = max(sw, 1.0)
        
        # Update State Machine
        state = classifier.update(pts, global_desk_y, sw)
        
        raw_results.append({
            "frame_id": frame_id,
            "track_id": track_id,
            "posture": state.posture,
            "posture_confidence": state.posture_conf,
            "action": state.action,
            "action_confidence": state.action_conf
        })

    # Convert raw to dataframe and save
    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(raw_out, index=False)
    
    # ---------------------------------------------------------
    # GENERATE BEHAVIORAL EPISODE SUMMARY (Squashing)
    # ---------------------------------------------------------
    print(f"[Body Labeling] Aggregating temporal episodes...")
    episodes = []
    
    for track_id in tracks:
        track_data = raw_df[raw_df['track_id'] == track_id].sort_values('frame_id')
        
        current_beh = None
        current_start = None
        current_end = None
        confs = []
        
        for _, row in track_data.iterrows():
            fid = row['frame_id']
            # Determine dominant behavior (Action > Posture)
            beh = row['action'] if row['action'] != "none" else row['posture']
            conf = row['action_confidence'] if row['action'] != "none" else row['posture_confidence']
            
            if current_beh is None:
                current_beh = beh
                current_start = fid
                current_end = fid
                confs.append(conf)
            elif beh == current_beh and (fid - current_end) <= 5: 
                # Allowed frame gap of 5 to bridge temporary stutters
                current_end = fid
                confs.append(conf)
            else:
                # Save previous episode
                episodes.append({
                    "track_id": track_id,
                    "behaviour": current_beh,
                    "start_frame": current_start,
                    "end_frame": current_end,
                    "duration_frames": (current_end - current_start) + 1,
                    "confidence_avg": np.mean(confs)
                })
                # Reset
                current_beh = beh
                current_start = fid
                current_end = fid
                confs = [conf]
                
        # Final flush for track
        if current_beh is not None:
             episodes.append({
                 "track_id": track_id,
                 "behaviour": current_beh,
                 "start_frame": current_start,
                 "end_frame": current_end,
                 "duration_frames": (current_end - current_start) + 1,
                 "confidence_avg": np.mean(confs)
             })
             
    # Save Summary
    sum_df = pd.DataFrame(episodes)
    sum_df.to_csv(summary_out, index=False)
    print(f"[Body Labeling] ✅ Done. Generated {len(episodes)} discrete behavioral episodes.")
    print(f"    Raw Out : {raw_out}")
    print(f"    Sum Out : {summary_out}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upper-Body Behavioral Inference")
    parser.add_argument("--body", required=True, help="Path to raw_body_multi.csv")
    parser.add_argument("--sum_out", required=True, help="Output path for behaviour_summary.csv")
    parser.add_argument("--raw_out", required=True, help="Output path for behaviour_raw_frames.csv")
    args = parser.parse_args()
    
    process_body_data(args.body, args.sum_out, args.raw_out)
