"""
Advanced Footfall Counter using YOLOv8 + ByteTrack
Author: Abhinandan
Date: October 2024

This script counts people entering/exiting through a defined region using:
- YOLOv8 for real-time person detection
- ByteTrack for robust multi-object tracking
- State machine logic to prevent double counting
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import torch
import sys
import os

# Check if required libraries are installed
try:
    from ultralytics import YOLO
    import supervision as sv
except ImportError as e:
    print(f"Error: Missing required library - {e}")
    print("\nPlease install requirements:")
    print("pip install ultralytics supervision opencv-python")
    sys.exit(1)


class ByteTrackManager:
    """
    Manages ByteTrack tracker for multi-object tracking.
    
    ByteTrack is a state-of-the-art tracker that:
    - Associates detections across frames using motion + appearance
    - Handles occlusions (people temporarily hidden)
    - Maintains unique IDs for each person
    """
    
    def __init__(self, track_activation_threshold=0.25, lost_track_buffer=30):
        """
        Initialize ByteTrack tracker.
        
        Args:
            track_activation_threshold: Confidence threshold to start tracking (0-1)
            lost_track_buffer: Number of frames to keep lost tracks before removing
        """
        # Initialize ByteTrack from supervision library
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer
        )
        
        # Store trajectory history for each tracked person
        # Key: tracker_id, Value: deque of (x, y) centroids
        self.centroids_history = defaultdict(lambda: deque(maxlen=20))
    
    def update(self, detections):
        """
        Update tracker with new detections from current frame.
        
        Args:
            detections: supervision.Detections object with bounding boxes
            
        Returns:
            Updated detections with tracker_id assigned to each detection
        """
        # Update ByteTrack with new detections
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Store centroid history for trajectory visualization
        if tracked_detections.tracker_id is not None:
            for tracker_id, box in zip(tracked_detections.tracker_id, tracked_detections.xyxy):
                # Calculate centroid from bounding box
                cx = (box[0] + box[2]) / 2  # Center X
                cy = (box[1] + box[3]) / 2  # Center Y
                self.centroids_history[int(tracker_id)].append((cx, cy))
        
        return tracked_detections


class FootfallCounter:
    """
    Complete footfall counting system with GPU acceleration.
    
    Features:
    - YOLOv8 detection (GPU-optimized)
    - ByteTrack tracking
    - Entry/Exit counting with state machine
    - Real-time visualization
    - Performance monitoring
    """
    
    def __init__(self, video_path, model_size='s', line_position=0.5, 
                 direction='horizontal', device='auto'):
        """
        Initialize the footfall counter.
        
        Args:
            video_path (str): Path to input video file
            model_size (str): YOLOv8 model size
                - 'n' (nano): Fastest, lowest accuracy (good for testing)
                - 's' (small): Balanced speed/accuracy (RECOMMENDED)
                - 'm' (medium): Better accuracy, slower
            line_position (float): Position of counting line (0.0 to 1.0)
                - 0.5 = middle of frame
            direction (str): Counting line orientation
                - 'horizontal': Line goes left-right (counts up/down movement)
                - 'vertical': Line goes up-down (counts left/right movement)
            device (str): Processing device
                - 'auto': Automatically use GPU if available, else CPU
                - 'gpu': Force GPU (will fail if not available)
                - 'cpu': Force CPU only
        """
        print(f"\n{'='*70}")
        print(f"{'FOOTFALL COUNTER INITIALIZATION':^70}")
        print(f"{'='*70}\n")
        
        self.video_path = video_path
        self.line_position = line_position
        self.direction = direction
        
        # Setup processing device (GPU or CPU)
        self.device = self._setup_device(device)
        
        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Properties:")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total Frames: {self.total_frames}")
        print(f"  Duration: {self.total_frames/self.fps:.1f} seconds\n")
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8-{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        # Move model to GPU if available
        if self.device == 'cuda':
            self.model.to('cuda')
        
        self.model_size = model_size
        print(f"✓ Model loaded successfully\n")
        
        # Initialize ByteTrack tracker
        self.tracker_manager = ByteTrackManager()
        
        # Counting variables
        self.entry_count = 0  # People entering (crossing line downward)
        self.exit_count = 0   # People exiting (crossing line upward)
        
        # Track which side of line each person is on
        # Key: tracker_id, Value: 'above' or 'below'
        self.person_states = {}
        
        # Set of all unique people tracked
        self.tracked_people = set()
        
        # Setup video writer for output
        output_path = 'output_footfall.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                   (self.width, self.height))
        print(f"Output video: {output_path}\n")
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)  # FPS calculation
        self.detection_times = deque(maxlen=30)  # Detection timing
        self.tracking_times = deque(maxlen=30)   # Tracking timing
    
    def _setup_device(self, device_preference):
        """
        Setup CUDA GPU or CPU for processing.
        
        Args:
            device_preference: 'auto', 'gpu', or 'cpu'
            
        Returns:
            'cuda' if GPU is used, 'cpu' otherwise
        """
        if device_preference == 'cpu':
            print("Processing Device: CPU (forced)\n")
            return 'cpu'
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            # Get GPU information
            device_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"Processing Device: GPU")
            print(f"  Name: {device_name}")
            print(f"  Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
            print(f"  Total VRAM: {total_memory:.1f} GB")
            print(f"  CUDA Version: {torch.version.cuda}\n")
            
            return 'cuda'
        else:
            if device_preference == 'gpu':
                print("⚠ Warning: GPU requested but CUDA not available")
                print("Falling back to CPU\n")
            else:
                print("Processing Device: CPU (no GPU detected)\n")
            return 'cpu'
    
    def get_line_coordinates(self):
        """
        Calculate counting line coordinates based on direction.
        
        Returns:
            Tuple of ((x1, y1), (x2, y2)) representing line start and end
        """
        if self.direction == 'horizontal':
            # Horizontal line (left-right across frame)
            y = int(self.height * self.line_position)
            return ((0, y), (self.width, y))
        else:
            # Vertical line (top-bottom across frame)
            x = int(self.width * self.line_position)
            return ((x, 0), (x, self.height))
    
    def get_signed_distance(self, point, line_start):
        """
        Calculate signed distance from point to counting line.
        
        Args:
            point: (x, y) coordinates
            line_start: (x, y) start point of line
            
        Returns:
            Negative = above/left of line
            Positive = below/right of line
        """
        x, y = point
        x1, y1 = line_start
        
        if self.direction == 'horizontal':
            return y - y1  # Distance from horizontal line
        else:
            return x - x1  # Distance from vertical line
    
    def detect_people(self, frame):
        """
        Detect people in frame using YOLOv8.
        
        Args:
            frame: Input frame from video
            
        Returns:
            supervision.Detections object with person detections only
        """
        # Run YOLOv8 inference
        # conf=0.45: Only keep detections with 45%+ confidence
        # verbose=False: Don't print detection info
        results = self.model(frame, conf=0.45, verbose=False, device=self.device)
        
        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter only person class (class_id = 0 in COCO dataset)
        person_detections = detections[detections.class_id == 0]
        
        return person_detections
    
    def update_counts(self, tracked_detections, line_start):
        """
        Update entry/exit counts based on line crossings.
        
        Uses state machine logic:
        - Track which side of line each person is on
        - When person crosses line, increment appropriate counter
        - State changes only once per crossing (prevents double counting)
        
        Args:
            tracked_detections: Detections with tracker IDs
            line_start: Start point of counting line
        """
        if tracked_detections.tracker_id is None:
            return
        
        for tracker_id, box in zip(tracked_detections.tracker_id, 
                                   tracked_detections.xyxy):
            tracker_id = int(tracker_id)
            self.tracked_people.add(tracker_id)
            
            # Calculate centroid of bounding box
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            centroid = (cx, cy)
            
            # Get signed distance to line
            distance = self.get_signed_distance(centroid, line_start)
            
            # Determine current state (which side of line)
            current_state = 'above' if distance < 0 else 'below'
            
            # Check if this is first time seeing this person
            if tracker_id not in self.person_states:
                # Initialize state
                self.person_states[tracker_id] = current_state
            else:
                # Get previous state
                previous_state = self.person_states[tracker_id]
                
                # Check if person crossed the line
                if previous_state != current_state:
                    # Line crossing detected!
                    if current_state == 'below':
                        # Crossed from above to below = ENTRY
                        self.entry_count += 1
                        print(f"✓ ENTRY detected (ID: {tracker_id}) | Total Entries: {self.entry_count}")
                    else:
                        # Crossed from below to above = EXIT
                        self.exit_count += 1
                        print(f"✓ EXIT detected (ID: {tracker_id}) | Total Exits: {self.exit_count}")
                    
                    # Update state
                    self.person_states[tracker_id] = current_state
    
    def draw_annotations(self, frame, tracked_detections, line_coords):
        """
        Draw all visual annotations on frame.
        
        Includes:
        - Counting line
        - Bounding boxes around people
        - Tracker IDs
        - Trajectory trails
        - Entry/Exit counts
        - Performance stats
        
        Args:
            frame: Input frame
            tracked_detections: Detections with tracker IDs
            line_coords: ((x1, y1), (x2, y2)) line coordinates
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw counting line
        cv2.line(annotated, line_coords[0], line_coords[1], 
                (0, 255, 0), 3, cv2.LINE_AA)
        
        # Add line label
        label_pos = (line_coords[0][0] + 10, line_coords[0][1] - 10)
        cv2.putText(annotated, "COUNTING LINE", label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw bounding boxes using supervision
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            color=sv.Color.from_hex("#00FF00")
        )
        annotated = box_annotator.annotate(
            scene=annotated, 
            detections=tracked_detections
        )
        
        # Draw tracker IDs
        if tracked_detections.tracker_id is not None:
            label_annotator = sv.LabelAnnotator(
                text_scale=0.6,
                text_thickness=2
            )
            labels = [f"ID: {int(tid)}" for tid in tracked_detections.tracker_id]
            annotated = label_annotator.annotate(
                scene=annotated,
                detections=tracked_detections,
                labels=labels
            )
        
        # Draw trajectory trails
        for tracker_id, history in self.tracker_manager.centroids_history.items():
            if len(history) > 1:
                points = list(history)
                for i in range(1, len(points)):
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    # Draw line with fading effect
                    alpha = i / len(points)  # Fade older points
                    color = (int(200 * alpha), int(100 * alpha), 255)
                    cv2.line(annotated, pt1, pt2, color, 2)
        
        # Draw statistics panel
        self._draw_stats_panel(annotated)
        
        return annotated
    
    def _draw_stats_panel(self, frame):
        """
        Draw statistics panel on frame.
        
        Shows:
        - Entry count
        - Exit count
        - Net footfall
        - FPS
        """
        # Dark semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Border
        cv2.rectangle(frame, (10, 10), (400, 180), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "FOOTFALL STATISTICS", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Entries (Green)
        cv2.putText(frame, f"ENTRIES: {self.entry_count}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Exits (Red)
        cv2.putText(frame, f"EXITS: {self.exit_count}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Net footfall (Yellow)
        net = self.entry_count - self.exit_count
        cv2.putText(frame, f"NET: {net}", (20, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # FPS (White)
        avg_fps = np.mean(list(self.frame_times)) if self.frame_times else 0
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (250, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def process_video(self, display=True):
        """
        Process entire video frame by frame.
        
        Args:
            display (bool): Show real-time window if True
        """
        frame_count = 0
        line_start, line_end = self.get_line_coordinates()
        
        print(f"{'='*70}")
        print(f"Processing video...")
        print(f"{'='*70}\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_start = datetime.now()
                
                # STEP 1: Detect people
                det_start = datetime.now()
                person_detections = self.detect_people(frame)
                det_time = (datetime.now() - det_start).total_seconds()
                self.detection_times.append(det_time)
                
                # STEP 2: Track people across frames
                track_start = datetime.now()
                tracked_detections = self.tracker_manager.update(person_detections)
                track_time = (datetime.now() - track_start).total_seconds()
                self.tracking_times.append(track_time)
                
                # STEP 3: Update entry/exit counts
                self.update_counts(tracked_detections, line_start)
                
                # STEP 4: Draw annotations
                annotated_frame = self.draw_annotations(
                    frame, tracked_detections, (line_start, line_end)
                )
                
                # Add frame counter
                progress_text = f"Frame: {frame_count}/{self.total_frames}"
                cv2.putText(annotated_frame, progress_text,
                           (self.width - 280, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                # Write to output video
                self.out.write(annotated_frame)
                
                # Display real-time window
                if display:
                    cv2.imshow('Footfall Counter - Press Q to quit', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing stopped by user")
                        break
                
                # Calculate FPS
                frame_time = (datetime.now() - frame_start).total_seconds()
                if frame_time > 0:
                    self.frame_times.append(1 / frame_time)
                
                frame_count += 1
                
                # Progress update every 30 frames
                if frame_count % 30 == 0:
                    progress = (frame_count / self.total_frames) * 100
                    avg_fps = np.mean(list(self.frame_times))
                    print(f"Progress: {progress:.1f}% | "
                          f"FPS: {avg_fps:.1f} | "
                          f"Entries: {self.entry_count} | "
                          f"Exits: {self.exit_count}")
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            self.cleanup()
            self.generate_report()
    
    def generate_report(self):
        """Generate and print final report with statistics."""
        avg_fps = np.mean(list(self.frame_times)) if self.frame_times else 0
        avg_det = np.mean(list(self.detection_times)) * 1000 if self.detection_times else 0
        avg_track = np.mean(list(self.tracking_times)) * 1000 if self.tracking_times else 0
        
        print(f"\n{'='*70}")
        print(f"{'FINAL REPORT':^70}")
        print(f"{'='*70}\n")
        
        print("📊 COUNTING RESULTS:")
        print(f"  Total Entries:        {self.entry_count}")
        print(f"  Total Exits:          {self.exit_count}")
        print(f"  Net Footfall:         {self.entry_count - self.exit_count}")
        print(f"  Unique People Tracked: {len(self.tracked_people)}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  Average FPS:          {avg_fps:.2f}")
        print(f"  Detection Time:       {avg_det:.2f}ms per frame")
        print(f"  Tracking Time:        {avg_track:.2f}ms per frame")
        
        print(f"\n🎯 CONFIGURATION:")
        print(f"  Detection Model:      YOLOv8-{self.model_size}")
        print(f"  Tracking Algorithm:   ByteTrack")
        print(f"  Processing Device:    {self.device.upper()}")
        print(f"  Counting Line:        {self.direction} at {self.line_position*100:.0f}%")
        
        print(f"\n📁 OUTPUT:")
        print(f"  Video Saved:          output_footfall.mp4")
        print(f"\n{'='*70}\n")
    
    def cleanup(self):
        """Release all resources."""
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        # Clear GPU memory if used
        if self.device == 'cuda':
            torch.cuda.empty_cache()


def main():
    """Main function to run the footfall counter."""
    
    # Use relative path for video
    VIDEO_PATH = "input_video.mp4"  
    
    print("\nChecking requirements...")
    
    # Verify video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"\n❌ Error: Video file not found at {VIDEO_PATH}")
        print("Please ensure your video file:")
        print("1. Is named 'input_video.mp4'")
        print("2. Is in the same folder as this script")
        return
        
    print("✓ Found video file")
    
    try:
        # Create counter instance
        counter = FootfallCounter(
            video_path=VIDEO_PATH,
            model_size='s',
            line_position=0.5,
            direction='horizontal',
            device='auto'
        )
        
        # Process video
        counter.process_video(display=True)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()