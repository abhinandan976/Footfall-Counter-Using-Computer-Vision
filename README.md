# AI Footfall Counter using Computer Vision

**Author:** Abhinandan  
**Date:** October 2024  
**Assignment:** Computer Vision-based Footfall Counter  

---

## Project Overview

This project implements a real-time footfall counter that tracks and counts people entering and exiting through a defined region in video footage. The system uses YOLOv8 for detection and ByteTrack for tracking to achieve robust performance in challenging scenarios with occlusions and overlapping people.

### Main Features

- Real-time person detection using YOLOv8
- Robust multi-object tracking with ByteTrack algorithm
- Accurate entry/exit counting using state machine logic
- GPU acceleration support for faster processing
- Visual trajectory display showing movement paths
- Live statistics dashboard with counts and performance metrics
- High-quality annotated output video

---

## Technical Approach

### Detection System

I chose YOLOv8 (You Only Look Once version 8) as the detection model for this project. YOLOv8 is currently one of the best real-time object detection models available, offering significant improvements over previous versions.

**Why YOLOv8:**
- Faster inference speed compared to YOLOv5 (approximately 37% improvement)
- Better accuracy on person detection (over 97% on COCO dataset)
- Optimized specifically for real-time applications
- Built-in support for GPU acceleration

**Model Configuration:**
- Used YOLOv8-nano variant (yolov8n.pt) which prioritizes speed over accuracy
- Set confidence threshold at 0.4 to filter out weak detections
- Filtered only the 'person' class (class_id=0 from COCO dataset)

The detection pipeline processes each video frame through the YOLO network, which outputs bounding boxes around detected people along with confidence scores. I chose the nano model for this project because it provides the best balance of speed and accuracy for real-time pedestrian counting applications.

### Tracking Implementation

For tracking, I implemented ByteTrack algorithm which is currently state-of-the-art for multi-object tracking. ByteTrack maintains consistent IDs for each person across video frames even when they are temporarily occluded or move close to each other.

**How ByteTrack Works:**
- Combines motion prediction using Kalman filtering with appearance matching
- Handles both high-confidence and low-confidence detections intelligently
- Can recover lost tracks after occlusions lasting up to 30 frames
- Assigns and maintains unique IDs for each person throughout the video

**Tracking Parameters:**
- track_activation_threshold: 0.25 (minimum confidence needed to start tracking)
- lost_track_buffer: 30 (number of frames before removing a lost track)

I chose ByteTrack over other options like SORT or DeepSORT because:
- Better performance in crowded scenes with occlusions
- Lower rate of ID switching between people
- Faster processing since it doesn't require deep feature extraction
- Proven performance on standard MOT benchmarks

### Counting Logic

The counting system uses a state machine approach to accurately count entries and exits while preventing double counting. This was one of the most critical parts of the implementation.

**How the Counting Works:**

First, I define a virtual counting line positioned at 50% of the frame height (this is configurable). For each tracked person, the system calculates the centroid (center point) of their bounding box and determines which side of the line they are on.

The state machine works with two states:
- 'above': person's centroid is above the counting line
- 'below': person's centroid is below the counting line

**State Transition Logic:**
```
When a person moves from 'above' to 'below' → count as ENTRY
When a person moves from 'below' to 'above' → count as EXIT
```

**Implementation Details:**

For each tracked person in every frame:
1. Calculate their current position relative to the line
2. If this is the first time seeing this person, record their initial state
3. If their state has changed from the previous frame:
   - If changed to 'below': increment entry count
   - If changed to 'above': increment exit count
   - Update their stored state

This approach has several advantages:
- Completely eliminates double counting because state only changes once per crossing
- Handles bidirectional movement naturally
- Works with any line orientation (horizontal or vertical)
- Robust to temporary tracking failures since state is preserved

### Performance Optimizations

The system includes GPU acceleration support which approximately doubles the processing speed on NVIDIA GPUs. The implementation automatically detects if CUDA is available and uses the GPU for inference if possible.

**Memory Management:**
- Limited trajectory history to last 20 points per person to prevent memory buildup
- Frame buffer size capped at 30 frames for FPS calculations
- Automatic GPU memory clearing after processing completes

On my test system (NVIDIA RTX 2050), the GPU-accelerated version processes 1080p video at 60-80 frames per second, compared to 30-40 FPS on CPU only.

---

## Video Source

For testing and development, I used a people counting surveillance video that I found on YouTube.

**Test Video Details:**
- Source: YouTube surveillance video
- Link: https://www.youtube.com/watch?v=YzcawvDGe4Y
- Title: People counting video footage
- Resolution: 1280x720 (HD)
- Frame rate: 25 FPS
- Content: Pedestrian walkway with people walking in both directions

**Why I chose this video:**
- Specifically designed for testing counting systems
- Contains clear bidirectional movement (people walking both ways)
- Good camera angle from above showing full body of people
- Decent video quality for accurate detection
- Realistic crowd density with occasional overlapping people
- Represents actual surveillance camera scenario

**Video Characteristics:**
The video shows a pedestrian corridor or walkway filmed from an overhead angle. People walk in both directions (towards and away from the camera), which is perfect for testing entry/exit counting logic. The overhead perspective provides clear visibility of individuals and reduces occlusion issues compared to side-angle videos.

**Alternative sources you can use:**
- Pexels website has free stock footage of people walking
- MOT Challenge dataset provides annotated videos for testing
- You can record your own video using a smartphone

---

## Setup Instructions

### System Requirements

**Minimum Hardware:**
- Processor: Intel Core i5 or equivalent
- RAM: 8GB (16GB recommended for better performance)
- Storage: 2GB free space for models and dependencies
- GPU: Optional but recommended - NVIDIA GPU with CUDA support (RTX 2050 or higher with 4GB+ VRAM)

**Software Requirements:**
- Operating System: Windows 10/11, Linux (Ubuntu 20.04 or later), or macOS
- Python: Version 3.8, 3.9, 3.10, or 3.11 (I used Python 3.11)
- CUDA Toolkit: Version 12.1 or later (only if using GPU acceleration)

### Installation Process

**Step 1: Clone or Download the Project**

```bash
git clone https://github.com/yourusername/footfall-counter.git
cd footfall-counter
```

**Step 2: Set Up Python Virtual Environment**

For Windows:
```bash
python -m venv cuda_env
cuda_env\Scripts\activate
```

For Linux/macOS:
```bash
python3 -m venv cuda_env
source cuda_env/bin/activate
```

**Step 3: Install Required Packages**

If you want GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install other dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ultralytics==8.0.196
pip install supervision==0.26.1
pip install opencv-python==4.12.0
pip install numpy==2.3.4
```

**Step 4: Get a Test Video**

You can download the test video I used:
```bash
pip install yt-dlp
yt-dlp -f "best[height<=720]" -o "input_video.mp4" "https://www.youtube.com/watch?v=YzcawvDGe4Y"
```

Or place your own video file named 'input_video.mp4' in the project folder.

**Step 5: Verify Everything Works**

Run the diagnostic script:
```bash
python check.py
```

You should see output confirming all packages are installed and your video file is detected.

---

## How to Use

### Basic Usage

Simply run the main script:
```bash
python footfall_counter.py
```

The program will:
1. Load the YOLOv8 model (downloads automatically on first run, about 6MB for nano model)
2. Open your input video
3. Show a real-time window with detections and counts
4. Save the annotated output as 'output.mp4'
5. Print final statistics when done

### Configuration Options

You can modify settings by editing the main() function in footfall_counter.py:

```python
VIDEO_PATH = "input_video.mp4"    # Change to your video filename
MODEL_SIZE = 'n'                   # Options: 'n' (fast), 's' (balanced), 'm' (accurate)
LINE_POSITION = 0.5                # Position of counting line (0.0 to 1.0)
DIRECTION = 'horizontal'           # Line orientation: 'horizontal' or 'vertical'
DEVICE = 'auto'                    # Processing device: 'auto', 'gpu', or 'cpu'
```

### Understanding the Parameters

**MODEL_SIZE:**
- 'n' (nano): Fastest processing, around 80+ FPS, good for real-time applications (this is what I used)
- 's' (small): Balanced option, 60-70 FPS, slightly better accuracy
- 'm' (medium): Best accuracy, 40-50 FPS, use if accuracy is more important than speed

**LINE_POSITION:**
- 0.0 means top of frame
- 0.5 means middle of frame (default)
- 1.0 means bottom of frame
- Adjust based on where people cross in your video

**DIRECTION:**
- 'horizontal': Draws line left-to-right, counts people moving up/down
- 'vertical': Draws line top-to-bottom, counts people moving left/right

---

## Output and Results

### What You'll See in the Console

When you run the program, you'll see output like this:

```
======================================================================
           FOOTFALL COUNTER INITIALIZATION
======================================================================

Processing Device: GPU
  Name: NVIDIA GeForce RTX 2050
  Compute Capability: 8.6
  Total VRAM: 4.0 GB
  CUDA Version: 12.1

Video Properties:
  Resolution: 1280x720
  FPS: 25
  Total Frames: 3494
  Duration: 139.8 seconds

Loading YOLOv8-n model...
Model loaded successfully

Output video: output.mp4

======================================================================
Processing video...
======================================================================

ENTRY detected (ID: 1) | Total Entries: 1
ENTRY detected (ID: 2) | Total Entries: 2
Progress: 2.2% | FPS: 78.5 | Entries: 2 | Exits: 0
EXIT detected (ID: 1) | Total Exits: 1
Progress: 4.4% | FPS: 80.2 | Entries: 3 | Exits: 1
...

======================================================================
                           FINAL REPORT
======================================================================

COUNTING RESULTS:
  Total Entries:        73
  Total Exits:          78
  Net Footfall:         -5
  Unique People Tracked: 89

PERFORMANCE METRICS:
  Average FPS:          79.34
  Detection Time:       6.8ms per frame
  Tracking Time:        1.2ms per frame

CONFIGURATION:
  Detection Model:      YOLOv8-n
  Tracking Algorithm:   ByteTrack
  Processing Device:    CUDA
  Counting Line:        horizontal at 50%

OUTPUT:
  Video Saved:          output.mp4

======================================================================
```

### Output Video Contents

The generated output video includes several visual elements:

1. **Counting Line** - A green line showing where entries/exits are detected
2. **Bounding Boxes** - Green rectangles around each detected person
3. **Tracker IDs** - Labels showing unique ID for each person (e.g., "ID: 1")
4. **Trajectory Trails** - Colored lines showing recent movement paths
5. **Statistics Panel** - Box in top-left showing:
   - Current entry count
   - Current exit count
   - Net footfall (entries minus exits)
   - Processing speed in FPS
6. **Frame Counter** - Shows current frame number and total frames

### My Test Results

I tested the system on a pedestrian counting video from YouTube and achieved the following results.

**Processing Configuration:**
- GPU: NVIDIA GeForce RTX 2050
- Model: YOLOv8-nano (yolov8n.pt)
- Processing Device: CUDA-enabled GPU
- Python Version: 3.11.0
- CUDA Version: 12.1

**Video Properties:**
- Source: https://www.youtube.com/watch?v=YzcawvDGe4Y
- Resolution: 1280x720 (HD)
- Total Frames Processed: 3494 frames
- Estimated Duration: ~140 seconds (based on 25 FPS)
- Video Type: Overhead pedestrian walkway surveillance

**Counting Results:**
- Total Entries: 73 people
- Total Exits: 78 people
- Net Footfall: -5 (5 more exits than entries, indicating more people left than entered)
- Processing: Successfully processed all 3494 frames

**Observations:**

The system successfully processed the entire video and maintained stable counting throughout. The overhead camera angle in this video provided excellent visibility for detection and tracking.

The results show 73 entries and 78 exits, with a net footfall of -5. This means slightly more people exited the monitored area than entered during the video duration, which makes sense for pedestrian flow in a public walkway where people are constantly moving in both directions.

Some key observations from the processing:
- The YOLOv8-nano model performed well for this use case, successfully detecting people throughout the video
- ByteTrack tracking maintained consistent IDs even in crowded sections
- The state machine counting logic prevented double counting effectively
- GPU acceleration enabled smooth real-time processing

The counting line was positioned horizontally at 50% of frame height, which worked well given the overhead camera angle. People crossing the line in the downward direction were counted as entries, while those crossing upward were counted as exits.

**Validation:**
I manually verified counts in several sections of the video by watching the output. The automated counts matched my manual observations, confirming the system's accuracy for this surveillance scenario.

---

## Project Structure

The project files are organized as follows:

```
footfall-counter/
├── cuda_env/                     Python virtual environment folder
├── footfall_counter.py           Main implementation file (complete code)
├── input_video.mp4               Input test video
├── output.mp4                    Generated output with annotations
└── yolov8n.pt                    YOLOv8 nano model weights
```

Note: The virtual environment folder (cuda_env) and model weights (yolov8n.pt) should not be included when sharing the project. The model will download automatically on first run.

### Code Architecture

The main implementation file contains three key components:

**ByteTrackManager Class:**
- Manages the ByteTrack tracking algorithm
- Stores trajectory history for each tracked person
- Updates tracker with new detections each frame

**FootfallCounter Class:**
- Main class that orchestrates everything
- Handles video input/output
- Runs detection and tracking
- Implements counting logic
- Draws annotations and statistics
- Manages GPU/CPU processing

**Main Function:**
- Entry point of the program
- Sets configuration parameters
- Creates counter instance and processes video

---

## Testing and Edge Cases

During development, I tested the system with various challenging scenarios:

**Occlusions:** When one person temporarily blocks another, ByteTrack maintains the correct IDs and resumes tracking when they become visible again.

**Overlapping People:** When multiple people are close together, the detection system can still identify individuals, and tracking keeps their IDs separate.

**Fast Movement:** Quick movements across the counting line are captured correctly because the state machine logic processes every frame.

**Re-entries:** If the same person enters, exits, and re-enters, each crossing is counted separately which is the correct behavior.

**Partial Views:** People partially in frame are still detected if enough of their body is visible.

**Lighting Variations:** The system handles shadows and bright spots reasonably well thanks to YOLO's training on diverse data.

---

## Challenges and Solutions

**Challenge 1: Double Counting**
Initially, I noticed people getting counted multiple times as they crossed the line. I solved this by implementing the state machine approach where counts only increment when state changes, not on every frame.

**Challenge 2: ID Switching**
In crowded scenes, tracker IDs would sometimes switch between people. Using ByteTrack instead of simpler tracking algorithms significantly reduced this problem.

**Challenge 3: GPU Memory**
Processing high-resolution videos caused GPU memory issues. I added automatic memory cleanup and optimized the trajectory storage to use less memory.

**Challenge 4: Processing Speed**
Original implementation was too slow for real-time use. Adding GPU support and optimizing the detection pipeline improved speed by about 2x.

---

## Possible Improvements

While the current system works well, there are several enhancements I considered for future versions:

- **Webcam Support:** Add ability to process live webcam feed instead of just video files
- **Multiple Counting Lines:** Support for counting in different zones simultaneously
- **Heatmap Generation:** Visualize where people spend most time in the frame
- **API Interface:** Create a REST API using Flask or FastAPI for easier integration
- **Database Logging:** Store counting data in database for historical analysis
- **Alert System:** Send notifications when counts exceed certain thresholds

---

## Dependencies

This project uses the following Python packages:

**Core Dependencies:**
- torch 2.5.1 (PyTorch deep learning framework)
- torchvision 0.15.2 (Computer vision utilities)
- ultralytics 8.0.196 (YOLOv8 implementation)
- supervision 0.26.1 (Tracking and visualization tools)
- opencv-python 4.12.0 (Image and video processing)
- numpy 2.3.4 (Numerical operations)

**Optional:**
- yt-dlp (For downloading test videos from YouTube)

All dependencies can be installed using the provided requirements.txt file.

---

## Troubleshooting

**Problem: "CUDA not available" message**

This means PyTorch cannot detect your GPU. Solutions:
1. Make sure you have NVIDIA GPU with updated drivers
2. Install CUDA Toolkit version 12.1 or later
3. Reinstall PyTorch with CUDA support using the command provided in setup instructions
4. Verify installation: python -c "import torch; print(torch.cuda.is_available())"

**Problem: "Cannot open video file"**

Check these things:
1. Make sure the video file exists in the correct location
2. Verify filename is exactly "input_video.mp4"
3. Try re-encoding the video if it uses an unusual codec
4. Check if the file is corrupted by playing it in a media player

**Problem: Low FPS or slow processing**

Try these solutions:
1. Use smaller model (MODEL_SIZE = 'n')
2. Reduce input video resolution
3. Make sure GPU is being used (check initialization output)
4. Close other applications using GPU
5. Process smaller section of video for testing

**Problem: Inaccurate counts**

Adjust these parameters:
1. Move counting line position (LINE_POSITION)
2. Lower confidence threshold in detect_people() method
3. Increase lost_track_buffer in ByteTrackManager
4. Try different model size for better detection

---

## References

**Research Papers:**
- YOLOv8 Documentation: Ultralytics YOLOv8 (2023)
- ByteTrack: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" by Zhang et al., ECCV 2022
- COCO Dataset: "Microsoft COCO: Common Objects in Context" by Lin et al., 2014

**Libraries Used:**
- Ultralytics: https://github.com/ultralytics/ultralytics
- Supervision: https://github.com/roboflow/supervision
- OpenCV: https://opencv.org/
- PyTorch: https://pytorch.org/

**Learning Resources:**
- YOLOv8 official documentation
- ByteTrack GitHub repository
- MOT Challenge benchmark website

---

## Contact Information

**Author:** Abhinandan  
**Email:** [abhinandanvanajol8@gmail.com]  
**GitHub:** https://github.com/abhinandan976/Footfall-Counter-Using-Computer-Vision

For questions or issues with this project, you can:
- Open an issue on the GitHub repository
- Contact me via email
- Check the troubleshooting section above

---

## License and Acknowledgments

This project was created as part of an AI assignment for educational purposes.

**Acknowledgments:**
- Ultralytics team for YOLOv8 implementation
- ByteTrack authors for the tracking algorithm
- Roboflow for the Supervision library
- COCO dataset contributors
- Open-source computer vision community

**Model Licenses:**
- YOLOv8 is licensed under AGPL-3.0
- COCO dataset uses Creative Commons Attribution 4.0 License

---

Last Updated: October 2024  
Version: 1.0  
Python Version: 3.11.0
