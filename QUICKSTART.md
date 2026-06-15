# Quick Start Guide

Get started with the Enhanced Red Light Violation Detection System in 5 minutes!

## Step 0: Preview Your Video (Optional)

Before processing, you can preview your video:

```bash
python3 pmain1.py --play-video
```

**Controls:**
- **SPACE** - Pause/Resume
- **→/←** - Skip forward/backward
- **S** - Save screenshot
- **Q** - Quit

This helps you identify the best areas to draw detection zones!

## Step 1: Install Dependencies

```bash
# Install in virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python ultralytics numpy pandas cvzone
```

## Step 2: Draw Detection Zones

Run the system with interactive ROI selector:

```bash
python3 pmain1.py --interactive-roi
```

### Drawing Instructions:

1. **The first frame of your video will appear**
2. **Click and drag** with left mouse button to draw detection zones
3. **Draw all areas** where you want to detect violations
4. **Press 'S'** to save and start processing

![ROI Selection Demo](https://via.placeholder.com/800x400.png?text=Draw+Detection+Zones+Here)

**Tips:**
- Draw boxes where vehicles should stop during red light
- You can draw multiple zones for complex intersections
- Right-click to delete a zone if you make a mistake
- Press 'D' to delete the last zone
- Press 'C' to clear all and start over

## Step 3: Process Your Video

After saving ROIs, the system will automatically:
- ✅ Detect vehicles (cars, trucks, buses, motorcycles)
- ✅ Track them across frames
- ✅ Detect traffic light state
- ✅ Record violations when vehicles enter zones during red light
- ✅ Save violation images with evidence

**You'll see a live window showing:**
- Vehicle bounding boxes (green = normal, red = violation)
- Detection zones
- Traffic light state
- Real-time statistics

**Press 'Q' to stop processing anytime**

## Step 4: Check Results

### Violation Images
```
saved_images/2026-01-31/
├── ROI1_ID5_car_14-30-45-123.jpg
├── ROI1_ID8_truck_14-31-02-456.jpg
└── ...
```

Each image shows:
- Vehicle that violated
- Vehicle ID for tracking
- Timestamp of violation
- ROI where violation occurred

### Processed Video
- **File**: `output_video.avi`
- Contains complete processed video with annotations

### Logs
- **File**: `violations.log`
- Detailed log of all violations with timestamps

## Example Output

```
2026-01-31 14:30:45 - INFO - Processing video: tr.mp4
2026-01-31 14:30:46 - INFO - Loaded 2 ROIs from roi_config.json
2026-01-31 14:30:50 - WARNING - VIOLATION: car ID:5 in ROI 1 during RED light
2026-01-31 14:31:02 - WARNING - VIOLATION: truck ID:8 in ROI 1 during RED light
2026-01-31 14:35:20 - INFO - Processed 500 frames...
2026-01-31 14:40:15 - INFO - End of video reached

==================================================
PROCESSING COMPLETE
==================================================
Total Frames Processed: 1247
Total Detections: 3456
Total Violations: 12
Violations by ROI:
  ROI 1: 8 violations
  ROI 2: 4 violations
==================================================
```

## Next Steps

### Run Again with Saved ROIs

Once you've created ROIs, just run:
```bash
python3 pmain1.py
```

The system will use your saved zones automatically.

### Customize Settings

Edit `config.json` to adjust:
- Detection confidence
- Vehicle types to monitor
- Tracking parameters
- Output settings

See [README.md](README.md) for full configuration options.

## Common First-Time Issues

### ❌ "Could not open video file"
**Fix**: Make sure `tr.mp4` exists in the same directory

### ❌ "No ROIs defined"
**Fix**: You must draw at least one detection zone. Press 'S' only after drawing zones.

### ❌ Model downloading takes time
**Fix**: This is normal on first run. YOLOv10 model (~16MB) will be downloaded automatically.

### ❌ Video window not showing
**Fix**: 
1. Make sure you're not running in headless environment
2. Or set `"show_video_window": false` in config.json

### ❌ Too many false violations
**Fix**: 
1. Draw ROIs more precisely (only where vehicles should stop)
2. Increase detection confidence in config.json:
   ```json
   "detection": {
       "confidence_threshold": 0.6  // Increase from 0.5
   }
   ```

## Tips for Best Results

1. **Clear ROI Boundaries**: Draw zones only where vehicles MUST stop during red light

2. **Multiple Zones**: Create separate ROIs for different lanes or directions

3. **Test First**: Process first 100 frames to verify zones work correctly

4. **Adjust as Needed**: You can always re-run with `--interactive-roi` to redraw zones

5. **Check Logs**: If something seems wrong, check `violations.log` for details

## Get Help

For more detailed information:
- See [README.md](README.md) for full documentation
- Check configuration options in `config.json`
- Review violation logs in `violations.log`

## Video Tutorial

1. Start: `python3 pmain1.py --interactive-roi`
2. Draw zones on the intersection
3. Press 'S' to save
4. Watch as violations are detected
5. Check `saved_images/` folder for evidence

That's it! You're ready to detect traffic violations! 🚦🚗

