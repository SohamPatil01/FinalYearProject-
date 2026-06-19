import cv2
import json
import numpy as np
import os


class ROISelector:
    """Interactive ROI drawing. Use mode='violation' for vehicle zones, mode='signal' for one traffic-light box."""

    def __init__(self, video_path, config_file='roi_config.json', mode='violation'):
        self.video_path = video_path
        self.config_file = config_file
        self.mode = mode  # 'violation' | 'signal'
        self.rois = []
        self.current_roi = None
        self.drawing = False
        self.frame = None
        self.display_frame = None
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing ROIs"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_roi = [x, y, x, y]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_roi[2] = x
                self.current_roi[3] = y
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_roi[2] = x
            self.current_roi[3] = y
            
            # Normalize coordinates (ensure x1 < x2 and y1 < y2)
            x1, y1, x2, y2 = self.current_roi
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Only add if the ROI has a minimum size
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                if self.mode == 'signal':
                    self.rois = [[x1, y1, x2, y2]]
                    print(f"Signal ROI set: ({x1}, {y1}) to ({x2}, {y2})")
                else:
                    self.rois.append([x1, y1, x2, y2])
                    print(f"ROI {len(self.rois)} added: ({x1}, {y1}) to ({x2}, {y2})")
            
            self.current_roi = None
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to delete the nearest ROI
            if self.rois:
                # Find the ROI closest to the click
                min_dist = float('inf')
                delete_idx = -1
                for idx, roi in enumerate(self.rois):
                    x1, y1, x2, y2 = roi
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        delete_idx = idx
                
                if delete_idx >= 0:
                    deleted_roi = self.rois.pop(delete_idx)
                    print(f"Deleted ROI {delete_idx + 1}: {deleted_roi}")
    
    def draw_rois(self):
        """Draw all ROIs on the display frame"""
        self.display_frame = self.frame.copy()
        
        # Draw existing ROIs
        for idx, roi in enumerate(self.rois):
            x1, y1, x2, y2 = roi
            color = self.colors[idx % len(self.colors)]
            
            # Draw rectangle
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ROI number
            label = f"ROI {idx + 1}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(self.display_frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), color, -1)
            cv2.putText(self.display_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw current ROI being drawn
        if self.current_roi is not None:
            x1, y1, x2, y2 = self.current_roi
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
        
        if self.mode == 'signal':
            instructions = [
                "Draw ONE box around the traffic light only",
                "New drag replaces the signal box",
                "L: Load saved  |  S: Save  |  Q: Quit",
            ]
        else:
            instructions = [
                "Left Click & Drag: Draw ROI",
                "Right Click: Delete nearest ROI",
                "D: Delete last ROI",
                "C: Clear all ROIs",
                "L: Load saved ROIs",
                "S: Save and continue",
                "Q: Quit without saving",
            ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(self.display_frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
    
    def _read_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading config: {e}")
        return {}

    def load_rois(self):
        """Load ROIs from config file (violation_rois or legacy rois; signal_roi for signal mode)."""
        data = self._read_config()
        if not data and not os.path.exists(self.config_file):
            print(f"No config file found at {self.config_file}")
            return False
        try:
            if self.mode == 'signal':
                sr = data.get('signal_roi')
                self.rois = [sr] if sr and len(sr) == 4 else []
                print(f"Loaded signal ROI from {self.config_file}" if self.rois else "No saved signal ROI")
            else:
                self.rois = list(data.get('violation_rois') or data.get('rois') or [])
                print(f"Loaded {len(self.rois)} violation zone(s) from {self.config_file}")
            return True
        except Exception as e:
            print(f"Error loading ROIs: {e}")
            return False

    def save_rois(self):
        """Merge-save: violation mode writes violation_rois; signal mode writes signal_roi."""
        try:
            data = self._read_config()
            data['video_path'] = self.video_path
            data['frame_size'] = [self.frame.shape[1], self.frame.shape[0]]
            if self.mode == 'signal':
                if not self.rois:
                    print("No signal ROI to save.")
                    return False
                data['signal_roi'] = self.rois[0]
                if 'violation_rois' not in data and data.get('rois'):
                    data['violation_rois'] = data['rois']
            else:
                data['violation_rois'] = self.rois
                if 'rois' in data:
                    data['rois'] = self.rois
                else:
                    data['rois'] = self.rois
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved to {self.config_file} ({self.mode})")
            return True
        except Exception as e:
            print(f"Error saving ROIs: {e}")
            return False
    
    def run(self):
        """Run the interactive ROI selector"""
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return None
        
        # Read first frame
        ret, self.frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            cap.release()
            return None
        
        # Resize frame to standard size
        self.frame = cv2.resize(self.frame, (1020, 600))
        cap.release()
        
        # Try to load existing ROIs
        self.load_rois()
        
        window_name = (
            "ROI: traffic SIGNAL (one box)"
            if self.mode == 'signal'
            else "ROI: VIOLATION zones (vehicles)"
        )
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n=== ROI Selector Started ===")
        if self.mode == 'signal':
            print("Draw ONE rectangle tightly around the traffic light (color detection uses only this region).")
        else:
            print("Draw rectangle(s) where crossing vehicles count as violations.")
        print("Press 'S' to save and continue, 'Q' to quit\n")
        
        while True:
            self.draw_rois()
            cv2.imshow(window_name, self.display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                if len(self.rois) == 0:
                    print("Warning: Draw at least one ROI first.")
                    continue
                if self.mode == 'signal' and len(self.rois) != 1:
                    print("Signal mode expects exactly one box; redraw if needed.")
                    continue
                if self.save_rois():
                    print(f"\nROI selection complete. {len(self.rois)} zones defined.")
                    break
                
            elif key == ord('q') or key == ord('Q'):
                # Quit without saving
                print("\nQuitting without saving...")
                self.rois = None
                break
                
            elif key == ord('d') or key == ord('D'):
                # Delete last ROI
                if self.rois:
                    deleted = self.rois.pop()
                    print(f"Deleted last ROI: {deleted}")
                    
            elif key == ord('c') or key == ord('C'):
                # Clear all ROIs
                if self.rois:
                    self.rois = []
                    print("Cleared all ROIs")
                    
            elif key == ord('l') or key == ord('L'):
                # Load ROIs
                self.load_rois()
        
        cv2.destroyAllWindows()
        return self.rois


def main():
    """Main function to run ROI selector standalone"""
    import sys
    
    video_path = 'tr.mp4'
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    selector = ROISelector(video_path)
    rois = selector.run()
    
    if rois:
        print(f"\nFinal ROIs: {rois}")
    else:
        print("\nNo ROIs selected.")


if __name__ == "__main__":
    main()

