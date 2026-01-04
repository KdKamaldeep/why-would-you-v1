#!/usr/bin/env python3
"""
Face Alignment and Preprocessing for LatentSync

Detects faces in video frames, smooths bounding boxes over time,
crops and resizes to fixed resolution for LatentSync processing.
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import json

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("⚠️ MediaPipe not available, falling back to OpenCV face detection")

try:
    from insightface import app as insightface_app
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.debug("InsightFace not available, using MediaPipe/OpenCV")


class FaceAligner:
    """
    Face detection, alignment, and preprocessing for LatentSync.
    """
    
    def __init__(
        self,
        detection_method: str = "mediapipe",  # "mediapipe", "insightface", "opencv"
        crop_size: int = 512,
        smoothing_alpha: float = 0.7,  # EMA smoothing factor (0-1, higher = less smoothing)
        min_face_size: int = 64,
        min_confidence: float = 0.5,
        expand_ratio: float = 1.3,  # Expand bbox to include chin and mouth
    ):
        """
        Initialize face aligner.
        
        Args:
            detection_method: "mediapipe", "insightface", or "opencv"
            crop_size: Output crop size (default: 512 for LatentSync)
            smoothing_alpha: EMA smoothing factor (0-1)
            min_face_size: Minimum face size in pixels
            min_confidence: Minimum detection confidence
            expand_ratio: Ratio to expand bbox (1.3 = 30% larger)
        """
        self.detection_method = detection_method
        self.crop_size = crop_size
        self.smoothing_alpha = smoothing_alpha
        self.min_face_size = min_face_size
        self.min_confidence = min_confidence
        self.expand_ratio = expand_ratio
        
        # Initialize detection models
        self.face_detector = None
        self.mp_face_detection = None
        self.mp_drawing = None
        self.opencv_cascade = None
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize face detection model based on method."""
        if self.detection_method == "mediapipe" and MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_drawing = mp.solutions.drawing_utils
                self.face_detector = self.mp_face_detection.FaceDetection(
                    model_selection=1,  # 0 = short range, 1 = full range
                    min_detection_confidence=self.min_confidence
                )
                logger.info("✅ Initialized MediaPipe face detection")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize MediaPipe: {e}, falling back to OpenCV")
                self.detection_method = "opencv"
        
        if self.detection_method == "insightface" and INSIGHTFACE_AVAILABLE:
            try:
                # InsightFace requires model path - would need to be configured
                logger.warning("⚠️ InsightFace requires model setup, falling back to MediaPipe/OpenCV")
                self.detection_method = "mediapipe" if MEDIAPIPE_AVAILABLE else "opencv"
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize InsightFace: {e}")
                self.detection_method = "mediapipe" if MEDIAPIPE_AVAILABLE else "opencv"
        
        if self.detection_method == "opencv" or not self.face_detector:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.opencv_cascade = cv2.CascadeClassifier(cascade_path)
                if self.opencv_cascade.empty():
                    raise FileNotFoundError("OpenCV cascade file not found")
                logger.info("✅ Initialized OpenCV face detection")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenCV face detection: {e}")
                raise
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Detect face in a frame.
        
        Returns:
            (x, y, width, height, confidence) or None if not found
        """
        if self.detection_method == "mediapipe" and self.face_detector:
            return self._detect_mediapipe(frame)
        else:
            return self._detect_opencv(frame)
    
    def _detect_mediapipe(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """Detect face using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]  # Use first (largest) face
            bbox = detection.location_data.relative_bounding_box
            
            h, w = frame.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            confidence = detection.score[0]
            
            # Validate size
            if width >= self.min_face_size and height >= self.min_face_size:
                return (x, y, width, height, confidence)
        
        return None
    
    def _detect_opencv(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """Detect face using OpenCV Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.opencv_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            # Use largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            # OpenCV doesn't provide confidence, use 1.0
            return (x, y, w, h, 1.0)
        
        return None
    
    def smooth_bbox(
        self,
        current_bbox: Optional[Tuple[int, int, int, int, float]],
        previous_bbox: Optional[Tuple[int, int, int, int, float]]
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Apply EMA smoothing to bounding box.
        
        Args:
            current_bbox: Current frame bbox (x, y, w, h, confidence)
            previous_bbox: Previous frame bbox (x, y, w, h, confidence)
        
        Returns:
            Smoothed bbox or None
        """
        if current_bbox is None:
            return previous_bbox  # Use previous if current not detected
        
        if previous_bbox is None:
            return current_bbox  # Use current if no previous
        
        x1, y1, w1, h1, conf1 = current_bbox
        x2, y2, w2, h2, conf2 = previous_bbox
        
        # Smooth center and size
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        
        cx_smooth = int(self.smoothing_alpha * cx1 + (1 - self.smoothing_alpha) * cx2)
        cy_smooth = int(self.smoothing_alpha * cy1 + (1 - self.smoothing_alpha) * cy2)
        w_smooth = int(self.smoothing_alpha * w1 + (1 - self.smoothing_alpha) * w2)
        h_smooth = int(self.smoothing_alpha * h1 + (1 - self.smoothing_alpha) * h2)
        
        # Reconstruct bbox from smoothed center and size
        x_smooth = cx_smooth - w_smooth // 2
        y_smooth = cy_smooth - h_smooth // 2
        
        # Use higher confidence
        conf_smooth = max(conf1, conf2)
        
        return (x_smooth, y_smooth, w_smooth, h_smooth, conf_smooth)
    
    def expand_bbox(
        self,
        bbox: Tuple[int, int, int, int, float],
        frame_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int, float]:
        """
        Expand bbox to include chin and mouth area.
        
        Args:
            bbox: (x, y, w, h, confidence)
            frame_shape: (height, width)
        
        Returns:
            Expanded bbox
        """
        x, y, w, h, conf = bbox
        h_frame, w_frame = frame_shape
        
        # Expand by ratio
        new_w = int(w * self.expand_ratio)
        new_h = int(h * self.expand_ratio)
        
        # Center the expansion
        cx, cy = x + w // 2, y + h // 2
        new_x = max(0, cx - new_w // 2)
        new_y = max(0, cy - new_h // 2)
        
        # Adjust if out of bounds
        if new_x + new_w > w_frame:
            new_x = w_frame - new_w
        if new_y + new_h > h_frame:
            new_y = h_frame - new_h
        
        # Ensure minimum size
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, w_frame - new_x)
        new_h = min(new_h, h_frame - new_y)
        
        return (new_x, new_y, new_w, new_h, conf)
    
    def crop_and_resize(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int, float]
    ) -> np.ndarray:
        """
        Crop face region and resize to fixed size.
        
        Args:
            frame: Input frame
            bbox: (x, y, w, h, confidence)
        
        Returns:
            Cropped and resized frame (crop_size x crop_size)
        """
        x, y, w, h, _ = bbox
        h_frame, w_frame = frame.shape[:2]
        
        # Clamp to frame bounds
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        # Crop
        crop = frame[y:y+h, x:x+w]
        
        # Resize to fixed size
        resized = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def process_video(
        self,
        video_path: str,
        output_path: str,
        save_debug: bool = False,
        debug_frames: List[int] = None
    ) -> Dict[str, Any]:
        """
        Process video: detect faces, smooth, crop, resize.
        
        Args:
            video_path: Input video path
            output_path: Output preprocessed video path
            save_debug: Save debug frames with bbox visualization
            debug_frames: List of frame indices to save debug frames for
        
        Returns:
            Dictionary with:
            - success: bool
            - bbox_track: List of bboxes per frame
            - avg_confidence: float
            - frames_processed: int
            - frames_with_face: int
            - error: Optional[str]
        """
        result = {
            "success": False,
            "bbox_track": [],
            "avg_confidence": 0.0,
            "frames_processed": 0,
            "frames_with_face": 0,
            "error": None
        }
        
        if debug_frames is None:
            debug_frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"📹 Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (self.crop_size, self.crop_size))
            
            if not out.isOpened():
                raise ValueError(f"Could not create output video: {output_path}")
            
            # Track bboxes
            bbox_track = []
            previous_bbox = None
            confidences = []
            frames_with_face = 0
            frame_idx = 0
            
            # Create debug output directory if needed
            debug_dir = None
            if save_debug:
                debug_dir = Path(output_path).parent / f"{Path(output_path).stem}_debug"
                debug_dir.mkdir(exist_ok=True)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect face
                detected_bbox = self.detect_face(frame)
                
                # Smooth with previous
                smoothed_bbox = self.smooth_bbox(detected_bbox, previous_bbox)
                
                if smoothed_bbox:
                    # Expand bbox
                    expanded_bbox = self.expand_bbox(smoothed_bbox, frame.shape[:2])
                    
                    # Crop and resize
                    cropped = self.crop_and_resize(frame, expanded_bbox)
                    out.write(cropped)
                    
                    # Track
                    bbox_track.append(expanded_bbox)
                    confidences.append(expanded_bbox[4])
                    frames_with_face += 1
                    previous_bbox = expanded_bbox
                    
                    # Save debug frame if requested
                    if save_debug and frame_idx in debug_frames:
                        debug_frame = frame.copy()
                        x, y, w, h, conf = expanded_bbox
                        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(debug_frame, f"Conf: {conf:.2f}", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        debug_path = debug_dir / f"frame_{frame_idx:05d}.jpg"
                        cv2.imwrite(str(debug_path), debug_frame)
                else:
                    # No face detected - use previous bbox or black frame
                    if previous_bbox:
                        cropped = self.crop_and_resize(frame, previous_bbox)
                        out.write(cropped)
                        bbox_track.append(previous_bbox)
                    else:
                        # No previous bbox - write black frame
                        black_frame = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
                        out.write(black_frame)
                        bbox_track.append(None)
                
                frame_idx += 1
                
                if frame_idx % 30 == 0:
                    logger.debug(f"   Processed {frame_idx}/{total_frames} frames")
            
            cap.release()
            out.release()
            
            # Calculate statistics
            result["success"] = True
            result["bbox_track"] = bbox_track
            result["frames_processed"] = frame_idx
            result["frames_with_face"] = frames_with_face
            result["avg_confidence"] = np.mean(confidences) if confidences else 0.0
            
            # Save bbox track to JSON
            track_file = Path(output_path).parent / f"{Path(output_path).stem}_bbox_track.json"
            with open(track_file, 'w') as f:
                json.dump({
                    "bbox_track": bbox_track,
                    "avg_confidence": result["avg_confidence"],
                    "frames_with_face": frames_with_face,
                    "total_frames": frame_idx
                }, f, indent=2)
            
            logger.info(f"✅ Preprocessed {frame_idx} frames: {frames_with_face} with faces (avg conf: {result['avg_confidence']:.2f})")
            logger.info(f"📁 Saved bbox track to: {track_file}")
            
            if save_debug and debug_dir:
                logger.info(f"📁 Saved debug frames to: {debug_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error processing video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result["error"] = str(e)
        
        return result


def get_face_aligner(
    detection_method: Optional[str] = None,
    crop_size: int = 512,
    smoothing_alpha: float = 0.7,
    min_face_size: int = 64,
    min_confidence: float = 0.5
) -> FaceAligner:
    """
    Factory function to create FaceAligner instance.
    
    Args:
        detection_method: "mediapipe", "insightface", or "opencv" (auto-detect if None)
        crop_size: Output crop size
        smoothing_alpha: EMA smoothing factor
        min_face_size: Minimum face size
        min_confidence: Minimum detection confidence
    
    Returns:
        FaceAligner instance
    """
    if detection_method is None:
        # Auto-detect best available
        if MEDIAPIPE_AVAILABLE:
            detection_method = "mediapipe"
        else:
            detection_method = "opencv"
    
    return FaceAligner(
        detection_method=detection_method,
        crop_size=crop_size,
        smoothing_alpha=smoothing_alpha,
        min_face_size=min_face_size,
        min_confidence=min_confidence
    )

