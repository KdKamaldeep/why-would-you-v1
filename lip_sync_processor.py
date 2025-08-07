#!/usr/bin/env python3
"""
Lip-Sync Processor Module - Handles lip-sync using Wav2Lip
"""

import os
import logging
import subprocess
import torch
from typing import List

logger = logging.getLogger(__name__)

class LipSyncProcessor:
    """Handles lip-sync using Wav2Lip."""
    
    def __init__(self, wav2lip_path: str = "Wav2Lip"):
        self.wav2lip_path = wav2lip_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def process_lip_sync(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Apply lip-sync to video using Wav2Lip."""
        try:
            logger.info(f"Applying lip-sync to: {video_path}")
            
            # Change to Wav2Lip directory
            original_dir = os.getcwd()
            os.chdir(self.wav2lip_path)
            
            # Run Wav2Lip inference
            cmd = [
                'python', 'inference.py',
                '--checkpoint_path', 'checkpoints/wav2lip.pth',
                '--face', video_path,
                '--audio', audio_path,
                '--outfile', output_path,
                '--pads', '0', '20', '0', '0'  # Adjust padding as needed
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Return to original directory
            os.chdir(original_dir)
            
            logger.info(f"Applied lip-sync: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in lip-sync: {e}")
            # Fallback to simple audio overlay
            return self._simple_audio_overlay(video_path, audio_path, output_path)
    
    def process_multiple_videos(self, video_paths: List[str], audio_path: str, output_dir: str) -> List[str]:
        """Apply lip-sync to multiple videos."""
        lip_sync_paths = []
        for i, video_path in enumerate(video_paths):
            output_path = f"{output_dir}/scene_{i+1}_lipsync.mp4"
            lip_sync_path = self.process_lip_sync(video_path, audio_path, output_path)
            lip_sync_paths.append(lip_sync_path)
        return lip_sync_paths
    
    def _simple_audio_overlay(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Simple audio overlay as fallback."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Applied simple audio overlay: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in simple audio overlay: {e}")
            return video_path

