
"""
Video extraction utility for HOI-M3/NeuralDome dataset processing.

This module provides functionality to extract frames from videos and convert them
to image sequences, specifically designed for the HOI-M3/NeuralDome dataset processing pipeline.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from utils.path_utils import check_input_path, prepare_output_path

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoInfoReader:
    """
    A class to extract video information using ffprobe.
    
    This class provides a convenient interface to get video metadata
    similar to ffmpeg-python library.
    """
    
    def __init__(self, input_path: str) -> None:
        """
        Initialize the video info reader.
        
        Args:
            input_path: Path to the video file
            
        Raises:
            FileNotFoundError: If the input path doesn't exist
            RuntimeError: If no video stream is found in the file
        """
        self.input_path = input_path
        self._validate_input()
        self.video_stream = self._extract_video_info()
    
    def _validate_input(self) -> None:
        """Validate the input video file."""
        check_input_path(
            self.input_path,
            allowed_suffix=['.mp4', '.gif', '.png', '.jpg', '.jpeg'],
            tag='input file',
            path_type='file'
        )
    
    def _extract_video_info(self) -> Dict:
        """
        Extract video information using ffprobe.
        
        Returns:
            Dictionary containing video stream information
            
        Raises:
            RuntimeError: If ffprobe fails or no video stream is found
        """
        cmd = [
            'ffprobe', '-show_format', '-show_streams', '-of', 'json',
            self.input_path
        ]
        
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = process.communicate()
            
            if process.returncode != 0:
                error_msg = err.decode('utf-8') if err else "Unknown ffprobe error"
                raise RuntimeError(f"ffprobe failed: {error_msg}")
            
            probe = json.loads(out.decode('utf-8'))
            
            # Find video stream
            video_stream = next(
                (stream for stream in probe['streams'] 
                 if stream['codec_type'] == 'video'), 
                None
            )
            
            if video_stream is None:
                raise RuntimeError('No video stream found in the file')
                
            return video_stream
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}")
        except Exception as e:
            raise RuntimeError(f"Error extracting video info: {e}")
    
    def __getitem__(self, key: str) -> Union[str, int, float]:
        """
        Get video stream property by key.
        
        Args:
            key: Property name to retrieve
            
        Returns:
            The value of the requested property
            
        Raises:
            KeyError: If the key is not found in video stream
        """
        if key not in self.video_stream:
            available_keys = list(self.video_stream.keys())
            raise KeyError(f"Key '{key}' not found. Available keys: {available_keys}")
        
        return self.video_stream[key]
    
    def get_frame_count(self) -> int:
        """Get the total number of frames in the video."""
        return int(self['nb_frames'])
    
    def get_duration(self) -> float:
        """Get the duration of the video in seconds."""
        return float(self['duration'])
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get the resolution of the video as (width, height)."""
        return (int(self['width']), int(self['height']))



def video_to_images(
    input_path: str,
    output_folder: str,
    resolution: Optional[Tuple[int, int]] = None,
    img_format: str = '%06d.png',
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    quality: int = 2,
    threads: int = 64,
    verbose: bool = True
) -> None:
    """
    Convert a video to a folder of images using ffmpeg.
    
    Args:
        input_path: Path to the input video file
        output_folder: Directory to save the extracted images
        resolution: Optional (width, height) for resizing output images
        img_format: Format string for output filenames (e.g., '%06d.png')
        start_frame: Starting frame index (inclusive, 0-based)
        end_frame: Ending frame index (exclusive). If None, extract to end
        quality: JPEG quality (2-31, where 2 is highest quality)
        threads: Number of threads to use for processing
        verbose: Whether to print the ffmpeg command
        
    Raises:
        FileNotFoundError: If input path doesn't exist
        RuntimeError: If ffmpeg command fails
    """
    # Validate inputs
    check_input_path(
        input_path,
        allowed_suffix=['.mp4'],
        tag='input video',
        path_type='file'
    )
    prepare_output_path(
        output_folder,
        allowed_suffix=[],
        tag='output image folder',
        path_type='dir',
        overwrite=True
    )
    
    # Get video information
    try:
        video_info = VideoInfoReader(input_path)
        total_frames = video_info.get_frame_count()
        
        # Normalize frame indices
        start_frame = max(0, min(start_frame, total_frames - 1))
        if end_frame is None:
            end_frame = total_frames
        else:
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            
        logger.info(f"Extracting frames {start_frame} to {end_frame-1} from {total_frames} total frames")
        
    except Exception as e:
        logger.error(f"Failed to get video information: {e}")
        raise
    
    # Build ffmpeg command
    command = _build_ffmpeg_command(
        input_path, output_folder, img_format, start_frame, end_frame,
        resolution, quality, threads
    )
    
    # Execute command
    if verbose:
        logger.info(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Successfully extracted images to {output_folder}")
        
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg failed with return code {e.returncode}"
        if e.stderr:
            error_msg += f": {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    except FileNotFoundError:
        error_msg = "ffmpeg not found. Please ensure ffmpeg is installed and in PATH"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def _build_ffmpeg_command(
    input_path: str,
    output_folder: str,
    img_format: str,
    start_frame: int,
    end_frame: int,
    resolution: Optional[Tuple[int, int]],
    quality: int,
    threads: int
) -> List[str]:
    """
    Build the ffmpeg command for video extraction.
    
    Args:
        input_path: Path to input video
        output_folder: Output directory
        img_format: Image format string
        start_frame: Starting frame
        end_frame: Ending frame
        resolution: Optional (width, height) for resizing
        quality: JPEG quality setting
        threads: Number of threads
        
    Returns:
        List of command arguments
    """
    command = [
        'ffmpeg',
        '-i', input_path,
        '-qscale:v', str(quality),
        '-filter_complex',
        f'[0]trim=start_frame={start_frame}:end_frame={end_frame}[v0]',
        '-map', '[v0]',
        '-f', 'image2',
        '-v', 'error',
        '-start_number', '0',
        '-threads', str(threads)
    ]
    
    # Add resolution if specified
    if resolution:
        width, height = resolution
        command.insert(3, '-s')
        command.insert(4, f'{width}x{height}')
    
    # Add output path
    command.append(f'{output_folder}/{img_format}')
    
    return command




def process_hoim3_sequence(
    sequence_name: str,
    base_video_path: str,
    base_output_path: str,
    num_cameras: int = 42,
    img_format: str = '%06d.jpg',
    quality: int = 2
) -> None:
    """
    Process a HOI-M3 sequence by extracting frames from all camera videos.
    
    Args:
        sequence_name: Name of the sequence to process
        base_video_path: Base path containing video files
        base_output_path: Base path for output images
        num_cameras: Number of camera views to process
        img_format: Format for output image filenames
        quality: JPEG quality setting
    """
    logger.info(f"Processing sequence: {sequence_name}")
    
    for camera_id in range(num_cameras):
        input_path = os.path.join(
            base_video_path, sequence_name, 'videos', f'{camera_id}.mp4'
        )
        output_folder = os.path.join(
            base_output_path, sequence_name, str(camera_id)
        )
        
        try:
            logger.info(f"Processing camera {camera_id}/{num_cameras-1}")
            video_to_images(
                input_path=input_path,
                output_folder=output_folder,
                img_format=img_format,
                quality=quality,
                verbose=True
            )
            
        except FileNotFoundError:
            logger.warning(f"Video file not found: {input_path}")
            continue
        except Exception as e:
            logger.error(f"Failed to process camera {camera_id}: {e}")
            continue
    
    logger.info(f"Completed processing sequence: {sequence_name}")


def main():
    """Main function to process HOI-M3 video sequences."""
    # Configuration
    SEQUENCE_NAME = 'bedroom_data05'
    BASE_VIDEO_PATH = './videos'
    BASE_OUTPUT_PATH = './images'
    NUM_CAMERAS = 42
    IMG_FORMAT = '%06d.jpg'
    QUALITY = 2
    
    try:
        process_hoim3_sequence(
            sequence_name=SEQUENCE_NAME,
            base_video_path=BASE_VIDEO_PATH,
            base_output_path=BASE_OUTPUT_PATH,
            num_cameras=NUM_CAMERAS,
            img_format=IMG_FORMAT,
            quality=QUALITY
        )
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == '__main__':
    main()



