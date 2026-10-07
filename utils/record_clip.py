#------------------------------------------------------------
# RECORD A TEST CLIP FROM THE FRIDGE OR DOOR CAMERA ON THE PI
#------------------------------------------------------------
# The orchestrator will own capture later. This is for trying the models on the
# real fridge before the camera adapters exist.
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def record_command(settings, seconds, output):
    """Build the capture command for a Pi camera module or a USB camera."""
    width, height, fps = int(settings["width"]), int(settings["height"]), int(settings["fps"])
    if not 1 <= seconds <= 120:
        raise ValueError("Record between 1 and 120 seconds.")
    if settings["kind"] == "rpicam":
        executable = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
        if not executable:
            raise RuntimeError("rpicam-vid is missing. Install rpicam-apps on the Pi.")
        return [executable, "--camera", str(int(settings["camera"])), "-t", str(int(seconds * 1000)),
                "--width", str(width), "--height", str(height), "--framerate", str(fps),
                "--codec", "libav", "--libav-format", "mp4", "--nopreview", "-o", str(output)]
    if settings["kind"] == "v4l2":
        executable = shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError("FFmpeg is missing.")
        return [executable, "-nostdin", "-y", "-f", "v4l2", "-framerate", str(fps),
                "-video_size", f"{width}x{height}", "-i", str(settings["device"]),
                "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", str(output)]
    raise ValueError("capture kind must be rpicam or v4l2.")


def record(settings, seconds, directory, camera):
    """Record one clip to a new file and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{camera}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
    process = subprocess.run(record_command(settings, seconds, output), capture_output=True,
                             text=True, timeout=seconds + 30)
    if process.returncode or not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Recording failed: {(process.stderr or process.stdout)[-800:]}")
    return output


if __name__ == "__main__":
    from production.video import load_config
    parser = argparse.ArgumentParser(description="Record a clip from a configured camera.")
    parser.add_argument("camera", help="fridge or door")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/video.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.camera not in config["capture"]:
        parser.error(f"camera must be one of {sorted(config['capture'])}")
    print(record(config["capture"][args.camera], args.seconds,
                 Path(config["acquisition_directory"]) / "recordings", args.camera))
