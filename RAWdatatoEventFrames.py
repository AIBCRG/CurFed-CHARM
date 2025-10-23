import numpy as np
from metavision_core.event_io import EventsIterator
import cv2
import os

# Parameters
input_folder = "path to input folder containing RAW files"  # Folder containing the .raw file
output_folder = "path to save processed frames"  # Folder to save processed frames
frame_width = 1280
frame_height = 720
accumulation_time_ms = 40  # Accumulation time in milliseconds

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process .raw files in the input directory
for raw_file in os.listdir(input_folder):
    if raw_file.endswith(".raw"):
        input_raw_file = os.path.join(input_folder, raw_file)

        # Create a sub-folder for each raw file to save its frames
        raw_output_folder = os.path.join(output_folder, os.path.splitext(raw_file)[0])
        os.makedirs(raw_output_folder, exist_ok=True)

        # Initialize event reader
        events_iterator = EventsIterator(input_raw_file, mode="delta_t", delta_t=accumulation_time_ms * 1000)

        frame_count = 0
        for events in events_iterator:
            # Create an empty frame with three channels (RGB)
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            # Accumulate events in the frame
            for e in events:
                x, y, polarity = e["x"], e["y"], e["p"]
                if polarity:  # ON events in green
                    frame[y, x] = [0,255,0]
                else:  # OFF events in red
                    frame[y, x] = [0, 0, 255]

            # Save the frame as an image
            frame_filename = os.path.join(raw_output_folder, f"frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)

            # Save the events of this frame as .npy file
            events_filename = os.path.join(raw_output_folder, f"events_{frame_count:05d}.npy")
            np.save(events_filename, events)

            print(f"Saved frame {frame_count} from file '{raw_file}'.")
            frame_count += 1

        print(f"Total frames saved for '{raw_file}': {frame_count}")

print("Processing complete.")