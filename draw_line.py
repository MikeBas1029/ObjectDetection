import os


# Compute the average center of detected bright regions
    if bright_regions:
        avg_x = int(np.mean([pt[0] for pt in bright_regions]))
        avg_y = int(np.mean([pt[1] for pt in bright_regions]))
    else:
        print(f"No bright regions detected for {image_path}. Skipping line drawing.")
        continue

    # Load the false origin coordinates from its corresponding txt file
    false_origin_file = os.path.join(false_origin_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    print(f"Looking for file: " + false_origin_file)

    if not os.path.exists(false_origin_file):
        #print(f"Warning: No false origin file found for {false_origin_file}. Skipping.")
        continue

    with open(false_origin_file, "r") as f:
        try:
            content = f.readline().strip()
            print(f"Reading from {false_origin_file}: {content}")  # Debugging output
            false_origin_x, false_origin_y = map(int, content.split(','))  # Fix: Split by comma
        except ValueError:
            print(f"Error: Could not parse '{content}' in {false_origin_file}. Expected two integers separated by a comma.")
            continue

    # Draw a line from the false origin to the computed average center
    cv2.line(image, (false_origin_x, false_origin_y), (avg_x, avg_y), (0, 255, 0), 2)