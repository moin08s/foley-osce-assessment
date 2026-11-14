"""Task 3: Investigate Step 4 detector firing at 00:00 in session 016.

Requirements:
- Show which pixels/regions triggered it.
- Save frames:
  1. Raw video frame at 00:00
  2. Binary mask of triggering pixels (R > 0.65, G > 0.65, B > 0.65)
  3. Overlay highlighting triggering pixels
- Report exact numerical properties: spatial coordinates, pixel counts, percentages.
- Do NOT guess the cause.
"""

import os
import cv2
import numpy as np

VIDEO_PATH = r"C:\New folder (2)\103NZ_30\016.MP4"
OUT_DIR = r"C:\Medical\reports"
os.makedirs(OUT_DIR, exist_ok=True)

def investigate():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return

    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        print("Error: Could not read frame at 00:00")
        return

    h, w, c = frame_bgr.shape
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Threshold used by Step 4 glove detector:
    # (r > 0.65) & (g > 0.65) & (b > 0.65) on normalized RGB [0, 1]
    # In 8-bit [0, 255], 0.65 * 255 = 165.75, so pixel values >= 166 in R, G, and B.
    norm = frame_rgb.astype(np.float32) / 255.0
    r = norm[..., 0]
    g = norm[..., 1]
    b = norm[..., 2]

    trigger_mask = (r > 0.65) & (g > 0.65) & (b > 0.65)
    total_pixels = h * w
    trigger_pixel_count = int(np.sum(trigger_mask))
    trigger_percentage = (trigger_pixel_count / total_pixels) * 100.0

    # Save raw frame
    raw_path = os.path.join(OUT_DIR, "session_016_frame_00s_raw.png")
    cv2.imwrite(raw_path, frame_bgr)

    # Save binary mask (255 where triggered, 0 elsewhere)
    mask_uint8 = (trigger_mask.astype(np.uint8)) * 255
    mask_path = os.path.join(OUT_DIR, "session_016_frame_00s_mask.png")
    cv2.imwrite(mask_path, mask_uint8)

    # Create overlay: raw frame with trigger pixels highlighted in red
    overlay_bgr = frame_bgr.copy()
    overlay_bgr[trigger_mask] = [0, 0, 255]  # Bright red in BGR
    # Blend with original for visibility
    blended = cv2.addWeighted(frame_bgr, 0.4, overlay_bgr, 0.6, 0)
    overlay_path = os.path.join(OUT_DIR, "session_016_frame_00s_overlay.png")
    cv2.imwrite(overlay_path, blended)

    # Find connected components / spatial bounding boxes of triggered regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
    
    # Sort components by pixel area (excluding background at index 0)
    component_indices = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)

    print("=" * 90)
    print("TASK 3: INVESTIGATION OF STEP 4 DETECTOR FIRING AT 00:00 (SESSION 016)")
    print("=" * 90)
    print(f"Video Source:             {VIDEO_PATH}")
    print(f"Frame Timestamp:          00:00 (Frame 0)")
    print(f"Frame Dimensions:         {w} x {h} ({total_pixels:,} total pixels)")
    print(f"Detector Threshold:       R > 165.75 AND G > 165.75 AND B > 165.75 (normalized > 0.65)")
    print(f"Triggering Pixels Count:  {trigger_pixel_count:,} pixels")
    print(f"Triggering Frame Area:    {trigger_percentage:.2f}% of total frame")
    print("\nSaved Diagnostic Images:")
    print(f"1. Raw Frame:             {raw_path}")
    print(f"2. Trigger Mask:          {mask_path}")
    print(f"3. Red Highlight Overlay: {overlay_path}")

    print("\nSpatial Location of Largest Triggering Regions (Bounding Boxes):")
    print(f"{'REGION #':<10} | {'AREA (pixels)':<14} | {'% FRAME':<10} | {'[X, Y, W, H]':<22} | {'CENTROID (X, Y)'}")
    print("-" * 80)
    for rank, idx in enumerate(component_indices[:8], 1):
        area = stats[idx, cv2.CC_STAT_AREA]
        pct = (area / total_pixels) * 100.0
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        width = stats[idx, cv2.CC_STAT_WIDTH]
        height = stats[idx, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[idx]
        print(f"Region {rank:<3d} | {area:<14,d} | {pct:<9.2f}% | [{x}, {y}, {width}, {height}] | ({cx:.1f}, {cy:.1f})")

    # Downsampled 128x128 feature check
    cap_resized = cv2.VideoCapture(VIDEO_PATH)
    ret_r, frame_raw = cap_resized.read()
    cap_resized.release()
    frame_128 = cv2.resize(frame_raw, (128, 128))
    norm_128 = cv2.cvtColor(frame_128, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r128 = norm_128[..., 0]
    g128 = norm_128[..., 1]
    b128 = norm_128[..., 2]
    val_128 = float(((r128 > 0.65) & (g128 > 0.65) & (b128 > 0.65)).mean())

    print("\nFeature Vector Value in Downsampled 128x128 Model Input:")
    print(f"glove_white fraction at t=00:00: {val_128:.4f} ({val_128*100:.2f}% of 128x128 pixels)")

    # Check detector condition on session 016
    data = np.load(r"C:\Medical\dataset\features_128px\016_feat.npz")
    gw_series = data["X"][:, 116]
    T = len(gw_series)
    el = max(3, int(T * 0.45))
    gw_early = gw_series[:el]
    p65 = np.percentile(gw_early, 65)
    print(f"\nStep 4 Detector Thresholds for Session 016:")
    print(f"T={T} windows, Early window limit={el} windows ({el*2}s)")
    print(f"Window 0 (00:00) glove_white value: {gw_series[0]:.4f}")
    print(f"Early window 65th percentile:      {p65:.4f}")
    print(f"Condition: Window 0 ({gw_series[0]:.4f}) >= p65 ({p65:.4f}) -> {gw_series[0] >= p65}")
    print("=" * 90)

if __name__ == "__main__":
    investigate()
