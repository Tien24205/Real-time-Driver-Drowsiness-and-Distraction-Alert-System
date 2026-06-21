import matplotlib.pyplot as plt
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Parameters
total_frames = 500
calibration_frames = 100
baseline_ear = 0.32
alpha = 0.78
dynamic_threshold = baseline_ear * alpha

# Simulate EAR data
frames = np.arange(total_frames)
ear_values = np.random.normal(baseline_ear, 0.015, total_frames)

# Introduce normal blinks (quick drops)
blinks = [150, 220, 380, 450]
for b in blinks:
    ear_values[b:b+4] = np.random.normal(0.12, 0.02, 4) # Very quick drop
    # smoothing transition
    ear_values[b-2:b] = np.linspace(baseline_ear, 0.15, 2)
    ear_values[b+4:b+7] = np.linspace(0.12, baseline_ear, 3)

# Introduce a genuine drowsiness event (prolonged drop)
drowsy_start = 280
drowsy_end = 330
ear_values[drowsy_start:drowsy_end] = np.random.normal(0.18, 0.015, drowsy_end - drowsy_start)
ear_values[drowsy_start-5:drowsy_start] = np.linspace(baseline_ear, 0.18, 5)
ear_values[drowsy_end:drowsy_end+10] = np.linspace(0.18, baseline_ear, 10)

# Apply a simple moving average (similar to main_app.py window=5)
window_size = 5
ear_smoothed = np.convolve(ear_values, np.ones(window_size)/window_size, mode='same')
# Fix the edges
ear_smoothed[:2] = ear_values[:2]
ear_smoothed[-2:] = ear_values[-2:]

# --- Plotting ---
plt.figure(figsize=(12, 6))

# Plot the raw and smoothed EAR
plt.plot(frames, ear_smoothed, label="Smoothed EAR (Moving Avg = 5)", color='#1f77b4', linewidth=2)
plt.plot(frames, ear_values, alpha=0.3, label="Raw EAR", color='#a0c4ff')

# Draw Baseline (from calibration)
plt.axhline(y=baseline_ear, xmin=0.2, xmax=1.0, color='green', linestyle='--', linewidth=2, label=f"Calibrated Baseline EAR ({baseline_ear:.2f})")

# Draw Dynamic Threshold
plt.axhline(y=dynamic_threshold, xmin=0.2, xmax=1.0, color='orange', linestyle='-.', linewidth=2, label=f"Dynamic Threshold ({dynamic_threshold:.3f})")

# Highlight Calibration Phase
plt.axvspan(0, calibration_frames, color='gray', alpha=0.2, label='Calibration Phase (100 frames)')
plt.axvline(x=calibration_frames, color='black', linestyle=':', linewidth=2)

# Highlight Normal Blinks (Debounce prevents alarm)
plt.annotate('Normal Blink\n(Duration < 15f)\nNo Alarm', xy=(150, 0.12), xytext=(120, 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=9)

# Highlight Drowsiness Alarm Trigger
alarm_trigger_frame = drowsy_start + 15  # Debounce limit = 15
plt.axvspan(alarm_trigger_frame, drowsy_end, color='red', alpha=0.3, label='ALARM: Drowsiness Detected')
plt.annotate('Alarm Triggered\n(Duration $\geq$ 15f)', xy=(alarm_trigger_frame, 0.18), xytext=(alarm_trigger_frame - 40, 0.08),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6), fontsize=10, color='darkred', weight='bold')

# Formatting
plt.title("Dynamic Calibration and Debounce Filtering of Eye Aspect Ratio (EAR)", fontsize=14, pad=15)
plt.xlabel("Frame Index", fontsize=12)
plt.ylabel("Eye Aspect Ratio (EAR)", fontsize=12)
plt.ylim(0.0, 0.45)
plt.xlim(0, total_frames)
plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Save the figure
out_path = r'E:\Project2026\Real-time Driver Drowsiness and Distraction Alert System\ear_dynamic_threshold.png'
plt.savefig(out_path, dpi=300)
print(f"Chart saved successfully to: {out_path}")
