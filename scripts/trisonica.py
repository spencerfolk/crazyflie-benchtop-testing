import serial
import math
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

TOKEN_RE = re.compile(r'([A-Z]{1,2})\s*([+-]?\d+(?:\.\d+)?)')

def parse_trisonica_line(line):
    pairs = TOKEN_RE.findall(line)
    if not pairs:
        return None
    out = {}
    for k, v in pairs:
        try:
            out[k] = float(v)
        except ValueError:
            pass
    if all(axis in out for axis in ("U", "V", "W")):
        out["S_from_UVW"] = math.sqrt(out["U"]**2 + out["V"]**2 + out["W"]**2)
    return out

def live_plot(port="COM7", baudrate=115200, history=500):
    ser = serial.Serial(port, baudrate, timeout=1)
    print(f"Connected to {port}")

    # History buffers
    t = deque(maxlen=history)
    u_hist, v_hist, w_hist, s_hist = (deque(maxlen=history) for _ in range(4))

    # Layout: 2 columns
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(4, 2, width_ratios=[2, 1], hspace=0.4)

    # Left column: 4 stacked subplots
    ax_u = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[2, 0])
    ax_s = fig.add_subplot(gs[3, 0])
    axs = [ax_u, ax_v, ax_w, ax_s]

    # Right column: polar subplot
    ax_polar = fig.add_subplot(gs[:, 1], polar=True)

    # Initialize lines
    line_u, = ax_u.plot([], [], "r-", linewidth=3)
    line_v, = ax_v.plot([], [], "g-", linewidth=3)
    line_w, = ax_w.plot([], [], "b-", linewidth=3)
    line_s, = ax_s.plot([], [], "k-", linewidth=3)

    # Initialize text objects in top-left corner
    txt_u = ax_u.text(0.02, 0.9, "", transform=ax_u.transAxes, color="r", fontsize=10, va='top')
    txt_v = ax_v.text(0.02, 0.9, "", transform=ax_v.transAxes, color="g", fontsize=10, va='top')
    txt_w = ax_w.text(0.02, 0.9, "", transform=ax_w.transAxes, color="b", fontsize=10, va='top')
    txt_s = ax_s.text(0.02, 0.9, "", transform=ax_s.transAxes, color="k", fontsize=10, va='top')
    txts = [txt_u, txt_v, txt_w, txt_s]

    # Setup axes
    for ax, lbl in zip(axs, ["U component", "V component", "W component", "Wind Magnitude"]):
        ax.set_xlim(0, history)
        ax.set_ylim(-10, 10)
        ax.set_title(lbl)

    ax_polar.set_theta_zero_location("E")
    ax_polar.set_rlim(0, 10)

    arrow = ax_polar.arrow(0, 0, 0, 0, width=0.02)  # initial arrow

    i = 0
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        data = parse_trisonica_line(line)
        if not data:
            continue

        # Update histories
        u_hist.append(data.get("U", np.nan))
        v_hist.append(data.get("V", np.nan))
        w_hist.append(data.get("W", np.nan))
        s_hist.append(data.get("S_from_UVW", np.nan))
        t.append(i)
        i += 1

        # Update line data
        line_u.set_data(range(len(u_hist)), u_hist)
        line_v.set_data(range(len(v_hist)), v_hist)
        line_w.set_data(range(len(w_hist)), w_hist)
        line_s.set_data(range(len(s_hist)), s_hist)

        # Update text annotations with latest value
        latest_values = [u_hist[-1], v_hist[-1], w_hist[-1], s_hist[-1]]
        for txt, val in zip(txts, latest_values):
            txt.set_text(f"{val:.2f}")

        # Update polar arrow
        if "U" in data and "V" in data:
            u, v = data["U"], data["V"]
            mag = math.sqrt(u**2 + v**2)
            direction = math.atan2(v, u)
            arrow.remove()
            arrow = ax_polar.arrow(
                direction, 0, 0, mag,
                width=0.02, length_includes_head=True, color="C3"
            )
            ax_polar.set_rlim(0, max(5, mag + 1))

        # Rescale y dynamically
        for ax, hist in zip(axs, [u_hist, v_hist, w_hist, s_hist]):
            if hist:
                ax.set_ylim(min(hist)-1, max(hist)+1)

        plt.pause(0.05)  # ~20 FPS max

if __name__ == "__main__":
    try:
        live_plot()
    except KeyboardInterrupt:
        print("Exiting...")
