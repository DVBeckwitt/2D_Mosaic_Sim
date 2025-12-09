import os
import platform
from pathlib import Path
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import PillowWriter  # Use PillowWriter for debugging

# ------------------ Downloads path ------------------
def get_downloads_dir() -> Path:
    sysname = platform.system()
    base = Path(os.environ.get("USERPROFILE", Path.home())) if sysname == "Windows" else Path.home()
    p = base / "Downloads"
    p.mkdir(parents=True, exist_ok=True)
    return p

DOWNLOADS = get_downloads_dir()
GIF_PATH = DOWNLOADS / "ewald_sphere_overlap_optimized_gpu.gif"

# ------------------ Config ------------------
z_min, z_max = 0.0, 5.0
r = 1.0
dot_L_vals = np.array([0, 1, 2], dtype=float)  # Reduce the number of dots for testing
num_line_pts = 30                     # Reduce the resolution of the rods
num_frames = 60                       # Decrease the number of frames for debugging
gif_fps = 30                          # frames per second for the GIF
hold_seconds = 5.0                    # hold final frame for 5 seconds
rod_color = "black"                   # all rods are the same color
dot_color = "red"                     # the six highlighted dots
rod_lw = 0.8                          # thickness of the rods
ewald_radius = 2                      # radius of the Ewald sphere
theta_i = np.radians(10)              # initial theta of Ewald sphere
# --------------------------------------------

# Ewald sphere configuration
ewald_center = cp.array([0, 0, 0])  # center at the origin
theta_vals = np.linspace(0.0, 2*np.pi, num_frames, endpoint=True)
hold_frames_gif = int(round(hold_seconds * gif_fps))

# Cylinder (reciprocal space rod) and dots (same as in the previous animation)
z_line = np.linspace(z_min, z_max, num_line_pts)
base_line = np.column_stack([np.full_like(z_line, r), np.zeros_like(z_line), z_line])
base_dots = np.column_stack([np.full_like(dot_L_vals, r), np.zeros_like(dot_L_vals), dot_L_vals])

# Convert to CuPy arrays
base_line = cp.array(base_line)  # Convert to CuPy array
base_dots = cp.array(base_dots)  # Convert to CuPy array

# Reciprocal space rotation around Gz (using CuPy for GPU acceleration)
def Rz(x, y, th):
    c, s = cp.cos(th), cp.sin(th)
    return x*c - y*s, x*s + y*c

# Ewald sphere (parametric form of a sphere in reciprocal space, GPU-accelerated)
def ewald_sphere(radius, theta, phi):
    x = radius * cp.sin(theta) * cp.cos(phi)
    y = radius * cp.sin(theta) * cp.sin(phi)
    z = radius * cp.cos(theta)
    return x, y, z

# Figure setup
fig = plt.figure(figsize=(4, 4))  # Smaller figure size for faster rendering
ax = fig.add_subplot(111, projection="3d")

# Set the axes limits and labels
def set_axes():
    ax.set_xlim([-1.2, 1.2])   # Zoomed-in view around r=1
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([z_min, z_max])
    ax.set_box_aspect((1, 1, 3.8))  # Scaling to make z-axis more prominent
    ax.set_xlabel("Gx")
    ax.set_ylabel("Gy")
    ax.set_zlabel("Gz")
    ax.set_title("Ewald sphere overlapping with reciprocal space cylinder")
    # Draw the XYZ axes in gray for context
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], linewidth=1, color="gray")
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], linewidth=1, color="gray")
    ax.plot([0, 0], [0, 0], [z_min, z_max], linewidth=1, color="gray")

# Initialize the figure for each frame
def init():
    ax.cla()
    set_axes()
    return []

# Draw the cylinder (reciprocal space rods) and dots
def draw_copy(theta):
    # Draw thin rod (reciprocal space rod)
    x_line, y_line = Rz(base_line[:, 0], base_line[:, 1], theta)
    ax.plot(x_line.get(), y_line.get(), base_line[:, 2].get(), linewidth=rod_lw, color=rod_color)  # Convert to numpy with `.get()`
    
    # Draw dots at (1, 0, L)
    x_d, y_d = Rz(base_dots[:, 0], base_dots[:, 1], theta)
    ax.scatter(x_d.get(), y_d.get(), base_dots[:, 2].get(), s=16, color=dot_color)  # Convert to numpy with `.get()`

# Draw the Ewald sphere, and leave a trace where it overlaps with the cylinder
def draw_ewald_sphere(frame):
    # Gradually reduce the sphere's opacity by decreasing its size
    radius = ewald_radius * (1 - frame / num_frames)  # Shrinks over time
    theta = theta_i  # Fixed theta_i for initial orientation
    phi_vals = cp.linspace(0, 2*np.pi, 50)  # Reduced the number of points for speed

    # Create a trace of the Ewald sphere's overlap with the cylinder
    for phi in phi_vals:
        x, y, z = ewald_sphere(radius, theta, phi)
        ax.scatter(x.get(), y.get(), z.get(), color="blue", s=5)  # Trace of the Ewald sphere

# Function to draw all copies of the rods and the Ewald sphere overlap
def draw_all_copies(frame):
    for th in theta_vals:
        draw_copy(th)
    draw_ewald_sphere(frame)

total_frames_gif = num_frames + hold_frames_gif

# Animation function to generate frames
def animate(i):
    ax.cla()
    set_axes()
    if i < num_frames:
        for th in theta_vals[: i + 1]:
            draw_copy(th)
        draw_ewald_sphere(i)
    else:
        draw_all_copies(i)  # Hold final frame with trace of Ewald sphere
    return []

# Animation with PillowWriter for faster GIF creation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=total_frames_gif, interval=1000/gif_fps, blit=True)

# Save the GIF using PillowWriter (for debugging purposes)
anim.save(str(GIF_PATH), writer=PillowWriter(fps=gif_fps))
print(f"Saved GIF to: {GIF_PATH}")
