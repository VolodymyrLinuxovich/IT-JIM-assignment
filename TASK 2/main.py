import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the global map image.
def load_map(mp):
    mimg = cv2.imread(mp)
    if mimg is None:
        raise FileNotFoundError(f"Failed to load global map: {mp}")
    return mimg

# Load all cropped frames from the specified folder.
def load_frms(fld):
    frms = []
    # Sort files by name (assumes correct ordering).
    f_files = sorted(os.listdir(fld))
    for f in f_files:
        fpath = os.path.join(fld, f)
        frm = cv2.imread(fpath)
        if frm is not None:
            frms.append(frm)
        else:
            print(f"Failed to load frame: {fpath}")
    print(f"Loaded {len(frms)} frames from {fld}")
    return frms

# Compute homography between a source image and the target map using ORB detector.
def find_homog(src, tgt, det):
    kp_src, dsrc = det.detectAndCompute(src, None)
    kp_tgt, dtgt = det.detectAndCompute(tgt, None)
    
    if dsrc is None or dtgt is None:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(dsrc, dtgt)
    matches = sorted(matches, key=lambda x: x.distance)

    # Select best 50 matches or all if fewer.
    good = matches[:50] if len(matches) >= 50 else matches

    if len(good) >= 4:  # At least 4 matches needed.
        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        tgt_pts = np.float32([kp_tgt[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC, 5.0)
        return H, mask
    else:
        return None, None

# Calculate the center point of the frame transformed into the global map coordinate system.
def get_center(H, frm_shp):
    h, w = frm_shp[:2]
    ctr = np.array([[[w / 2, h / 2]]], dtype=np.float32)
    ctr_tr = cv2.perspectiveTransform(ctr, H)
    return ctr_tr[0][0]  # (x, y)

# Smooth the trajectory using a moving average.
def smooth_traj(pts, win=5):
    smt = []
    n = len(pts)
    for i in range(n):
        xs = [pts[j][0] for j in range(max(0, i-win), min(n, i+win+1))]
        ys = [pts[j][1] for j in range(max(0, i-win), min(n, i+win+1))]
        smt.append((np.mean(xs), np.mean(ys)))
    return smt

# Visualize the trajectory on the global map with both raw and smoothed points.
# Additionally, save the visualization as an image file.
def vis_traj(mimg, pts, spts, save_path='drone_trajectory.png'):
    mimg_rgb = cv2.cvtColor(mimg, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(mimg_rgb)

    xs = [p[0] for p in spts]
    ys = [p[1] for p in spts]
    plt.plot(xs, ys, marker='o', color='red', linewidth=2, markersize=4, label='Trajectory')
    plt.scatter([xs[0]], [ys[0]], color='green', s=100, label='Start')
    plt.scatter([xs[-1]], [ys[-1]], color='blue', s=100, label='Finish')

    for idx, (x, y) in enumerate(spts):
        plt.text(x, y, str(idx+1), color="yellow", fontsize=8)

    plt.legend()
    plt.title('Drone Trajectory Reconstruction')
    # Save the figure before displaying it.
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Visualization saved as: {save_path}")
    plt.show()

# Create a video with an animation of the trajectory.
def create_vid(mimg, spts, out_path='drone_route.avi', fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    h, w = mimg.shape[:2]
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    temp = mimg.copy()

    for i in range(len(spts)):
        frm_img = temp.copy()
        for j in range(1, i+1):
            pt1 = (int(spts[j-1][0]), int(spts[j-1][1]))
            pt2 = (int(spts[j][0]), int(spts[j][1]))
            cv2.line(frm_img, pt1, pt2, (0, 0, 255), 2)
            cv2.circle(frm_img, pt2, 4, (0, 255, 0), -1)
        if i == 0:
            cv2.circle(frm_img, (int(spts[0][0]), int(spts[0][1])), 8, (0, 255, 0), -1)
        if i == len(spts)-1:
            cv2.circle(frm_img, (int(spts[-1][0]), int(spts[-1][1])), 8, (255, 0, 0), -1)
        out.write(frm_img)

    out.release()
    print(f"Video saved: {out_path}")

# Global execution block (no main() function is used for data input)
map_path = 'data/global_map.png'  # Path to the global map image.
fld = 'data/crops/'               # Path to the folder containing cropped frames.

gmap = load_map(map_path)         # Load the global map.
frms = load_frms(fld)             # Load cropped frames.

orb = cv2.ORB_create(nfeatures=1000)  # Initialize ORB detector with increased features.
traj_pts = []

# For each frame, compute homography and determine the transformed center.
for idx, frm in enumerate(frms):
    H, mask = find_homog(frm, gmap, orb)
    if H is not None:
        ctr = get_center(H, frm.shape)
        traj_pts.append(ctr)
        print(f"Frame {idx+1}: Homography found, center = {ctr}")
    else:
        print(f"Frame {idx+1}: Could not compute stable homography.")

if not traj_pts:
    print("No trajectory points computed, exiting.")
    exit(1)

s_traj = smooth_traj(traj_pts, win=3)  # Smooth trajectory with a window size of 3.

# Visualize the trajectory on the global map and save the image.
vis_traj(gmap, traj_pts, s_traj, save_path='drone_trajectory.png')

# Create a video animation of the trajectory.
create_vid(gmap, s_traj, out_path='drone_route.avi', fps=10)
