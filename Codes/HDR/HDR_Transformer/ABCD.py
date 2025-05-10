import cv2
import numpy as np
from skimage.exposure import match_histograms

# Step 1: Preprocess input
def preprocess_image(img):
    return img.astype(np.float32) / 255.0

# Step 2: ECC alignment to reference
def ecc_align(frames, ref_idx=0):
    aligned = []
    ref_gray = cv2.cvtColor((frames[ref_idx] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    for i, frame in enumerate(frames):
        if i == ref_idx:
            aligned.append(frame)
            continue
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        warp = np.eye(3, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(ref_gray, gray, warp, cv2.MOTION_HOMOGRAPHY)
            aligned_frame = cv2.warpPerspective(frame, warp, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
            aligned.append(aligned_frame)
        except:
            aligned.append(frame)
    return aligned

# Step 3: Optical flow refinement
def refine_with_flow(frames, ref_idx=0):
    refined = []
    ref = frames[ref_idx]
    ref_gray = cv2.cvtColor((ref * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    for i, frame in enumerate(frames):
        if i == ref_idx:
            refined.append(ref)
            continue
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray, ref_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
        warped = cv2.remap(frame, (flow_map[..., 0] + flow[..., 0]), (flow_map[..., 1] + flow[..., 1]), interpolation=cv2.INTER_LINEAR)
        refined.append(warped)
    return refined

# Step 4: Histogram match to style frame
def apply_histogram_matching(frames, style_idx=1):
    return [match_histograms(f, frames[style_idx], channel_axis=-1) for f in frames]

# Step 5: Weight maps
def compute_weights(frames, alpha=2.0, style_idx=1, sigma=0.2):
    weights = []
    for idx, img in enumerate(frames):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        saturation = np.std(img, axis=2)
        well_exposedness = np.exp(-0.5 * ((img - 0.5) ** 2) / (sigma ** 2)).prod(axis=2)
        weight = contrast * saturation * well_exposedness
        if idx == style_idx:
            weight *= alpha
        weights.append(weight + 1e-12)
    weights = np.array(weights)
    weights /= np.sum(weights, axis=0, keepdims=True)
    return weights

# Step 6: Laplacian pyramid fusion
def gaussian_pyramid(img, levels):
    gp = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        gp.append(img)
    return gp

def laplacian_pyramid(img, levels):
    gp = gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        size = (gp[i].shape[1], gp[i].shape[0])
        GE = cv2.pyrUp(gp[i+1], dstsize=size)
        lp.append(gp[i] - GE)
    lp.append(gp[-1])
    return lp

def fuse_pyramid(frames, weights, levels=5):
    W_pyr = [gaussian_pyramid(w[..., None], levels) for w in weights]
    L_pyr = [laplacian_pyramid(f, levels) for f in frames]
    fused_pyr = []
    for l in range(levels + 1):
        fused = sum(W_pyr[i][l] * L_pyr[i][l] for i in range(len(frames)))
        fused_pyr.append(fused)
    return collapse_pyramid(fused_pyr)

def collapse_pyramid(pyr):
    img = pyr[-1]
    for l in reversed(pyr[:-1]):
        img = cv2.pyrUp(img, dstsize=(l.shape[1], l.shape[0])) + l
    return np.clip(img, 0, 1)

# Step 7: Tone mapping
def tone_map(img):
    return img / (1 + img)

# Main Pipeline
def hdr_fusion_pipeline(frames_rgb, ref_idx=0, style_idx=1, alpha=2.0, sigma=0.2, levels=5):
    frames = [preprocess_image(f) for f in frames_rgb]
    aligned = ecc_align(frames, ref_idx=ref_idx)
    refined = refine_with_flow(aligned, ref_idx=ref_idx)
    styled = apply_histogram_matching(refined, style_idx=style_idx)
    weights = compute_weights(styled, alpha=alpha, style_idx=style_idx, sigma=sigma)
    hdr = fuse_pyramid(styled, weights, levels)
    tone_mapped = tone_map(hdr)
    return (hdr * 255).astype(np.uint8), (tone_mapped * 255).astype(np.uint8)
