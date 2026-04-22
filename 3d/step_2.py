import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree

# --- Configuration ---
NUM_CLASSES = 6
CLASS_NAMES = ["Ground", "Tree", "Car", "LightPole", "Fence", "Building"]
CLASS_WEIGHTS = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 1.0], dtype=np.float32)
STPLS3D_COLORS = {
    0: [153, 153, 153], 1: [0, 255, 0], 2: [255, 0, 0],
    3: [0, 0, 255], 4: [255, 0, 255], 5: [255, 165, 0], -1: [51, 51, 51]
}

class CCCamera:
    def __init__(self, node, photogroup_params):
        self.image_path = node.find('ImagePath').text
        self.raw_name = os.path.basename(self.image_path)
        self.mask_name = os.path.splitext(self.raw_name)[0] + ".png" # JPG -> PNG conversion
        
        self.width = photogroup_params['width']
        self.height = photogroup_params['height']
        self.f_pixel = photogroup_params['f_pixel']
        self.ppx = photogroup_params['ppx']
        self.ppy = photogroup_params['ppy']
        self.disto = photogroup_params['disto']

        pose = node.find('Pose')
        rot = pose.find('Rotation')
        self.R = np.array([
            [float(rot.find('M_00').text), float(rot.find('M_01').text), float(rot.find('M_02').text)],
            [float(rot.find('M_10').text), float(rot.find('M_11').text), float(rot.find('M_12').text)],
            [float(rot.find('M_20').text), float(rot.find('M_21').text), float(rot.find('M_22').text)]
        ])
        center = pose.find('Center')
        self.C = np.array([float(center.find('x').text), float(center.find('y').text), float(center.find('z').text)])

    def project(self, points_3d):
        p_rel = points_3d - self.C
        p_cam = np.dot(p_rel, self.R.T)
        mask_z = p_cam[:, 2] > 0.1 
        z = p_cam[:, 2]
        x, y = p_cam[:, 0] / z, p_cam[:, 1] / z
        r2 = x*x + y*y
        k1, k2, k3, p1, p2 = self.disto
        radial = (1 + k1*r2 + k2*r2**2 + k3*r2**3)
        x_distorted = x * radial + (2*p1*x*y + p2*(r2 + 2*x*x))
        y_distorted = y * radial + (p1*(r2 + 2*y*y) + 2*p2*x*y)
        u = x_distorted * self.f_pixel + self.ppx
        v = y_distorted * self.f_pixel + self.ppy
        return u, v, mask_z

# --- Evaluation Functions ---
def get_gt_labels(plydata):
    """Read and map directly from the 'class' field of the PLY data"""
    if 'class' not in plydata['vertex']._property_lookup:
        print("❌ Error: 'class' field not found in point cloud!")
        return None
    
    raw_gt = plydata['vertex']['class'].astype(int)
    unique_vals = np.unique(raw_gt)
    
    # Mapping logic
    if np.max(unique_vals) > 6:
        print("    -> Fine-grained labels detected, performing STPLS3D mapping...")
        lut = np.full(256, -1, dtype=int)
        lut[[15, 17, 18, 19]] = 0        # Ground
        lut[[2, 3, 4]] = 1               # Tree
        lut[[5, 6, 8]] = 2               # Car
        lut[[11, 12]] = 3                # Light pole
        lut[[14]] = 4                    # Fence
        lut[[1, 7, 9, 10, 13, 16]] = 5   # Building
        return lut[raw_gt]
    else:
        # If already 0-5 or 1-6
        if np.min(unique_vals) >= 1:
            return raw_gt - 1
        return raw_gt

def evaluate_iou(pred, gt):
    if gt is None: return
    print("\n" + "="*40 + "\n🎯 Performance Evaluation Report\n" + "="*40)
    ious = []
    for c in range(NUM_CLASSES):
        itnt = np.logical_and(np.logical_and(pred == c, gt == c), gt != -1).sum()
        union = np.logical_and(np.logical_or(pred == c, gt == c), gt != -1).sum()
        iou = itnt / union if union > 0 else np.nan
        ious.append(iou)
        print(f"[{CLASS_NAMES[c]:<10}] IoU: {iou*100:.2f}%" if not np.isnan(iou) else f"[{CLASS_NAMES[c]:<10}] IoU: N/A")
    print("-" * 40)
    print(f"🏆 mIoU: {np.nanmean(ious)*100:.2f}%\n" + "="*40 + "\n")

# --- Multi-processing Worker ---
global_pts, global_m_dir = None, None
def init_worker(pts, m_dir):
    global global_pts, global_m_dir
    global_pts, global_m_dir = pts, m_dir

def worker_task(cam):
    path = os.path.join(global_m_dir, cam.mask_name)
    if not os.path.exists(path): return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    u, v, m_z = cam.project(global_pts)
    u_i, v_i = np.round(u).astype(np.int32), np.round(v).astype(np.int32)
    in_img = (m_z & (u_i >= 0) & (u_i < cam.width) & (v_i >= 0) & (v_i < cam.height))
    return np.where(in_img)[0], img[v_i[in_img], u_i[in_img]]

# --- Main Program ---
def main(config):
    print(f"Reading point cloud: {config['ply_in']}")
    plydata = PlyData.read(config['ply_in'])
    pts_3d = np.stack([plydata['vertex'][c] for c in 'xyz'], axis=1).astype(np.float32)
    num_pts = len(pts_3d)

    cams = [c for c in parse_cc_xml(config['xml_in'])[::config['sample_rate']]]
    if config['max_images']: cams = cams[:config['max_images']]

    v_buf = np.zeros((num_pts, NUM_CLASSES), dtype=np.uint16)
    with Pool(min(cpu_count(), 16), init_worker, (pts_3d, config['mask_dir'])) as p:
        results = list(tqdm(p.imap(worker_task, cams), total=len(cams), desc="Projecting"))

    for res in results:
        if res is None: continue
        idx, labels = res
        labels = np.clip(labels, 0, NUM_CLASSES - 1)
        for i in range(NUM_CLASSES):
            v_buf[idx[labels == i], i] += 1
    
    weighted_votes = v_buf * CLASS_WEIGHTS
    pred_labels = np.argmax(weighted_votes, axis=1)
    pred_labels[np.sum(v_buf, axis=1) == 0] = -1

    # Evaluation
    gt_labels = get_gt_labels(plydata)
    evaluate_iou(pred_labels, gt_labels)

    # Prepare output data
    print("Merging attributes and saving...")
    rgb = np.zeros((num_pts, 3), dtype=np.uint8)
    for l, c in STPLS3D_COLORS.items(): rgb[pred_labels == l] = c

    # Get all original properties except color and existing pred_label
    v_el = plydata['vertex']
    prop_names = [p.name for p in v_el.properties if p.name not in ['red', 'green', 'blue', 'pred_label']]
    
    dt = [(n, v_el[n].dtype) for n in prop_names]
    dt += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('pred_label', 'i2')]
    
    new_v = np.empty(num_pts, dtype=dt)
    for n in prop_names: new_v[n] = v_el[n]
    new_v['red'], new_v['green'], new_v['blue'] = rgb.T
    new_v['pred_label'] = pred_labels

    PlyData([PlyElement.describe(new_v, 'vertex')], text=False).write(config['ply_out'])
    print(f"🎉 Finished! Saved to: {config['ply_out']}")

def parse_cc_xml(path):
    root = ET.parse(path).getroot()
    cams = []
    for g in root.find('Block/Photogroups').findall('Photogroup'):
        p = {
            'width': int(g.find('ImageDimensions/Width').text),
            'height': int(g.find('ImageDimensions/Height').text),
            'f_pixel': float(g.find('FocalLengthPixels').text),
            'ppx': float(g.find('PrincipalPoint/x').text),
            'ppy': float(g.find('PrincipalPoint/y').text),
            'disto': [float(g.find(f'Distortion/{k}').text) for k in ['K1', 'K2', 'K3', 'P1', 'P2']]
        }
        for ph in g.findall('Photo'):
            if ph.find('Pose') is not None: cams.append(CCCamera(ph, p))
    return cams

if __name__ == "__main__":
    config = {
        "ply_in": "dataset/3D/WMSC/WMSC_points.ply",
        "xml_in": "dataset/3D/WMSC/WMSC_CamInfoCC.xml",
        "mask_dir": "./work_dirs/WMSC_2d_masks",
        "ply_out": "./work_dirs/WMSC_result/WMSC_result.ply",
        "sample_rate": 1,       # Sample every N images (set to 1 to use all)
        "max_images": None,     # Set to None if no limit
        "filter_list": None,    # Set to None if not using a text list
    }
    main(config)