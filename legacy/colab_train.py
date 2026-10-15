"""
MNemo Colab Training — FIXED VERSION
=====================================
Copy each CELL block into a SEPARATE Colab cell.
Runtime → Change runtime type → T4 GPU → Save

This version fixes the dataset download bug from before.
"""


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: INSTALL (copy this into first Colab cell and run)          ║
# ╚═══════════════════════════════════════════════════════════════════════╝
#
# !pip install -q torch torchvision ultralytics mediapipe opencv-python-headless Pillow matplotlib
# !pip install -q git+https://github.com/openai/CLIP.git
# !wget -q -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
# print("✅ All packages installed")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: DOWNLOAD DATASET (copy this into second Colab cell)        ║
# ╚═══════════════════════════════════════════════════════════════════════╝
#
# import os
# os.makedirs("data/epic_kitchens/videos", exist_ok=True)
#
# # Step A: Download annotations (instant — tiny CSV files from GitHub)
# !wget -q -O data/epic_kitchens/EPIC_100_train.csv "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_train.csv"
# !wget -q -O data/epic_kitchens/EPIC_100_noun_classes.csv "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_noun_classes.csv"
# !wget -q -O data/epic_kitchens/EPIC_100_verb_classes.csv "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_verb_classes.csv"
# print("✅ Annotations downloaded")
#
# # Step B: Download videos using the OFFICIAL script (not a pip package!)
# !git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts.git
# !cd epic-kitchens-download-scripts && python epic_downloader.py \
#     --output-path ../data/epic_kitchens/videos \
#     --participants P01 \
#     --extension MP4 \
#     --videos
#
# # Check what we got
# import glob
# vids = glob.glob("data/epic_kitchens/videos/**/*.MP4", recursive=True) + \
#        glob.glob("data/epic_kitchens/videos/**/*.mp4", recursive=True)
# print(f"✅ Downloaded {len(vids)} video files")
# if vids:
#     print(f"   First: {vids[0]}")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: MOUNT GOOGLE DRIVE (to save results)                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝
#
# from google.colab import drive
# drive.mount('/content/drive')


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: THE COMPLETE MODEL + TRAINING (paste and run)              ║
# ╚═══════════════════════════════════════════════════════════════════════╝
# Everything below is one giant cell. Just paste it all.

import os, csv, math, time, glob, json
import cv2, clip, torch, numpy as np
import torch.nn as nn
import mediapipe as mp_lib
from PIL import Image
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

print("="*60)
print("  MNemo — Multi-Modal Egocentric Action Recognizer")
print("="*60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Paths ──────────────────────────────────────────────────────
CSV_PATH    = "data/epic_kitchens/EPIC_100_train.csv"
VIDEO_DIR   = "data/epic_kitchens/videos"
FEAT_PATH   = "data/extracted_features.pt"
CKPT_DIR    = "checkpoints"
RESULTS_DIR = "evaluation_results"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Find videos (handle nested directories) ───────────────────
ALL_VIDEOS = {}
for pattern in ["**/*.MP4", "**/*.mp4"]:
    for p in glob.glob(os.path.join(VIDEO_DIR, pattern), recursive=True):
        vid_id = os.path.splitext(os.path.basename(p))[0]
        ALL_VIDEOS[vid_id] = p
print(f"Found {len(ALL_VIDEOS)} video files")

if not ALL_VIDEOS:
    print("\n⚠️  NO VIDEOS FOUND!")
    print("The download might have failed. Try these fixes:")
    print("1. Re-run Cell 2")
    print("2. Or manually download P01 videos from:")
    print("   https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqyp0t8m")
    print("   Upload them to data/epic_kitchens/videos/ in Colab")
    raise SystemExit("Cannot continue without videos")


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class ModalEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(drop))
    def forward(self, x): return self.net(x)

class CrossModalFusion(nn.Module):
    def __init__(self, d, heads=4, layers=2, drop=0.1):
        super().__init__()
        L = nn.TransformerEncoderLayer(d, heads, d*2, drop, "gelu", True, True)
        self.tf = nn.TransformerEncoder(L, layers)
        self.mod_emb = nn.Embedding(4, d)
    def forward(self, x):
        B = x.size(0)
        ids = torch.arange(4, device=x.device).unsqueeze(0).expand(B, -1)
        return self.tf(x + self.mod_emb(ids)).mean(1)

class TemporalTransformer(nn.Module):
    def __init__(self, d, heads=4, layers=2, maxlen=16, drop=0.1):
        super().__init__()
        L = nn.TransformerEncoderLayer(d, heads, d*2, drop, "gelu", True, True)
        self.tf = nn.TransformerEncoder(L, layers)
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return self.tf(x + self.pe[:x.size(1)].unsqueeze(0)).mean(1)

class EgoModel(nn.Module):
    def __init__(self, nv=20, nn_=30, d=128, T=8, drop=0.1):
        super().__init__()
        self.d = d
        self.enc_c = ModalEncoder(512, d, drop)
        self.enc_o = ModalEncoder(80, d, drop)
        self.enc_h = ModalEncoder(42, d, drop)
        self.enc_f = ModalEncoder(8, d, drop)
        self.cross = CrossModalFusion(d, 4, 2, drop)
        self.temp  = TemporalTransformer(d, 4, 2, T, drop)
        self.head_v = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,d), nn.GELU(), nn.Dropout(drop), nn.Linear(d,nv))
        self.head_n = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,d), nn.GELU(), nn.Dropout(drop), nn.Linear(d,nn_))

    def forward(self, cf, of, hf, ff):
        B, T, _ = cf.shape
        c,o,h,f_ = self.enc_c(cf), self.enc_o(of), self.enc_h(hf), self.enc_f(ff)
        tok = torch.stack([c,o,h,f_], dim=2).view(B*T, 4, self.d)
        fused = self.cross(tok).view(B, T, self.d)
        emb = self.temp(fused)
        return self.head_v(emb), self.head_n(emb)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

_m = EgoModel()
print(f"Model parameters: {_m.count_params():,}")
del _m


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTORS
# ═══════════════════════════════════════════════════════════════

class Extractor:
    def __init__(self):
        print("\nLoading feature extraction models...")
        self.clip_model, self.clip_pp = clip.load("ViT-B/32", device=DEVICE)
        print("  ✅ CLIP")
        self.yolo = YOLO("yolov8n.pt")
        print("  ✅ YOLOv8n")

        task_file = "hand_landmarker.task"
        if not os.path.exists(task_file):
            os.system(f"wget -q -O {task_file} https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_file),
            num_hands=2, min_hand_detection_confidence=0.4,
            running_mode=mp_vision.RunningMode.IMAGE)
        self.hands = mp_vision.HandLandmarker.create_from_options(opts)
        print("  ✅ MediaPipe Hands")
        print("All models loaded!\n")

    def clip_feat(self, frame):
        pil = Image.fromarray(frame[:,:,::-1])
        img = self.clip_pp(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = self.clip_model.encode_image(img)
            f /= f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()[0]

    def obj_feat(self, frame):
        v = np.zeros(80, dtype=np.float32)
        for r in self.yolo(frame, verbose=False, conf=0.3):
            for b in r.boxes:
                ci = int(b.cls.item())
                if 0 <= ci < 80: v[ci] = max(v[ci], float(b.conf.item()))
        return v

    def hand_feat(self, frame):
        v = np.zeros(42, dtype=np.float32)
        rgb = frame[:,:,::-1].copy()
        try:
            res = self.hands.detect(mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb))
        except: return v
        if not res.hand_landmarks: return v
        lm = res.hand_landmarks[0]
        xs = [l.x for l in lm]; ys = [l.y for l in lm]
        v[0],v[1],v[2],v[3] = min(xs),min(ys),max(xs),max(ys)
        v[4],v[5] = lm[0].x, lm[0].y
        v[6],v[7] = lm[8].x, lm[8].y
        v[8],v[9] = (v[0]+v[2])/2, (v[1]+v[3])/2
        v[10] = 1.0
        v[11] = 1.0 if res.handedness[0][0].category_name=="Right" else 0.0
        bw,bh = max(xs)-min(xs), max(ys)-min(ys)
        v[24] = bw*bh; v[25] = bw/max(bh,1e-6)
        v[26] = np.sqrt((lm[8].x-lm[0].x)**2 + (lm[8].y-lm[0].y)**2)
        return v

    def flow_feat(self, prev_g, curr_g):
        v = np.zeros(8, dtype=np.float32)
        if prev_g is None: v[7]=1.0; return v
        fl = cv2.calcOpticalFlowFarneback(prev_g, curr_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        u,vv = fl[...,0], fl[...,1]
        mag = np.sqrt(u**2+vv**2); mask = mag > 0.75
        if np.any(mask):
            mx,my = float(np.median(u[mask])), float(np.median(vv[mask]))
            mm = float(np.sqrt(mx**2+my**2))
        else: mx,my,mm = 0,0,0
        v[0],v[1],v[2] = mx,my,mm
        if mm < 1.5: v[7]=1.0
        elif abs(my)>abs(mx): v[5 if my<0 else 6]=1.0
        else: v[3 if mx>0 else 4]=1.0
        return v


# ═══════════════════════════════════════════════════════════════
# DATA LOADING + FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def parse_ts(s):
    p = s.split(":"); return int(p[0])*3600 + int(p[1])*60 + float(p[2])

def load_data():
    """Load annotations, filter, extract features."""
    if os.path.exists(FEAT_PATH):
        print(f"Loading cached features from {FEAT_PATH}")
        return torch.load(FEAT_PATH, weights_only=False)

    # Load annotations
    segs = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            vid = r["video_id"]
            if vid not in ALL_VIDEOS: continue
            segs.append({
                "video_id": vid, "path": ALL_VIDEOS[vid],
                "start": parse_ts(r["start_timestamp"]),
                "stop": parse_ts(r["stop_timestamp"]),
                "verb": r["verb"], "vc": int(r["verb_class"]),
                "noun": r["noun"], "nc": int(r["noun_class"]),
            })
    print(f"Matched {len(segs)} segments to available videos")

    if len(segs) == 0:
        raise RuntimeError("No segments matched! Check video file names vs annotation video_id column.")

    # Filter to top classes
    NV, NN = 20, 30
    vc_cnt = Counter(s["vc"] for s in segs)
    nc_cnt = Counter(s["nc"] for s in segs)
    top_v = {c for c,_ in vc_cnt.most_common(NV)}
    top_n = {c for c,_ in nc_cnt.most_common(NN)}
    v_map = {c:i for i,(c,_) in enumerate(vc_cnt.most_common(NV))}
    n_map = {c:i for i,(c,_) in enumerate(nc_cnt.most_common(NN))}
    segs = [s for s in segs if s["vc"] in top_v and s["nc"] in top_n]
    print(f"After filtering: {len(segs)} segments")

    # Print verb names
    vnames = {}
    for s in segs:
        if s["vc"] in v_map: vnames[v_map[s["vc"]]] = s["verb"]
    print(f"Verbs: {dict(sorted(vnames.items()))}")

    # Extract features
    T = 8
    ext = Extractor()
    ac, ao, ah, af, av, an = [], [], [], [], [], []
    skip = 0

    for i, s in enumerate(segs):
        dur = s["stop"] - s["start"]
        if dur < 0.3: skip+=1; continue

        cap = cv2.VideoCapture(s["path"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: cap.release(); skip+=1; continue

        sf, ef = int(s["start"]*fps), int(s["stop"]*fps)
        total = ef - sf
        if total < 2: cap.release(); skip+=1; continue

        idxs = [int(sf + j*total/T) for j in range(T)] if total>=T else list(range(sf,ef))
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, fr = cap.read()
            if ret: frames.append(fr)
        cap.release()

        if len(frames) < 2: skip+=1; continue

        cf = np.zeros((T,512), dtype=np.float32)
        of = np.zeros((T,80), dtype=np.float32)
        hf = np.zeros((T,42), dtype=np.float32)
        ff = np.zeros((T,8), dtype=np.float32)
        pg = None
        for j, fr in enumerate(frames[:T]):
            cf[j] = ext.clip_feat(fr)
            of[j] = ext.obj_feat(fr)
            hf[j] = ext.hand_feat(fr)
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ff[j] = ext.flow_feat(pg, g)
            pg = g

        ac.append(cf); ao.append(of); ah.append(hf); af.append(ff)
        av.append(v_map[s["vc"]]); an.append(n_map[s["nc"]])

        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{len(segs)}] extracted={len(ac)} skipped={skip}")

    data = {
        "clip_features": torch.tensor(np.stack(ac)),
        "obj_features": torch.tensor(np.stack(ao)),
        "hand_features": torch.tensor(np.stack(ah)),
        "flow_features": torch.tensor(np.stack(af)),
        "verb_labels": torch.tensor(av, dtype=torch.long),
        "noun_labels": torch.tensor(an, dtype=torch.long),
        "verb_remap": v_map, "noun_remap": n_map,
        "config": {"frames_per_clip": T, "top_n_verbs": NV, "top_n_nouns": NN},
    }
    torch.save(data, FEAT_PATH)
    sz = os.path.getsize(FEAT_PATH)/1e6
    print(f"\n✅ Saved {len(ac)} clips to {FEAT_PATH} ({sz:.1f} MB)")
    return data


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

class DS(Dataset):
    def __init__(self, d):
        self.c,self.o,self.h,self.f = d["clip_features"],d["obj_features"],d["hand_features"],d["flow_features"]
        self.v,self.n = d["verb_labels"],d["noun_labels"]
    def __len__(self): return len(self.v)
    def __getitem__(self,i): return self.c[i],self.o[i],self.h[i],self.f[i],self.v[i],self.n[i]

def train(data, epochs=50, bs=64, lr=1e-3):
    cfg = data["config"]
    ds = DS(data)
    vn = int(len(ds)*0.2); tn = len(ds)-vn
    tds, vds = random_split(ds, [tn,vn], generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tds, bs, shuffle=True)
    vl = DataLoader(vds, bs)

    model = EgoModel(cfg["top_n_verbs"], cfg["top_n_nouns"], 128, cfg["frames_per_clip"]).to(DEVICE)
    print(f"\nTraining: {model.count_params():,} params | {tn} train / {vn} val | device={DEVICE}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    vce = nn.CrossEntropyLoss(); nce = nn.CrossEntropyLoss()

    best = 0.0
    print(f"\n{'Ep':>3} | {'TrLoss':>7} | {'TrV%':>5} | {'TrN%':>5} | {'VaLoss':>7} | {'VaV%':>5} | {'VaN%':>5}")
    print("-"*55)

    for ep in range(1, epochs+1):
        model.train()
        tl_,tvc,tnc,tt = 0,0,0,0
        for c,o,h,f,vy,ny in tl:
            c,o,h,f,vy,ny = c.to(DEVICE),o.to(DEVICE),h.to(DEVICE),f.to(DEVICE),vy.to(DEVICE),ny.to(DEVICE)
            opt.zero_grad()
            vp,np_ = model(c,o,h,f)
            loss = 0.6*vce(vp,vy) + 0.4*nce(np_,ny)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tl_ += loss.item()*c.size(0); tvc += (vp.argmax(1)==vy).sum().item()
            tnc += (np_.argmax(1)==ny).sum().item(); tt += c.size(0)

        model.eval()
        vl_,vvc,vnc,vt = 0,0,0,0
        with torch.no_grad():
            for c,o,h,f,vy,ny in vl:
                c,o,h,f,vy,ny = c.to(DEVICE),o.to(DEVICE),h.to(DEVICE),f.to(DEVICE),vy.to(DEVICE),ny.to(DEVICE)
                vp,np_ = model(c,o,h,f)
                loss = 0.6*vce(vp,vy) + 0.4*nce(np_,ny)
                vl_ += loss.item()*c.size(0); vvc += (vp.argmax(1)==vy).sum().item()
                vnc += (np_.argmax(1)==ny).sum().item(); vt += c.size(0)

        sch.step()
        va = vvc/max(vt,1)
        print(f"{ep:>3} | {tl_/tt:>7.4f} | {tvc/tt:>4.0%} | {tnc/tt:>4.0%} | {vl_/max(vt,1):>7.4f} | {va:>4.0%} | {vnc/max(vt,1):>4.0%}")

        if va > best:
            best = va
            torch.save({"epoch":ep, "model_state_dict":model.state_dict(),
                "verb_remap":data["verb_remap"], "noun_remap":data["noun_remap"],
                "config":cfg, "embed_dim":128, "best_verb_acc":best},
                f"{CKPT_DIR}/best_model.pt")
            print(f"  → Saved best ({best:.1%})")

    print(f"\n✅ Training complete! Best verb accuracy: {best:.1%}")
    return model


# ═══════════════════════════════════════════════════════════════
# ABLATION STUDY (for research paper)
# ═══════════════════════════════════════════════════════════════

def ablation(model, data):
    ds = DS(data)
    vn = int(len(ds)*0.2)
    _, vds = random_split(ds, [len(ds)-vn, vn], generator=torch.Generator().manual_seed(42))
    dl = DataLoader(vds, 64)

    def eval_mask(zero_idx=None):
        vc,nc,tot = 0,0,0
        with torch.no_grad():
            for c,o,h,f,vy,ny in dl:
                ins = [c.to(DEVICE),o.to(DEVICE),h.to(DEVICE),f.to(DEVICE)]
                if zero_idx is not None: ins[zero_idx] = torch.zeros_like(ins[zero_idx])
                vp,np_ = model(*ins)
                vc += (vp.argmax(1)==vy.to(DEVICE)).sum().item()
                nc += (np_.argmax(1)==ny.to(DEVICE)).sum().item()
                tot += c.size(0)
        return vc/tot, nc/tot

    print(f"\n{'='*55}")
    print("ABLATION STUDY — Each modality's contribution")
    print(f"{'='*55}")
    print(f"{'Variant':<30} | {'Verb%':>6} | {'Noun%':>6} | {'Drop':>6}")
    print("-"*55)

    fv, fn = eval_mask(None)
    print(f"{'Full model':<30} | {fv:>5.1%} | {fn:>5.1%} |    —")
    for i, name in enumerate(["CLIP","YOLO Objects","Hand Pose","Optical Flow"]):
        v, n = eval_mask(i)
        print(f"{'Without '+name:<30} | {v:>5.1%} | {n:>5.1%} | {fv-v:>+5.1%}")

    # Save results
    with open(f"{RESULTS_DIR}/ablation.txt", "w") as f_out:
        f_out.write(f"Full: verb={fv:.4f} noun={fn:.4f}\n")
        for i, name in enumerate(["CLIP","YOLO","Hand","Flow"]):
            v, n = eval_mask(i)
            f_out.write(f"Without {name}: verb={v:.4f} noun={n:.4f} drop={fv-v:+.4f}\n")
    print(f"\n✅ Results saved to {RESULTS_DIR}/ablation.txt")


# ═══════════════════════════════════════════════════════════════
# RUN EVERYTHING
# ═══════════════════════════════════════════════════════════════

data = load_data()
model = train(data)
ablation(model, data)

# Save to Google Drive if mounted
if os.path.exists("/content/drive/MyDrive"):
    import shutil
    shutil.copy(f"{CKPT_DIR}/best_model.pt", "/content/drive/MyDrive/best_model.pt")
    if os.path.exists(f"{RESULTS_DIR}/ablation.txt"):
        shutil.copy(f"{RESULTS_DIR}/ablation.txt", "/content/drive/MyDrive/ablation.txt")
    print("\n✅ Model + results saved to Google Drive!")
else:
    print("\n⚠️  Google Drive not mounted. Download best_model.pt manually from checkpoints/")

print("\n" + "="*55)
print("ALL DONE! You have:")
print(f"  1. Trained model:  {CKPT_DIR}/best_model.pt")
print(f"  2. Ablation study: {RESULTS_DIR}/ablation.txt")
print("="*55)
