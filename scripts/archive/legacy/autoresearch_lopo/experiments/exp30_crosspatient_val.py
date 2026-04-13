#!/usr/bin/env python3
"""Exp 30: Cross-patient validation in Stage 1 (Rank 4).

KEY CHANGE: Instead of within-patient 80/20 val split, hold out one entire
source patient as val. Early-stop based on TRANSFER quality to unseen patient.

Hold out S22 (8x16, same grid as S14) as permanent val patient.
This means S1 trains on 8 source patients, validates on S22.
"""
from __future__ import annotations
import math, os, sys, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare

DEVICE = os.environ.get("DEVICE", "mps")
SEED = 42
S1_EPOCHS = 200; S1_BATCH_SIZE = 16; S1_LR = 1e-3; S1_READIN_LR_MULT = 3.0
S1_WEIGHT_DECAY = 1e-4; S1_GRAD_CLIP = 5.0; S1_WARMUP_EPOCHS = 20
S1_PATIENCE = 7; S1_EVAL_EVERY = 10
S2_EPOCHS = 150; S2_BATCH_SIZE = 16; S2_LR = 1e-3; S2_BACKBONE_LR_MULT = 0.1
S2_READIN_LR_MULT = 3.0; S2_WEIGHT_DECAY = 1e-4; S2_GRAD_CLIP = 5.0
S2_WARMUP_EPOCHS = 10; S2_PATIENCE = 7; S2_EVAL_EVERY = 5
LABEL_SMOOTHING = 0.1; FOCAL_GAMMA = 2.0; MIXUP_ALPHA = 0.2
TTA_COPIES = 16; KNN_K = 10

VAL_PATIENT = "S22"  # Hold out as cross-patient val (same 8x16 grid as S14)

def time_shift(x, mf=30):
    if mf == 0: return x
    B,H,W,T = x.shape; shifts = torch.randint(-mf, mf+1, (B,)); out = torch.zeros_like(x)
    for i in range(B):
        s = shifts[i].item()
        if s > 0: out[i,:,:,s:] = x[i,:,:,:T-s]
        elif s < 0: out[i,:,:,:T+s] = x[i,:,:,-s:]
        else: out[i] = x[i]
    return out
def amplitude_scale(x, std=0.3):
    if std == 0: return x
    return x * torch.exp(torch.randn(x.shape[0],x.shape[1],x.shape[2],1,device=x.device)*std)
def channel_dropout(x, mp=0.2):
    if mp == 0: return x
    B,H,W,_ = x.shape; p = torch.rand(1).item()*mp
    return x * (torch.rand(B,H,W,device=x.device) > p).float().unsqueeze(-1)
def gaussian_noise(x, frac=0.05):
    if frac == 0: return x
    return x + torch.randn_like(x)*(frac*x.std())
def temporal_stretch(x, mr=0.15):
    if mr == 0: return x
    B,H,W,T = x.shape; out = torch.zeros_like(x)
    for i in range(B):
        rate = 1.0+(torch.rand(1).item()*2-1)*mr; nT = max(int(round(T*rate)),2)
        flat = x[i].reshape(-1,1,T); s = F.interpolate(flat,size=nT,mode="linear",align_corners=False)
        L = min(nT,T); out[i,:,:,:L] = s[:,0,:L].reshape(H,W,L)
    return out
def augment(x):
    return gaussian_noise(channel_dropout(amplitude_scale(temporal_stretch(time_shift(x)))))

class SpatialReadIn(nn.Module):
    def __init__(self, gh, gw, C=8, ph=4, pw=8):
        super().__init__(); self.conv = nn.Conv2d(1,C,3,padding=1); self.pool = nn.AdaptiveAvgPool2d((ph,pw)); self.d_flat = C*ph*pw
    def forward(self, x):
        B,H,W,T = x.shape; x = x.permute(0,3,1,2).reshape(B*T,1,H,W); x = F.relu(self.conv(x))
        if x.device.type == "mps": x = self.pool(x.cpu()).to("mps")
        else: x = self.pool(x)
        return x.reshape(B,T,-1).permute(0,2,1)

class Backbone(nn.Module):
    def __init__(self, d_in=256, d=32, gh=32, gl=2, stride=10, gd=0.3, fdm=0.3, tmn=2, tmx=5):
        super().__init__(); self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(nn.Conv1d(d_in,d,kernel_size=stride,stride=stride),nn.GELU())
        self.gru = nn.GRU(d,gh,num_layers=gl,batch_first=True,bidirectional=True,dropout=gd if gl>1 else 0.0)
        self.out_dim = gh*2; self.fdm=fdm; self.tmn=tmn; self.tmx=tmx
    def forward(self, x):
        x = self.ln(x.permute(0,2,1)).permute(0,2,1)
        if self.training and self.fdm > 0:
            p = torch.rand(1).item()*self.fdm; mask = (torch.rand(x.shape[1],device=x.device)>p).float()
            x = x*mask.unsqueeze(0).unsqueeze(-1)/(1-p+1e-8)
        x = self.temporal_conv(x).permute(0,2,1)
        if self.training and self.tmx > 0:
            T=x.shape[1]; ml=torch.randint(self.tmn,self.tmx+1,(1,)).item(); st=torch.randint(0,max(T-ml,1),(1,)).item()
            taper = 0.5*(1-torch.cos(torch.linspace(0,torch.pi,ml,device=x.device))); x=x.clone(); x[:,st:st+ml,:]*=(1-taper).unsqueeze(-1)
        h,_ = self.gru(x); return h

class CEHead(nn.Module):
    def __init__(self, d_in=64, np_=3, nc=9, do=0.3):
        super().__init__(); self.drop = nn.Dropout(do); self.heads = nn.ModuleList([nn.Linear(d_in,nc) for _ in range(np_)])
    def forward(self, h):
        p = self.drop(h.mean(dim=1)); return torch.stack([hd(p) for hd in self.heads], dim=1)

def focal_ce(logits, targets, gamma=FOCAL_GAMMA):
    ce = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING, reduction="none")
    if gamma > 0: pt = torch.exp(-ce); ce = ((1-pt)**gamma)*ce
    return ce.mean()
def compute_loss(logits, labels, ml=None, lam=1.0):
    loss = torch.tensor(0.0, device=logits.device)
    for pos in range(prepare.N_POSITIONS):
        tgt = torch.tensor([l[pos]-1 for l in labels], dtype=torch.long, device=logits.device)
        pl = focal_ce(logits[:,pos,:], tgt)
        if ml is not None:
            tgt2 = torch.tensor([l[pos]-1 for l in ml], dtype=torch.long, device=logits.device)
            pl = lam*pl + (1-lam)*focal_ce(logits[:,pos,:], tgt2)
        loss = loss + pl
    return loss / prepare.N_POSITIONS
def decode(logits): return (logits.argmax(dim=-1)+1).cpu().tolist()
def extract_embeddings(bb, ri, grids):
    bb.eval(); ri.eval()
    with torch.no_grad(): return bb(ri(grids.to(DEVICE))).mean(dim=1).cpu()
def knn_predict(te, tl, ve, k=KNN_K):
    sim = F.normalize(ve,dim=1) @ F.normalize(te,dim=1).T; ts, ti = sim.topk(k,dim=1)
    preds = []
    for i in range(ve.shape[0]):
        pred = []
        for pos in range(prepare.N_POSITIONS):
            cw = [0.0]*(prepare.N_CLASSES+1)
            for j in range(k): cw[tl[ti[i,j].item()][pos]] += ts[i,j].item()
            pred.append(int(np.argmax(cw[1:])+1))
        preds.append(pred)
    return preds
def prototype_predict(te, tl, ve):
    tn = F.normalize(te,dim=1); vn = F.normalize(ve,dim=1); preds = []
    for i in range(vn.shape[0]):
        pred = []
        for pos in range(prepare.N_POSITIONS):
            bs,bc = -float("inf"),1
            for cls in range(1,prepare.N_CLASSES+1):
                mask = [j for j,l in enumerate(tl) if l[pos]==cls]
                if not mask: continue
                c = F.normalize(tn[mask].mean(dim=0),dim=0); s = (vn[i]*c).sum().item()
                if s > bs: bs,bc = s,cls
            pred.append(bc)
        preds.append(pred)
    return preds

def train_stage1(all_data):
    torch.manual_seed(SEED); np.random.seed(SEED)
    # Train on all source patients EXCEPT val patient
    train_pids = [p for p in prepare.SOURCE_PATIENTS if p != VAL_PATIENT]
    all_pids = train_pids + [VAL_PATIENT]  # need read-in for val patient too

    read_ins = {}
    for pid in all_pids:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = SpatialReadIn(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat
    backbone = Backbone(d_in=d_flat).to(DEVICE)
    head = CEHead(d_in=backbone.out_dim).to(DEVICE)

    readin_params = []
    for ri in read_ins.values(): readin_params.extend(ri.parameters())
    optimizer = AdamW([
        {"params": readin_params, "lr": S1_LR * S1_READIN_LR_MULT},
        {"params": backbone.parameters(), "lr": S1_LR},
        {"params": head.parameters(), "lr": S1_LR},
    ], weight_decay=S1_WEIGHT_DECAY)
    def lr_lambda(epoch):
        if S1_WARMUP_EPOCHS > 0 and epoch < S1_WARMUP_EPOCHS: return (epoch+1)/S1_WARMUP_EPOCHS
        progress = (epoch-S1_WARMUP_EPOCHS)/max(S1_EPOCHS-S1_WARMUP_EPOCHS,1)
        return 0.5*(1+math.cos(math.pi*progress))
    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss = float("inf"); best_state = None; patience_ctr = 0

    for epoch in range(S1_EPOCHS):
        backbone.train(); head.train()
        for ri in read_ins.values(): ri.train()
        pids = list(train_pids); np.random.shuffle(pids)
        epoch_loss = 0.0; nb = 0
        for pid in pids:
            grids = all_data[pid]["grids"]; labels = all_data[pid]["labels"]
            perm = np.random.permutation(len(labels))
            for start in range(0, len(labels), S1_BATCH_SIZE):
                bi = [perm[i] for i in range(start, min(start+S1_BATCH_SIZE, len(labels)))]
                x = augment(grids[bi]).to(DEVICE); y = [labels[i] for i in bi]
                my, lam = None, 1.0
                if MIXUP_ALPHA > 0 and len(bi) > 1:
                    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                    pm = torch.randperm(x.shape[0]); x = lam*x + (1-lam)*x[pm]; my = [y[i] for i in pm.tolist()]
                optimizer.zero_grad()
                loss = compute_loss(head(backbone(read_ins[pid](x))), y, ml=my, lam=lam)
                loss.backward()
                nn.utils.clip_grad_norm_(list(backbone.parameters())+list(head.parameters())+readin_params, S1_GRAD_CLIP)
                optimizer.step(); epoch_loss += loss.item(); nb += 1
        scheduler.step()

        if (epoch+1) % S1_EVAL_EVERY == 0:
            backbone.eval(); head.eval()
            for ri in read_ins.values(): ri.eval()
            # CROSS-PATIENT VAL: evaluate on held-out patient S22
            with torch.no_grad():
                vg = all_data[VAL_PATIENT]["grids"].to(DEVICE)
                vl = all_data[VAL_PATIENT]["labels"]
                val_loss = compute_loss(head(backbone(read_ins[VAL_PATIENT](vg))), vl).item()
            print(f"  S1 epoch {epoch+1}: train={epoch_loss/max(nb,1):.4f}, xval({VAL_PATIENT})={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {"backbone": deepcopy(backbone.state_dict()), "head": deepcopy(head.state_dict()),
                              "read_ins": {pid: deepcopy(ri.state_dict()) for pid, ri in read_ins.items()}}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= S1_PATIENCE: print(f"  S1 early stop at epoch {epoch+1}"); break
    if best_state:
        backbone.load_state_dict(best_state["backbone"]); head.load_state_dict(best_state["head"])
        for pid, ri in read_ins.items():
            if pid in best_state["read_ins"]: ri.load_state_dict(best_state["read_ins"][pid])
    return backbone, head, read_ins

def train_eval_fold(backbone, head_init, tg, tl, vg, vl, read_ins=None, all_data=None):
    bc = deepcopy(backbone); tri = SpatialReadIn(*prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT]).to(DEVICE)
    hd = deepcopy(head_init).to(DEVICE)
    opt = AdamW([{"params":tri.parameters(),"lr":S2_LR*S2_READIN_LR_MULT},{"params":bc.parameters(),"lr":S2_LR*S2_BACKBONE_LR_MULT},{"params":hd.parameters(),"lr":S2_LR}],weight_decay=S2_WEIGHT_DECAY)
    def lr_l(e):
        if S2_WARMUP_EPOCHS>0 and e<S2_WARMUP_EPOCHS: return (e+1)/S2_WARMUP_EPOCHS
        return 0.5*(1+math.cos(math.pi*(e-S2_WARMUP_EPOCHS)/max(S2_EPOCHS-S2_WARMUP_EPOCHS,1)))
    sch = LambdaLR(opt, lr_l); bvl=float("inf"); bs=None; pc=0
    for ep in range(S2_EPOCHS):
        bc.train();tri.train();hd.train(); pm=torch.randperm(len(tg))
        for st in range(0,len(tg),S2_BATCH_SIZE):
            idx=pm[st:st+S2_BATCH_SIZE]; x=augment(tg[idx]).to(DEVICE); y=[tl[i] for i in idx.tolist()]
            my,lam=None,1.0
            if MIXUP_ALPHA>0 and len(idx)>1: lam=float(np.random.beta(MIXUP_ALPHA,MIXUP_ALPHA)); pmx=torch.randperm(x.shape[0]); x=lam*x+(1-lam)*x[pmx]; my=[y[i] for i in pmx.tolist()]
            opt.zero_grad(); loss=compute_loss(hd(bc(tri(x))),y,ml=my,lam=lam)
            if math.isnan(loss.item()): print("FAIL"); sys.exit(1)
            loss.backward(); nn.utils.clip_grad_norm_(list(bc.parameters())+list(tri.parameters())+list(hd.parameters()),S2_GRAD_CLIP); opt.step()
        sch.step()
        if (ep+1)%S2_EVAL_EVERY==0:
            bc.eval();tri.eval();hd.eval()
            with torch.no_grad(): vl_=compute_loss(hd(bc(tri(vg.to(DEVICE)))),vl).item()
            if vl_<bvl: bvl=vl_; bs={"bb":deepcopy(bc.state_dict()),"ri":deepcopy(tri.state_dict()),"hd":deepcopy(hd.state_dict())}; pc=0
            else: pc+=1
            if pc>=S2_PATIENCE: break
    if bs: bc.load_state_dict(bs["bb"]); tri.load_state_dict(bs["ri"]); hd.load_state_dict(bs["hd"])
    bc.eval();tri.eval();hd.eval()
    with torch.no_grad():
        ls = torch.zeros(len(vg),prepare.N_POSITIONS,prepare.N_CLASSES,device=DEVICE)
        ls += hd(bc(tri(vg.to(DEVICE))))
        for _ in range(TTA_COPIES-1): ls += hd(bc(tri(augment(vg).to(DEVICE))))
        logits = ls/TTA_COPIES
    lp = prepare.compute_per(decode(logits), vl)
    te = extract_embeddings(bc,tri,tg)
    ve = extract_embeddings(bc,tri,vg)
    for _ in range(TTA_COPIES-1): ve = ve + extract_embeddings(bc,tri,augment(vg))
    ve = ve/TTA_COPIES
    kp = prepare.compute_per(knn_predict(te,list(tl),ve), vl)
    pp = prepare.compute_per(prototype_predict(te,list(tl),ve), vl)
    methods={"linear":(lp,decode(logits)),"knn":(kp,knn_predict(te,list(tl),ve)),"prototype":(pp,prototype_predict(te,list(tl),ve))}
    bn = min(methods,key=lambda k:methods[k][0])
    return methods[bn][0], methods[bn][1], methods

if __name__ == "__main__":
    t0 = time.time(); torch.manual_seed(SEED); np.random.seed(SEED)
    all_data = prepare.load_all_patients(); grids,labels,token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)
    N,H,W,T = grids.shape
    print(f"Target: {prepare.TARGET_PATIENT}  |  Trials: {N}  |  Grid: {H}x{W}  |  T: {T}")
    print(f"Source: {len(prepare.SOURCE_PATIENTS)}  |  Val patient: {VAL_PATIENT}  |  Device: {DEVICE}")
    print(f"CHANGE: Cross-patient val in S1 (hold out {VAL_PATIENT})")
    print()
    print("=== Stage 1: Cross-patient validation ===")
    s1t = time.time(); backbone,head,read_ins = train_stage1(all_data); s1_time = time.time()-s1t
    print(f"Stage 1 done in {s1_time:.1f}s\n")
    print("=== Stage 2 ===")
    fps=[]; ap=[]; ar=[]; mps={m:[] for m in ["linear","knn","prototype"]}
    for fi,(tri,vai) in enumerate(splits):
        ft0=time.time()
        per,preds,methods = train_eval_fold(backbone,head,grids[tri],[labels[i] for i in tri],grids[vai],[labels[i] for i in vai],read_ins=read_ins,all_data=all_data)
        fps.append(per); ap.extend(preds); ar.extend([labels[i] for i in vai])
        for m,(p,_) in methods.items(): mps[m].append(p)
        bm=min(methods,key=lambda k:methods[k][0])
        print(f"  Fold {fi+1}: PER={per:.4f} (lin={methods['linear'][0]:.4f} knn={methods['knn'][0]:.4f} proto={methods['prototype'][0]:.4f} best={bm})  ({time.time()-ft0:.1f}s)")
    mp=float(np.mean(fps)); col=prepare.compute_content_collapse(ap)
    print(f"\n--- Per-method ---")
    for m,ps in mps.items(): print(f"  {m}: {np.mean(ps):.4f} ± {np.std(ps):.4f}")
    print(f"\n---\nval_per:            {mp:.6f}\nval_per_std:        {float(np.std(fps)):.6f}\nfold_pers:          {fps}")
    for m,ps in mps.items(): print(f"{m}_pers:       {ps}")
    print(f"stage1_seconds:     {s1_time:.1f}\ntraining_seconds:   {time.time()-t0:.1f}\ncollapsed:          {col['collapsed']}\nmean_entropy:       {col['mean_entropy']:.3f}\nstereotypy:         {col['stereotypy']:.3f}\nunique_ratio:       {col['unique_ratio']:.3f}")
