#!/usr/bin/env python3
"""Exp 31: Freeze backbone in Stage 2 + prototype eval (Rank 8).

KEY CHANGE: S2_BACKBONE_LR_MULT = 0.0 (completely frozen backbone).
Train only: target read-in + head. Frozen backbone preserves cross-patient
cluster structure that prototype eval exploits.

exp05 tried this with linear eval and got 0.804. With prototype eval it may be better.
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

DEVICE = os.environ.get("DEVICE", "mps"); SEED = 42
S1_EPOCHS=200;S1_BATCH_SIZE=16;S1_LR=1e-3;S1_READIN_LR_MULT=3.0;S1_WEIGHT_DECAY=1e-4;S1_GRAD_CLIP=5.0;S1_WARMUP_EPOCHS=20;S1_PATIENCE=7;S1_EVAL_EVERY=10;S1_VAL_FRACTION=0.2
S2_EPOCHS=150;S2_BATCH_SIZE=16;S2_LR=1e-3
S2_BACKBONE_LR_MULT = 0.0  # <<< FROZEN
S2_READIN_LR_MULT=3.0;S2_WEIGHT_DECAY=1e-4;S2_GRAD_CLIP=5.0;S2_WARMUP_EPOCHS=10;S2_PATIENCE=10;S2_EVAL_EVERY=5  # more patience since fewer params
LABEL_SMOOTHING=0.1;FOCAL_GAMMA=2.0;MIXUP_ALPHA=0.2;TTA_COPIES=16;KNN_K=10

def time_shift(x, mf=30):
    if mf==0: return x
    B,H,W,T=x.shape;shifts=torch.randint(-mf,mf+1,(B,));out=torch.zeros_like(x)
    for i in range(B):
        s=shifts[i].item()
        if s>0: out[i,:,:,s:]=x[i,:,:,:T-s]
        elif s<0: out[i,:,:,:T+s]=x[i,:,:,-s:]
        else: out[i]=x[i]
    return out
def amplitude_scale(x,std=0.3):
    if std==0: return x
    return x*torch.exp(torch.randn(x.shape[0],x.shape[1],x.shape[2],1,device=x.device)*std)
def channel_dropout(x,mp=0.2):
    if mp==0: return x
    B,H,W,_=x.shape;p=torch.rand(1).item()*mp;return x*(torch.rand(B,H,W,device=x.device)>p).float().unsqueeze(-1)
def gaussian_noise(x,frac=0.05):
    if frac==0: return x
    return x+torch.randn_like(x)*(frac*x.std())
def temporal_stretch(x,mr=0.15):
    if mr==0: return x
    B,H,W,T=x.shape;out=torch.zeros_like(x)
    for i in range(B):
        rate=1.0+(torch.rand(1).item()*2-1)*mr;nT=max(int(round(T*rate)),2)
        flat=x[i].reshape(-1,1,T);s=F.interpolate(flat,size=nT,mode="linear",align_corners=False);L=min(nT,T);out[i,:,:,:L]=s[:,0,:L].reshape(H,W,L)
    return out
def augment(x): return gaussian_noise(channel_dropout(amplitude_scale(temporal_stretch(time_shift(x)))))

class SpatialReadIn(nn.Module):
    def __init__(self,gh,gw,C=8,ph=4,pw=8):
        super().__init__();self.conv=nn.Conv2d(1,C,3,padding=1);self.pool=nn.AdaptiveAvgPool2d((ph,pw));self.d_flat=C*ph*pw
    def forward(self,x):
        B,H,W,T=x.shape;x=x.permute(0,3,1,2).reshape(B*T,1,H,W);x=F.relu(self.conv(x))
        if x.device.type=="mps":x=self.pool(x.cpu()).to("mps")
        else:x=self.pool(x)
        return x.reshape(B,T,-1).permute(0,2,1)
class Backbone(nn.Module):
    def __init__(self,d_in=256,d=32,gh=32,gl=2,stride=10,gd=0.3,fdm=0.3,tmn=2,tmx=5):
        super().__init__();self.ln=nn.LayerNorm(d_in);self.temporal_conv=nn.Sequential(nn.Conv1d(d_in,d,kernel_size=stride,stride=stride),nn.GELU());self.gru=nn.GRU(d,gh,num_layers=gl,batch_first=True,bidirectional=True,dropout=gd if gl>1 else 0.0);self.out_dim=gh*2;self.fdm=fdm;self.tmn=tmn;self.tmx=tmx
    def forward(self,x):
        x=self.ln(x.permute(0,2,1)).permute(0,2,1)
        if self.training and self.fdm>0:
            p=torch.rand(1).item()*self.fdm;mask=(torch.rand(x.shape[1],device=x.device)>p).float();x=x*mask.unsqueeze(0).unsqueeze(-1)/(1-p+1e-8)
        x=self.temporal_conv(x).permute(0,2,1)
        if self.training and self.tmx>0:
            T=x.shape[1];ml=torch.randint(self.tmn,self.tmx+1,(1,)).item();st=torch.randint(0,max(T-ml,1),(1,)).item();taper=0.5*(1-torch.cos(torch.linspace(0,torch.pi,ml,device=x.device)));x=x.clone();x[:,st:st+ml,:]*=(1-taper).unsqueeze(-1)
        h,_=self.gru(x);return h
class CEHead(nn.Module):
    def __init__(self,d_in=64,np_=3,nc=9,do=0.3):
        super().__init__();self.drop=nn.Dropout(do);self.heads=nn.ModuleList([nn.Linear(d_in,nc) for _ in range(np_)])
    def forward(self,h):p=self.drop(h.mean(dim=1));return torch.stack([hd(p) for hd in self.heads],dim=1)

def focal_ce(logits,targets,gamma=FOCAL_GAMMA):
    ce=F.cross_entropy(logits,targets,label_smoothing=LABEL_SMOOTHING,reduction="none")
    if gamma>0:pt=torch.exp(-ce);ce=((1-pt)**gamma)*ce
    return ce.mean()
def compute_loss(logits,labels,ml=None,lam=1.0):
    loss=torch.tensor(0.0,device=logits.device)
    for pos in range(prepare.N_POSITIONS):
        tgt=torch.tensor([l[pos]-1 for l in labels],dtype=torch.long,device=logits.device);pl=focal_ce(logits[:,pos,:],tgt)
        if ml is not None:tgt2=torch.tensor([l[pos]-1 for l in ml],dtype=torch.long,device=logits.device);pl=lam*pl+(1-lam)*focal_ce(logits[:,pos,:],tgt2)
        loss=loss+pl
    return loss/prepare.N_POSITIONS
def decode(logits):return (logits.argmax(dim=-1)+1).cpu().tolist()
def extract_embeddings(bb,ri,grids):
    bb.eval();ri.eval()
    with torch.no_grad():return bb(ri(grids.to(DEVICE))).mean(dim=1).cpu()
def knn_predict(te,tl,ve,k=KNN_K):
    sim=F.normalize(ve,dim=1)@F.normalize(te,dim=1).T;ts,ti=sim.topk(k,dim=1);preds=[]
    for i in range(ve.shape[0]):
        pred=[]
        for pos in range(prepare.N_POSITIONS):
            cw=[0.0]*(prepare.N_CLASSES+1)
            for j in range(k):cw[tl[ti[i,j].item()][pos]]+=ts[i,j].item()
            pred.append(int(np.argmax(cw[1:])+1))
        preds.append(pred)
    return preds
def prototype_predict(te,tl,ve):
    tn=F.normalize(te,dim=1);vn=F.normalize(ve,dim=1);preds=[]
    for i in range(vn.shape[0]):
        pred=[]
        for pos in range(prepare.N_POSITIONS):
            bs,bc=-float("inf"),1
            for cls in range(1,prepare.N_CLASSES+1):
                mask=[j for j,l in enumerate(tl) if l[pos]==cls]
                if not mask:continue
                c=F.normalize(tn[mask].mean(dim=0),dim=0);s=(vn[i]*c).sum().item()
                if s>bs:bs,bc=s,cls
            pred.append(bc)
        preds.append(pred)
    return preds

def train_stage1(all_data):
    torch.manual_seed(SEED);np.random.seed(SEED)
    read_ins={};
    for pid in prepare.SOURCE_PATIENTS:H,W=all_data[pid]["grid_shape"];read_ins[pid]=SpatialReadIn(H,W).to(DEVICE)
    d_flat=list(read_ins.values())[0].d_flat;backbone=Backbone(d_in=d_flat).to(DEVICE);head=CEHead(d_in=backbone.out_dim).to(DEVICE)
    st,sv={},{}
    for pid in prepare.SOURCE_PATIENTS:
        n=len(all_data[pid]["labels"]);perm=np.random.permutation(n);nv=max(1,int(round(S1_VAL_FRACTION*n)));st[pid]=sorted(perm[nv:].tolist());sv[pid]=sorted(perm[:nv].tolist())
    rp=[];
    for ri in read_ins.values():rp.extend(ri.parameters())
    opt=AdamW([{"params":rp,"lr":S1_LR*S1_READIN_LR_MULT},{"params":backbone.parameters(),"lr":S1_LR},{"params":head.parameters(),"lr":S1_LR}],weight_decay=S1_WEIGHT_DECAY)
    def lrl(e):
        if S1_WARMUP_EPOCHS>0 and e<S1_WARMUP_EPOCHS:return (e+1)/S1_WARMUP_EPOCHS
        return 0.5*(1+math.cos(math.pi*(e-S1_WARMUP_EPOCHS)/max(S1_EPOCHS-S1_WARMUP_EPOCHS,1)))
    sch=LambdaLR(opt,lrl);bvl=float("inf");bs_=None;pc=0;pids=prepare.SOURCE_PATIENTS
    for epoch in range(S1_EPOCHS):
        backbone.train();head.train()
        for ri in read_ins.values():ri.train()
        np.random.shuffle(pids);el=0.0;nb=0
        for pid in pids:
            g=all_data[pid]["grids"];la=all_data[pid]["labels"];ti=st[pid];pm=np.random.permutation(len(ti))
            for s in range(0,len(ti),S1_BATCH_SIZE):
                bi=[ti[pm[i]] for i in range(s,min(s+S1_BATCH_SIZE,len(ti)))];x=augment(g[bi]).to(DEVICE);y=[la[i] for i in bi]
                my,lam=None,1.0
                if MIXUP_ALPHA>0 and len(bi)>1:lam=float(np.random.beta(MIXUP_ALPHA,MIXUP_ALPHA));pmx=torch.randperm(x.shape[0]);x=lam*x+(1-lam)*x[pmx];my=[y[i] for i in pmx.tolist()]
                opt.zero_grad();loss=compute_loss(head(backbone(read_ins[pid](x))),y,ml=my,lam=lam);loss.backward()
                nn.utils.clip_grad_norm_(list(backbone.parameters())+list(head.parameters())+rp,S1_GRAD_CLIP);opt.step();el+=loss.item();nb+=1
        sch.step()
        if (epoch+1)%S1_EVAL_EVERY==0:
            backbone.eval();head.eval()
            for ri in read_ins.values():ri.eval()
            vl=0.0;vb=0
            with torch.no_grad():
                for pid in prepare.SOURCE_PATIENTS:
                    vi=sv[pid]
                    if not vi:continue
                    vl+=compute_loss(head(backbone(read_ins[pid](all_data[pid]["grids"][vi].to(DEVICE)))),[all_data[pid]["labels"][i] for i in vi]).item();vb+=1
            vl/=max(vb,1);print(f"  S1 epoch {epoch+1}: train={el/max(nb,1):.4f}, val={vl:.4f}")
            if vl<bvl:bvl=vl;bs_={"bb":deepcopy(backbone.state_dict()),"hd":deepcopy(head.state_dict()),"ri":{pid:deepcopy(ri.state_dict()) for pid,ri in read_ins.items()}};pc=0
            else:pc+=1
            if pc>=S1_PATIENCE:print(f"  S1 early stop at epoch {epoch+1}");break
    if bs_:backbone.load_state_dict(bs_["bb"]);head.load_state_dict(bs_["hd"])
    for pid,ri in read_ins.items():
        if pid in bs_["ri"]:ri.load_state_dict(bs_["ri"][pid])
    return backbone,head,read_ins

def train_eval_fold(backbone,head_init,tg,tl,vg,vl,read_ins=None,all_data=None):
    bc=deepcopy(backbone);tri=SpatialReadIn(*prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT]).to(DEVICE);hd=deepcopy(head_init).to(DEVICE)
    # FREEZE backbone — keep train mode (cuDNN GRU needs it for backward)
    # but disable gradients so no params update
    for p in bc.parameters(): p.requires_grad = False
    # Disable stochastic regularization manually
    bc.fdm = 0.0; bc.tmx = 0

    opt=AdamW([{"params":tri.parameters(),"lr":S2_LR*S2_READIN_LR_MULT},{"params":hd.parameters(),"lr":S2_LR}],weight_decay=S2_WEIGHT_DECAY)
    def lrl(e):
        if S2_WARMUP_EPOCHS>0 and e<S2_WARMUP_EPOCHS:return (e+1)/S2_WARMUP_EPOCHS
        return 0.5*(1+math.cos(math.pi*(e-S2_WARMUP_EPOCHS)/max(S2_EPOCHS-S2_WARMUP_EPOCHS,1)))
    sch=LambdaLR(opt,lrl);bvl=float("inf");bs=None;pc=0
    for ep in range(S2_EPOCHS):
        tri.train();hd.train();pm=torch.randperm(len(tg))
        for st in range(0,len(tg),S2_BATCH_SIZE):
            idx=pm[st:st+S2_BATCH_SIZE];x=augment(tg[idx]).to(DEVICE);y=[tl[i] for i in idx.tolist()]
            my,lam=None,1.0
            if MIXUP_ALPHA>0 and len(idx)>1:lam=float(np.random.beta(MIXUP_ALPHA,MIXUP_ALPHA));pmx=torch.randperm(x.shape[0]);x=lam*x+(1-lam)*x[pmx];my=[y[i] for i in pmx.tolist()]
            opt.zero_grad()
            with torch.no_grad(): feat = bc(tri(x))
            loss=compute_loss(hd(feat.detach()),y,ml=my,lam=lam)
            if math.isnan(loss.item()):print("FAIL");sys.exit(1)
            loss.backward();nn.utils.clip_grad_norm_(list(tri.parameters())+list(hd.parameters()),S2_GRAD_CLIP);opt.step()
        sch.step()
        if (ep+1)%S2_EVAL_EVERY==0:
            tri.eval();hd.eval()
            with torch.no_grad():vl_=compute_loss(hd(bc(tri(vg.to(DEVICE)))),vl).item()
            if vl_<bvl:bvl=vl_;bs={"ri":deepcopy(tri.state_dict()),"hd":deepcopy(hd.state_dict())};pc=0
            else:pc+=1
            if pc>=S2_PATIENCE:break
    if bs:tri.load_state_dict(bs["ri"]);hd.load_state_dict(bs["hd"])
    tri.eval();hd.eval()
    with torch.no_grad():
        ls=torch.zeros(len(vg),prepare.N_POSITIONS,prepare.N_CLASSES,device=DEVICE);ls+=hd(bc(tri(vg.to(DEVICE))))
        for _ in range(TTA_COPIES-1):ls+=hd(bc(tri(augment(vg).to(DEVICE))))
    logits=ls/TTA_COPIES;lp=prepare.compute_per(decode(logits),vl)
    te=extract_embeddings(bc,tri,tg);ve=extract_embeddings(bc,tri,vg)
    for _ in range(TTA_COPIES-1):ve=ve+extract_embeddings(bc,tri,augment(vg))
    ve=ve/TTA_COPIES
    kp=prepare.compute_per(knn_predict(te,list(tl),ve),vl);pp=prepare.compute_per(prototype_predict(te,list(tl),ve),vl)
    methods={"linear":(lp,decode(logits)),"knn":(kp,knn_predict(te,list(tl),ve)),"prototype":(pp,prototype_predict(te,list(tl),ve))}
    bn=min(methods,key=lambda k:methods[k][0]);return methods[bn][0],methods[bn][1],methods

if __name__=="__main__":
    t0=time.time();torch.manual_seed(SEED);np.random.seed(SEED)
    all_data=prepare.load_all_patients();grids,labels,token_ids=prepare.load_target_data();splits=prepare.create_cv_splits(token_ids)
    N,H,W,T=grids.shape
    print(f"Target: {prepare.TARGET_PATIENT}  |  Trials: {N}  |  Grid: {H}x{W}  |  T: {T}")
    print(f"Source: {len(prepare.SOURCE_PATIENTS)}  |  Device: {DEVICE}")
    print(f"CHANGE: Freeze backbone in S2 (backbone_lr=0) + prototype eval")
    print()
    print("=== Stage 1 ===");s1t=time.time();backbone,head,read_ins=train_stage1(all_data);s1_time=time.time()-s1t;print(f"Stage 1 done in {s1_time:.1f}s\n")
    print("=== Stage 2 (frozen backbone) ===")
    fps=[];ap=[];mps={m:[] for m in ["linear","knn","prototype"]}
    for fi,(tri,vai) in enumerate(splits):
        ft0=time.time();per,preds,methods=train_eval_fold(backbone,head,grids[tri],[labels[i] for i in tri],grids[vai],[labels[i] for i in vai],read_ins=read_ins,all_data=all_data)
        fps.append(per);ap.extend(preds)
        for m,(p,_) in methods.items():mps[m].append(p)
        bm=min(methods,key=lambda k:methods[k][0])
        print(f"  Fold {fi+1}: PER={per:.4f} (lin={methods['linear'][0]:.4f} knn={methods['knn'][0]:.4f} proto={methods['prototype'][0]:.4f} best={bm})  ({time.time()-ft0:.1f}s)")
    mp=float(np.mean(fps));col=prepare.compute_content_collapse(ap)
    print(f"\n--- Per-method ---")
    for m,ps in mps.items():print(f"  {m}: {np.mean(ps):.4f} ± {np.std(ps):.4f}")
    print(f"\n---\nval_per:            {mp:.6f}\nval_per_std:        {float(np.std(fps)):.6f}\nfold_pers:          {fps}")
    for m,ps in mps.items():print(f"{m}_pers:       {ps}")
    print(f"stage1_seconds:     {s1_time:.1f}\ntraining_seconds:   {time.time()-t0:.1f}\ncollapsed:          {col['collapsed']}\nmean_entropy:       {col['mean_entropy']:.3f}\nstereotypy:         {col['stereotypy']:.3f}\nunique_ratio:       {col['unique_ratio']:.3f}")
