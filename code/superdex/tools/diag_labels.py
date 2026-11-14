import sys,glob,json,numpy as np,torch
sys.path.insert(0,"tools")
from train_f import FNet,f1_scores,THRESH
torch.set_num_threads(2)
OLD="../penspin-superdex-port-20261028/"
ck=torch.load("../penspin-student-falign-20261029/f_models/f_v3_tps_b100_s0.pt",map_location="cpu")
net=FNet(ck["din"],np.zeros(ck["din"],np.float32),np.ones(ck["din"],np.float32)); net.load_state_dict(ck["state_dict"]); net.eval()
cal=np.load(OLD+"runs/calib/calib_b100.npz"); taus=cal["taus"]; lag=int(cal["lag"])
def load(prefix):
    fs=sorted(glob.glob(prefix))
    ds=[np.load(f) for f in fs]
    return np.concatenate([d["episodes"] for d in ds]),np.concatenate([d["lengths"] for d in ds]),np.concatenate([d["resync_contact"] for d in ds]),np.concatenate([d["resync_pen_pos_err"] for d in ds])
SETS={"val_teacher":["pairT_seed102_777112"],"shift_BT":["pairBT_seed103_777113"],"own_f_student":["evalFA_seed42_777491","evalFA_seed43_777491","evalFA_seed44_777491"]}
k=np.arange(32)!=11
res={}
for name,recs in SETS.items():
    X=[];I=[];Cal=[];Y=[];Yo=[];R=[];E=[]
    for rec in recs:
        r=dict(np.load("isaac/runs/record_%s.npz"%rec))
        ep,ln,y,e=load("runs/pairs/%s_*.npz"%rec)
        haveold=glob.glob(OLD+"runs/pairs_v2/%s_*.npz"%rec)
        yo=load(OLD+"runs/pairs_v2/%s_*.npz"%rec)[2] if haveold else None
        pc=r["obj_point_cloud_local"]; ra=np.round((pc[...,2].max(1)-pc[...,2].min(1))/r["obj_scale"].astype(float),2)
        c=np.log1p(r["contact_norm"]/THRESH); a=np.clip(r["action"],-1,1); cn=r["contact_norm"]
        for E0,L in zip(ep,ln):
            for t in range(L):
                h1=c[E0,t-1] if t>=1 else 0*c[E0,t]; h2=c[E0,t-2] if t>=2 else 0*c[E0,t]; qp=r["q"][E0,t-1] if t>=1 else r["q"][E0,t]
                X.append(np.concatenate([c[E0,t],h1,h2,r["q"][E0,t],qp,r["target"][E0,t],a[E0,t]]))
                I.append(cn[E0,t]>=THRESH); Cal.append((cn[E0,t-lag] if t>=lag else 0*cn[E0,t])>=taus); R.append(ra[E0])
        Y.append(y>=THRESH); E.append(e)
        if yo is not None: Yo.append(yo>=THRESH)
    with torch.no_grad(): P=(net(torch.tensor(np.array(X),dtype=torch.float32))>0).numpy()
    I=np.array(I);Cal=np.array(Cal);Y=np.concatenate(Y);R=np.array(R)
    out={"rows":len(Y),"on_new":float(Y[:,k].mean()),"on_ig":float(I[:,k].mean()),"on_f":float(P[:,k].mean()),"on_calib":float(Cal[:,k].mean()),"pen_err_med_cm":float(np.median(np.concatenate(E))*100)}
    for n,p in (("identity",I),("calib",Cal),("f",P)):
        s=f1_scores(p,Y); out[n+"_F1_new"]=round(s["micro_f1_ex_pen"],4); out[n+"_prec_new"]=round(s["precision_ex_pen"],4); out[n+"_rec_new"]=round(s["recall_ex_pen"],4)
    if Yo:
        Yo=np.concatenate(Yo); out["on_old"]=float(Yo[:,k].mean())
        for n,p in (("identity",I),("calib",Cal),("f",P)): out[n+"_F1_old"]=round(f1_scores(p,Yo)["micro_f1_ex_pen"],4)
        out["old_as_pred_of_new_F1"]=round(f1_scores(Yo,Y)["micro_f1_ex_pen"],4)
        out["by_len"]={}
        for l in sorted(set(R.tolist())):
            m=R==l
            out["by_len"][l]={"rows":int(m.sum()),"on_old":round(float(Yo[m][:,k].mean()),4),"on_new":round(float(Y[m][:,k].mean()),4),"old_new_F1":round(f1_scores(Yo[m],Y[m])["micro_f1_ex_pen"],4),"f_F1_new":round(f1_scores(P[m],Y[m])["micro_f1_ex_pen"],4),"identity_F1_new":round(f1_scores(I[m],Y[m])["micro_f1_ex_pen"],4)}
    else:
        out["by_len"]={l:{"rows":int((R==l).sum()),"f_F1_new":round(f1_scores(P[R==l],Y[R==l])["micro_f1_ex_pen"],4),"identity_F1_new":round(f1_scores(I[R==l],Y[R==l])["micro_f1_ex_pen"],4)} for l in sorted(set(R.tolist()))}
    res[name]=out; print(name,json.dumps(out,indent=0).replace("\n"," "),flush=True)
json.dump(res,open("runs/diag/label_compare.json","w"),indent=1)
