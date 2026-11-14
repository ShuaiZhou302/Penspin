import json, numpy as np
P="<PROJECT_ROOT>/"
R=P+"penspin-student-fv5-20261106/runs/"; D=P+"penspin-superdex-penlen-20261031/runs/"
jobs=[l.split() for l in open(R+"jobs_fv5.txt") if l.strip()]
out={"ig":{},"sdx":{}}
v=[]
for t,tr,ev in jobs:
    r=[]
    for i in range(3):
        m=json.loads(open(f"{R}student_{ev}_{i}.out").read().split("PROTOCOL_AUDIT ")[-1].split("\n")[0])
        assert m["finished"]==64
        r.append((m["mean_rotation_reward_after_warmup"],m["mean_physical_duration_s"]))
    v.append(np.mean(r,0))
v=np.array(v); out["ig"]={"per_seed":v.round(2).tolist(),"mean":round(v[:,0].mean(),2),"sd":round(v[:,0].std(ddof=1),2),"duration_s":round(v[:,1].mean(),2)}
for mu in ["0.5","1.0","1.5"]:
    per=[];pi=0;tpi=0;N=0
    for t,tr,ev in jobs:
        r=[];d=[]
        for s in (42,43,44):
            j=json.load(open(f"{D}zs_mu{mu}_fv5_t{t}_s{s}.json"))
            assert tr in j["summary"]["options"]["ckpt"] and j["summary"]["episodes"]==64
            r.append(j["summary"]["sdx_rot_reward_mean"]); d.append(j["summary"]["sdx_duration_s_mean"])
            rr=np.array([e["sdx_rot_rad"] for e in j["episodes"]]); pi+=int((rr>=np.pi).sum()); tpi+=int((rr>=2*np.pi).sum()); N+=len(rr)
        per.append([np.mean(r),np.mean(d)])
    per=np.array(per)
    out["sdx"]["mu"+mu]={"per_seed":per.round(2).tolist(),"mean":round(per[:,0].mean(),2),"sd":round(per[:,0].std(ddof=1),2),"duration_s":round(per[:,1].mean(),2),"ge_pi":pi,"ge_2pi":tpi,"N":N}
print(json.dumps(out))
