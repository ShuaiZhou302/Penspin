import json, glob, re, numpy as np
P='<PROJECT_ROOT>/'
def ig(repo, ev):
    v=[]
    for i in range(3):
        t=open(f'{P}{repo}/runs/student_{ev}_{i}.out').read()
        m=json.loads(t.split('PROTOCOL_AUDIT ')[-1].split('\n')[0])
        v.append((m['mean_rotation_reward_after_warmup'], m['mean_physical_duration_s']))
    return np.mean(v,0)
igs={'fv4':{1:ig('penspin-student-penlen-20261031',789401),2:ig('penspin-student-penlen-20261031',789407)},
     'cv4':{1:ig('penspin-student-penlen-20261031',789403),2:ig('penspin-student-penlen-20261031',789409)},
     'trand':{1:ig('penspin-student-penlen-20261031',789405),2:ig('penspin-student-penlen-20261031',789412)},
     'btactile':{1:ig('penspin-student-objpose-dagger-20261022',772992),2:ig('penspin-student-objpose-dagger-20261022',772995)}}
igs['fv4'][0]=ig('penspin-student-penlen-20261031',783807); igs['cv4'][0]=ig('penspin-student-penlen-20261031',783808)
out={}
D=P+'penspin-superdex-penlen-20261031/runs/'
for n in ['btactile','trand','cv4','fv4']:
    out[n]={'ig':{t:[round(float(x),2) for x in igs[n].get(t,[np.nan,np.nan])] for t in (0,1,2)}}
    for mu in ['0.5','1.0','1.5']:
        per={}
        for t in (0,1,2):
            r=[];d=[];pi=0;tpi=0;N=0
            for s in (42,43,44):
                f=D+(f'zs_mu{mu}_{n}_s{s}.json' if t==0 else f'zs_mu{mu}_{n}_t{t}_s{s}.json')
                j=json.load(open(f)); r.append(j['summary']['sdx_rot_reward_mean']); d.append(j['summary']['sdx_duration_s_mean'])
                rr=np.array([e['sdx_rot_rad'] for e in j['episodes']]); pi+=(rr>=np.pi).sum(); tpi+=(rr>=2*np.pi).sum(); N+=len(rr)
            per[t]=[round(float(np.mean(r)),2),round(float(np.std(r)),2),round(float(np.mean(d)),2),int(pi),int(tpi),N]
        out[n][mu]=per
print(json.dumps(out))
