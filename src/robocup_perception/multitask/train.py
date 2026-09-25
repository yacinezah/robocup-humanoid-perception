"""Detached/resumable development experiment. All writes stay in this namespace."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,pathlib,time,random,hashlib,math,traceback,copy
import numpy as np,torch
from torch.utils.data import DataLoader
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss
from .model import MultiTask,field_loss,detector_kd
from .data import HERE,Samples,collate
from .evaluate import evaluate_field,evaluate_detector
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,default=str)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def run(arm,seed,steps=8000):
    assert arm in ['S-F','F','J','J-KD','S-D']
    assert os.environ.get('PERCEPTION_TRAINING_AUTHORIZED')=='1','Set explicit training authorization and use an isolated resumable runner'
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    device='cuda';batch_size=12;out=HERE/'runs'/f'{arm}-S{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    rows=json.loads((HERE/'manifest.json').read_text());m=MultiTask(HERE/'references/detector.pt').to(device)
    m.detector.args=get_cfg(overrides=m.detector.args);m.configure(arm)
    criterion=v8DetectionLoss(m.detector)
    opt=torch.optim.AdamW([{'params':list(m.detector.parameters()),'lr':1e-4},{'params':list(m.field.parameters()),'lr':1e-3}],weight_decay=1e-4)
    best=-1e9;start=0 if arm!='S-D' else 500;elapsed=0.;history=[]
    source_root=pathlib.Path(__file__).resolve().parent
    code_hashes={n:sha(source_root/n) for n in ['model.py','data.py','evaluate.py','train.py']};protocol_hash=sha(HERE/'PROTOCOL.md')
    if (out/'last.pt').exists():
        cp=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert cp['protocol_hash']==protocol_hash and cp['code_hashes']==code_hashes,'Code/protocol changed: explicit recovery revision required'
        m.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);start=cp['step'];best=cp['best'];elapsed=cp['elapsed'];history=cp['history']
        torch.set_rng_state(cp['rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);np.random.set_state(cp['numpy_rng']);random.setstate(cp['random_rng'])
    teacher=None
    if arm=='J-KD':
        teacher=MultiTask(HERE/'references/detector.pt').detector.to(device).eval()
        for p in teacher.parameters():p.requires_grad_(False)
    def loader(task):
        offset=(start if task=='field' else max(start-500,0))*batch_size
        count=(steps-start if task=='field' else steps-max(start,500))*batch_size
        ds=Samples(rows[task+'_train'],seed,offset,count,task,arm=='J-KD' and task=='field')
        return iter(DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True,collate_fn=collate,persistent_workers=True))
    field_it=loader('field') if arm!='S-D' else None
    det_it=loader('detection') if arm in ['J','J-KD','S-D'] else None
    preview=sorted(rows['detection_validation'],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:128]
    tick=time.monotonic();last_loss={};eval_det=None;latest_field=None
    for step in range(start+1,steps+1):
        m.configure(arm)
        if step<=500:
            for p in m.detector.parameters():p.requires_grad_(False)
        m.train();opt.zero_grad(set_to_none=True)
        factor=.05+.95*(1+math.cos(math.pi*(step-1)/steps))/2
        for group,lr in zip(opt.param_groups,[1e-4,1e-3]):group['lr']=lr*factor
        if arm in ['J','J-KD','S-D'] and step>500:
            b={k:v.to(device,non_blocking=True) for k,v in next(det_it).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                d=m(b['img'],task='detection');loss,items=criterion(d,b);loss=loss.sum()/batch_size;kd=loss*0
                if teacher is not None:
                    with torch.no_grad():td=teacher(b['img'])
                    kd=detector_kd(d,td);loss=loss+.25*kd
            assert torch.isfinite(loss),'Nonfinite detector loss';loss.backward();last_loss.update(det=float(loss.detach()),det_kd=float(kd.detach()))
        if field_it is not None:
            b={k:v.to(device,non_blocking=True) for k,v in next(field_it).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f=m(b['img'],task='field');loss,parts=field_loss(f.float(),b['target'],b['valid'],b['teacher'] if arm=='J-KD' else None)
            assert torch.isfinite(loss),'Nonfinite field loss';loss.backward();last_loss.update(field=float(loss.detach()),**{k:float(v) for k,v in parts.items()})
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        if step%100==0:
            write(out/'status.json',{'state':'TRAINING','arm':arm,'seed':seed,'step':step,'max_steps':steps,'loss':last_loss,'gradient_norm':float(norm),'elapsed_seconds':elapsed+time.monotonic()-tick,'pid':os.getpid(),'utc_epoch':time.time()});print(arm,seed,step,last_loss,flush=True)
        if step%500==0 or step==steps:
            m.eval()
            if arm!='S-D':
                latest_field,fr=evaluate_field(m,rows['field_open_bridge'],device);write(out/f'field_{step:05d}.json',{'metrics':latest_field,'records':fr})
                score=latest_field['dice']+.05*latest_field['bf1']-2*latest_field['false_field']+.05*latest_field['border_recall']
            else:score=-1e9
            if step%1000==0 and arm in ['J','J-KD','S-D']:
                eval_det,dr=evaluate_detector(m,preview,device);write(out/f'det_preview_{step:05d}.json',{'metrics':eval_det,'records':dr})
                if arm=='S-D':score=eval_det['map']
            # Apply the most recent preview penalty at EVERY selection checkpoint,
            # not just the 1,000-update detector-evaluation checkpoints.
            if arm in ['J','J-KD'] and eval_det is not None:
                baseline=json.loads((HERE/'baseline_preview.json').read_text())['map']
                score-=max(0,baseline-eval_det['map']-.01)*2
            improvement=score>best
            if improvement:best=score
            elapsed+=time.monotonic()-tick;tick=time.monotonic()
            history.append({'step':step,'field':latest_field,'detector':eval_det,'score':score,'elapsed_seconds':elapsed})
            cp={'model':m.state_dict(),'optimizer':opt.state_dict(),'step':step,'arm':arm,'seed':seed,'best':best,'elapsed':elapsed,'history':history,'code_hashes':code_hashes,'protocol_hash':protocol_hash,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'random_rng':random.getstate()}
            torch.save(cp,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
            if improvement:
                # Deployable best artifact excludes optimizer states.
                torch.save({k:v for k,v in cp.items() if k not in ['optimizer','rng','cuda_rng','numpy_rng','random_rng','history']},out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
            write(out/'history.json',history)
            print('EVALUATION',arm,step,'field',latest_field,'det_map',eval_det['map'] if eval_det else None,flush=True)
    cp=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['model']);m.eval()
    d,dr=evaluate_detector(m,rows['detection_validation'],device);write(out/'detection_validation.json',{'metrics':d,'records':dr})
    write(out/'complete.json',{'state':'COMPLETE_DISCOVERY_FIT','arm':arm,'seed':seed,'steps':steps,'selected_step':cp['step'],'checkpoint_sha256':sha(out/'best.pt'),'detection_validation':d,'elapsed_seconds':elapsed,'evidence':'OPEN_DEVELOPMENT_ONLY'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--steps',type=int,default=8000);a=p.parse_args()
    try:run(a.arm,a.seed,a.steps)
    except Exception:
        write(HERE/'RUN_ERROR.json',{'arm':a.arm,'seed':a.seed,'time':time.time(),'traceback':traceback.format_exc()});raise
