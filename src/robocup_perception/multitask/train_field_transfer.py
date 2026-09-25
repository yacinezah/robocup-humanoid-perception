"""Paired, resumable field-transfer fits. Detector immutable in EF-KD."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time,hashlib,random,math,traceback
import numpy as np,torch
from torch.utils.data import DataLoader
from .data import HERE,Samples,collate
from .model import field_loss
from .efficient_model import EfficientMTL,raw_efficient
from .evaluate import evaluate_field
from .train import write,sha
def state_hash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def run(arm,seed):
    assert os.environ.get('PERCEPTION_TRAINING_AUTHORIZED')=='1';assert arm in ['EF-KD','ESF-KD']
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    source=HERE/'references/detector_y26.pt';assert sha(source)=='3a3dbe1f07aad96f03b42d64bd564eae39ef0a087ca4394bb855737a669df845'
    rows=json.loads((HERE/'manifest.json').read_text());out=HERE/'runs'/f'{arm}-S{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    m=EfficientMTL(source).cuda();initial_detector=state_hash(m.detector);mode='F' if arm=='EF-KD' else 'S-F';m.configure(mode)
    opt=torch.optim.AdamW([{'params':list(m.detector.parameters()),'lr':1e-4},{'params':list(m.field.parameters()),'lr':1e-3}],weight_decay=1e-4)
    source_hashes={n:sha(__import__('pathlib').Path(__file__).resolve().parent/n) for n in ['efficient_model.py','model.py','train_field_transfer.py','train.py','data.py','evaluate.py']};protocol_hash=sha(HERE/'PROTOCOL.md')
    start=0;best=-1e9;elapsed=0.;history=[]
    if (out/'last.pt').exists():
        c=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert c['source_hashes']==source_hashes and c['protocol_hash']==protocol_hash
        m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);start=c['step'];best=c['best'];elapsed=c['elapsed'];history=c['history'];torch.set_rng_state(c['rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
    ds=Samples(rows['field_train'],seed,start*12,(8000-start)*12,'field',True)
    loader=iter(DataLoader(ds,batch_size=12,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True,collate_fn=collate))
    tick=time.monotonic()
    for step in range(start+1,8001):
        m.configure(mode)
        if step<=500:
            for p in m.detector.parameters():p.requires_grad_(False)
        m.train();opt.zero_grad(set_to_none=True);factor=.05+.95*(1+math.cos(math.pi*(step-1)/8000))/2
        for g,lr in zip(opt.param_groups,[1e-4,1e-3]):g['lr']=lr*factor
        b={k:v.cuda(non_blocking=True) for k,v in next(loader).items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):f=m(b['img'],task='field');loss,parts=field_loss(f.float(),b['target'],b['valid'],b['teacher'])
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        if step%100==0:
            write(out/'status.json',{'state':'TRAINING','arm':arm,'seed':seed,'step':step,'loss':float(loss),'gradient_norm':float(norm),'pid':os.getpid(),'utc_epoch':time.time(),'elapsed_seconds':elapsed+time.monotonic()-tick});print(arm,seed,step,float(loss),flush=True)
        if step%500==0:
            m.eval();metrics,records=evaluate_field(m,rows['field_open_bridge'],'cuda');score=metrics['dice']+.05*metrics['bf1']-2*metrics['false_field']+.05*metrics['border_recall'];improved=score>best;best=max(best,score)
            if arm=='EF-KD':assert state_hash(m.detector)==initial_detector,'Frozen detector state changed'
            elapsed+=time.monotonic()-tick;tick=time.monotonic();history.append({'step':step,'field':metrics,'score':score,'elapsed_seconds':elapsed})
            c={'model':m.state_dict(),'optimizer':opt.state_dict(),'step':step,'arm':arm,'seed':seed,'best':best,'elapsed':elapsed,'history':history,'source_hashes':source_hashes,'protocol_hash':protocol_hash,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'initial_detector_state_sha256':initial_detector}
            torch.save(c,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
            if improved:
                torch.save({k:v for k,v in c.items() if k not in ['optimizer','history','rng','cuda_rng']},out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
            write(out/f'field_{step:05d}.json',{'metrics':metrics,'records':records});write(out/'history.json',history);print('EVALUATION',arm,step,metrics,flush=True)
    c=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(c['model']);final_detector=state_hash(m.detector)
    if arm=='EF-KD':assert final_detector==initial_detector
    write(out/'complete.json',{'state':'COMPLETE_DISCOVERY_FIT','arm':arm,'seed':seed,'selected_step':c['step'],'checkpoint_sha256':sha(out/'best.pt'),'detector_state_preserved':final_detector==initial_detector,'initial_detector_state_sha256':initial_detector,'final_detector_state_sha256':final_detector,'selected_field':next(r['field'] for r in history if r['step']==c['step']),'elapsed_seconds':elapsed,'evidence':'OPEN_DEVELOPMENT_ONLY','detector_role':'immutable matched native detector' if arm=='EF-KD' else 'unused continuation; not a detector condition'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=101);a=p.parse_args()
    try:run(a.arm,a.seed)
    except Exception:
        write(HERE/'TRANSFER_ERROR.json',{'arm':a.arm,'seed':a.seed,'traceback':traceback.format_exc(),'utc_epoch':time.time()});raise
