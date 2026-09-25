"""GT-free measured-line sampling and bounded map-consistency potential."""
import numpy as np

def support_pixels(probability):
    prob=np.asarray(probability)
    if prob.shape!=(320,320) or not np.isfinite(prob).all():raise ValueError('Invalid probability map')
    candidates=[]
    for y in range(0,320,20):
        for x in range(0,320,20):
            tile=prob[y:y+20,x:x+20];i=int(np.argmax(tile));v=float(tile.flat[i])
            if v>=.45:candidates.append([x+i%20,y+i//20,v])
    if not candidates:return np.empty((0,3))
    values=np.asarray(candidates);selected=[int(np.argmax(values[:,2]))]
    distance=np.full(len(values),np.inf)
    while len(selected)<min(32,len(values)):
        delta=values[:,:2]-values[selected[-1],:2]
        distance=np.minimum(distance,np.sum(delta*delta,axis=1));distance[selected]=-1
        selected.append(int(np.argmax(distance)))
    return values[selected]

def project_support(pixels,inverse_homography,rotation,translation,width=640,height=480):
    pixels=np.asarray(pixels,dtype=float).reshape(-1,3)
    uv=(pixels[:,:2]+.5)*[width/320,height/320]-.5
    xyh=np.column_stack([uv,np.ones(len(uv))])@np.asarray(inverse_homography).T
    with np.errstate(divide='ignore',invalid='ignore'):xy=xyh[:,:2]/xyh[:,2:]
    camera=np.column_stack([xy,np.zeros(len(xy))])@np.asarray(rotation).T+translation
    valid=np.isfinite(xy).all(axis=1)&(camera[:,0]>0)&(np.linalg.norm(xy,axis=1)<=12)
    return {'robot_xy_m':xy[valid].tolist(),'confidence':pixels[valid,2].tolist(),
            'image_xy':uv[valid].tolist(),'invalid_rays':int((~valid).sum())}

def line_potential(particles,features,segments,circle_radius,circle_center=(0.,0.)):
    p=np.asarray(particles,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('Invalid particle state')
    xy=np.asarray(features['robot_xy_m'],dtype=float).reshape(-1,2)
    confidence=np.asarray(features['confidence'],dtype=float)
    if len(xy)!=len(confidence):raise ValueError('Mismatched features')
    valid=np.isfinite(xy).all(axis=1)&np.isfinite(confidence)
    xy,confidence=xy[valid],confidence[valid]
    if len(xy)<4:return np.zeros(len(p))
    confidence=np.clip(confidence,.05,.95)
    c,s=np.cos(p[:,2,None]),np.sin(p[:,2,None])
    world=np.stack([c*xy[:,0]-s*xy[:,1]+p[:,0,None],s*xy[:,0]+c*xy[:,1]+p[:,1,None]],axis=-1)
    distance=np.abs(np.linalg.norm(world-np.asarray(circle_center),axis=-1)-circle_radius)
    for a,b in np.asarray(segments):
        v=b-a;den=float(v@v)
        if den<=0:raise ValueError('Degenerate map segment')
        t=np.clip(((world-a)*v).sum(axis=-1)/den,0,1)
        d=np.linalg.norm(world-(a+t[...,None]*v),axis=-1)
        distance=np.minimum(distance,d)
    sigma=np.sqrt(.1**2+(.03*np.linalg.norm(xy,axis=1))**2)
    score=np.log((1-confidence)+confidence*np.exp(-.5*(distance/sigma)**2))
    return 3*score.mean(axis=1)
