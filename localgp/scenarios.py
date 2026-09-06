"""Random rectangular regions, UAV starts and unknown anisotropic user clusters."""
import numpy as np


def inside(points,bounds):
    points=np.asarray(points)
    return np.all((points>=bounds[0])&(points<=bounds[1]),axis=-1)


def generate(rng,config):
    area=config.area_size
    if config.map_family=='standard':size=rng.uniform(.65,1.,2)*area
    elif config.map_family=='elongated':
        size=np.array([rng.uniform(.32,.45),rng.uniform(.90,1.)])*area
        if rng.random()<.5:size=size[::-1]
    else:raise ValueError('Unknown map family')
    lower=rng.uniform(0,area-size)
    bounds=np.array([lower,lower+size])
    positions=[]
    for _ in range(config.agents):
        for attempt in range(10000):
            p=rng.uniform(bounds[0],bounds[1])
            if all(np.linalg.norm(p-q)>=config.min_separation for q in positions):
                positions.append(p);break
        else:raise ValueError('Cannot fit separated UAV starts inside map')
    clusters=int(rng.integers(2,min(6,config.users)+1))
    user_count=int(rng.integers(max(clusters,config.users//2),max(clusters,config.users*3//2)+1))
    centers=rng.uniform(bounds[0],bounds[1],(clusters,2))
    weights=rng.dirichlet(np.full(clusters,.7))
    labels=rng.choice(clusters,user_count,p=weights)
    scales=rng.uniform(.025,.12,(clusters,2))*min(size)
    angles=rng.uniform(0,2*np.pi,clusters)
    users=[]
    for k in labels:
        rot=np.array([[np.cos(angles[k]),-np.sin(angles[k])],[np.sin(angles[k]),np.cos(angles[k])]])
        for attempt in range(10000):
            p=centers[k]+rot@(rng.normal(size=2)*scales[k])
            if inside([p],bounds)[0]:users.append(p);break
        else:raise RuntimeError('Failed to sample clustered user inside map')
    metadata=dict(cluster_centers=centers.tolist(),cluster_scales=scales.tolist(),
                  cluster_angles=angles.tolist(),cluster_weights=weights.tolist(),
                  user_count=user_count,clusters=clusters,map_family=config.map_family)
    return bounds,np.array(positions),np.array(users),metadata
