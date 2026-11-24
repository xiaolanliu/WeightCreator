"""Visible horizontal support surfaces from calibrated depth; no scene state."""
import numpy as np


def support_surface(sensor, center, half_window, lower_z, upper_z, normal_cosine, max_edge):
    world = np.asarray(sensor['world'])
    du = world[1:-1, 2:] - world[1:-1, :-2]
    dv = world[2:, 1:-1] - world[:-2, 1:-1]
    normal = np.cross(du, dv)
    norm = np.linalg.norm(normal, axis=-1)
    horizontal = abs(normal[..., 2]) / np.maximum(norm, 1e-12) >= normal_cosine
    continuous = (np.linalg.norm(du, axis=-1) < max_edge) & (np.linalg.norm(dv, axis=-1) < max_edge)
    points = world[1:-1, 1:-1]
    inside = np.all(abs(points[..., :2] - center) < half_window, axis=-1)
    inside &= (points[..., 2] >= lower_z) & (points[..., 2] <= upper_z)
    return points[inside & horizontal & continuous]


def support_normal(sensor, center, half_window, reference_z, band, minimum_cosine, max_edge):
    """Median oriented depth normals near the observed support height."""
    world=np.asarray(sensor['world']);du=world[1:-1,2:]-world[1:-1,:-2];dv=world[2:,1:-1]-world[:-2,1:-1]
    normals=np.cross(du,dv);normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-12)
    normals*=np.where(normals[...,2:]>0,1.,-1.)
    points=world[1:-1,1:-1]
    mask=np.all(abs(points[...,:2]-center)<half_window,axis=-1)&(abs(points[...,2]-reference_z)<band)
    mask&=(normals[...,2]>minimum_cosine)&(np.linalg.norm(du,axis=-1)<max_edge)&(np.linalg.norm(dv,axis=-1)<max_edge)
    if np.count_nonzero(mask)<20:return None,int(np.count_nonzero(mask))
    normal=np.median(normals[mask],axis=0);normal/=np.linalg.norm(normal)
    return normal,int(np.count_nonzero(mask))
