import numpy as np
import matplotlib.pyplot as plt
from .utils import rotate_coords,rotate_to_obs_plane

def get_wake_cartesian(rp,phip,npts,rmin,rmax,HonR,q,sign):
    '''
       planet wake formula from Rafikov (2002)
    '''


    rplot = np.linspace(rmin,rmax,npts)
    rr = rplot/rp
    #          - here change rotation sign of wake
    phi = phip + sign *  np.sign(rmax-rmin)*(1./HonR) * ((rr**(q-0.5))/(q-0.5) \
                          - (rr**(q+1.))/(q+1.) \
                          - 3./((2.*q-1.)*(q+1.)))
    xx = rplot*np.cos(phi)
    yy = rplot*np.sin(phi)
    return xx,yy

def plot_wake(xy_obs,inc,PA,HonR,q,color="black"):
    '''
       plot planet wake
       and rotate to the observational viewpoint

    designed to work with dynamite inclination : if inc > 0, rotation is clockwise in plane of teh sky

    '''
    inc = np.deg2rad(inc)
    PA = np.deg2rad(PA)

    # we give planet location in the observational plane
    # bad attempt to provide the location in the rotated plane
    # by a simple linear scaling (does not get the angle correct)
    x_scale, y_scale, dum = rotate_coords(xy_obs[0],xy_obs[1],0.,inc,PA)
    x_p = xy_obs[0]*(xy_obs[0]/x_scale)
    y_p = xy_obs[1]*(xy_obs[1]/y_scale)

    # planet location in the unrotated plane
    rp = np.sqrt(x_p**2 + y_p**2)
    phip = np.arctan2(y_p,x_p)
    #print("rp = ",rp," phi planet = ",phip*180./np.pi)

    # radial range over which to plot the wake
    rmin = 1.e-3
    rmax = 3.*rp
    npts = 1000

    # outer wake
    xx,yy = get_wake_cartesian(rp,phip,npts,rp,rmax,HonR,q,np.sign(inc))
    xp,yp = rotate_to_obs_plane(xx,yy,inc,PA)
    plt.plot(xx,yy,color=color)

    # inner wake
    xx,yy = get_wake_cartesian(rp,phip,npts,rp,rmin,HonR,q,np.sign(inc))
    xp,yp = rotate_to_obs_plane(xx,yy,inc,PA)
    plt.plot(xx,yy,color=color)
