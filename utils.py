import numpy as np

def bin_image(im, n, func=np.sum):
    # bin an image in blocks of n x n pixels
    # return a image of size im.shape/n

    nx = im.shape[0]
    nx_new = nx//n
    x0 = (nx - nx_new * n)//2

    ny = im.shape[1]
    ny_new = ny//n
    y0 = (ny - ny_new * n)//2

    return np.reshape(np.array([func(im[x0+k1*n:(k1+1)*n,y0+k2*n:(k2+1)*n]) for k1 in range(nx_new) for k2 in range(ny_new)]),(nx_new,ny_new))
