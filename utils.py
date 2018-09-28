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


def gauss_kernel2(npix,sigma_x,sigma_y,PA):
    # Directly translation of my Yorick code
    '''
    gauss_kernel2(npix,sigma_x,sigma_y,PA)
    sigma_x et y en pix PA en degres
    PA defini depuis la direction verticale dans le sens horaire
    Cree un noyau gaussien dont l'integrale vaut 1
    Marche pour npix pair et impair
    '''

    if (sigma_x < 1.0e-30): sigma_x = 1.0e-30
    if (sigma_y < 1.0e-30): sigma_y = 1.0e-30

    PA = PA * np.pi / 180.

    centre = npix/2. - 0.5

    px = np.linspace(0,npix-1,npix)
    mx, my = np.meshgrid(px,px)
    mx = mx - centre
    my = my - centre

    x = mx * np.cos(PA) - my * np.sin(PA)
    y = mx * np.sin(PA) + my * np.cos(PA)

    x = x / sigma_x
    y = y / sigma_y

    dist2 = x**2 + y**2

    tmp = np.exp(-0.5*dist2)

    return tmp/sum(sum(tmp))


def convol2df(im,psf):
    # Directly translation of my Yorick code

    psf2 = im * 0.

    dx=im.shape[0]
    dy=im.shape[1]

    dx2=psf.shape[0]
    dy2=psf.shape[1]

    startx = int((dx-dx2)/2)
    starty = int((dy-dy2)/2)

    psf2[startx:startx+dx2,starty:starty+dy2] = psf

    fim = np.fft.fft2(im)
    fpsf = np.fft.fft2(psf2)

    fim = fim * fpsf

    im = np.fft.ifft2(fim)
    im = np.roll(im,(int(dx/2),int(dy/2)),(0,1))
    im = abs(im) / (dx*dy)

    return im
