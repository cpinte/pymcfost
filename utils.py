def undersample2d(im, npix):
    # Directly translation of my Yorick code
    pass

def gauss_kernel2(npix,sigma_x,sigma_y,PA):
    # Directly translation of my Yorick code

    '''
    gauss_kernel2(npix,sigma_x,sigma_y,PA)
    sigma_x et y en pix PA en degres
    PA defini depuis la direction verticale dans le sens horaire
    Cree un noyau gaussien dont l'integrale vaut 1
    Marche pour npix pair et impair
    SEE ALSO:
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
    dy2=psf.shape[0]

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
