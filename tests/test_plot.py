import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import pymcfost as mcfost
from pymcfost.disc_structure import Disc
import pytest

import os
working_dir = os.getcwd()

data_dir = working_dir + "/tests/test_ref4.0"



out_path = working_dir + '/tests/actual'
expected = working_dir + '/tests/expected_figures'
if os.path.exists(out_path) == False: os.mkdir(out_path)

def data_th_plot_T():
    # Plot figure test_data_th_plot_T
    model = mcfost.SED(data_dir + "/data_th/")
    model.disc = Disc(model.basedir)
    fig, ax = plt.subplots()
    model.plot_T()
    fig.savefig('%s/test_data_th_plot_T.png' % out_path)


def data_th_plot_T_log():
    # Plot figure test_data_th_plot_T_log
    model = mcfost.SED(data_dir + "/data_th/")
    model.disc = Disc(model.basedir)
    fig, ax = plt.subplots()
    model.plot_T(log=True)
    fig.savefig('%s/test_data_th_plot_T_log.png' % out_path)

def data_th_plot_0():
    # Plot figure test_data_th_plot_0
    model = mcfost.SED(data_dir + "/data_th/")
    model.disc = Disc(model.basedir)
    fig, ax = plt.subplots()
    model.plot(0)
    fig.savefig('%s/test_data_th_plot_0.png' % out_path)

def data_th_plot_0_contrib():
    # Plot figure test_data_th_plot_0_contrib
    model = mcfost.SED(data_dir + "/data_th/")
    fig, ax = plt.subplots()
    model.plot(0, contrib=True)
    ax.set_ylim(1e-16,2e-12)   # we reduce the range on the y axis
    fig.savefig('%s/test_data_th_plot_0_contrib.png' % out_path)

def data_1_0_scattered_light():
    # Plot figure test_data_1_0_scattered_light
    image_1mum = mcfost.Image(data_dir + "/data_1.0/")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
    cbar = False
    no_ylabel = False
    for i in range(3):
        if i==2:
            cbar=True
        if i>0:
            no_ylabel=True
        image_1mum.plot(i, ax=axes[i], vmax=1e-15, colorbar=cbar, no_ylabel=no_ylabel)
    fig.savefig('%s/test_data_1_0_scattered_light.png' % out_path)

def data_1_0_polarisation():
    # Plot figure test_data_1_0_polarisation
    image_1mum = mcfost.Image(data_dir + "/data_1.0/")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
    cbar = False
    no_ylabel = False
    for i in range(3):
        if i>0:
            no_ylabel=True
        image_1mum.plot(i, ax=axes[i], type="Qphi", vmax=1e-15, colorbar=cbar,
                        no_ylabel=no_ylabel, pola_vector=True, nbin=15)
    fig.savefig('%s/test_data_1_0_polarisation.png' % out_path)

def data_1300_scattered_light():
    # Plot figure test_data_1300_scattered_light
    image_1mm  = mcfost.Image(data_dir + "/data_1300/")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
    cbar = False
    no_ylabel = False
    for i in range(3):
        if i==2:
            cbar=True
        if i>0:
            no_ylabel=True
        image_1mm.plot(i, ax=axes[i], Tb=True, colorbar=cbar, no_ylabel=no_ylabel, vmax=30)
    fig.savefig('%s/test_data_1300_scattered_light.png' % out_path)

def data_CO_plot_line():
    # Plot figure test_data_CO_plot_line
    mol = mcfost.Line(data_dir + "/data_CO/")
    fig, ax = plt.subplots()
    mol.plot_line(2)
    fig.savefig('%s/test_data_CO_plot_line.png' % out_path)

def data_CO_plot_map1():
    # Plot figure test_data_CO_plot_map1
    mol = mcfost.Line(data_dir + "/data_CO/")
    fig, ax = plt.subplots()
    mol.plot_map(2,v=0.5, Tb=True)
    fig.savefig('%s/test_data_CO_plot_map1.png' % out_path)

def data_CO_plot_map2():
    # Plot figure test_data_CO_plot_map2
    mol = mcfost.Line(data_dir + "/data_CO/")
    fig, ax = plt.subplots()
    mol.plot_map(2,v=0.5, bmaj=0.1, bmin=0.1, bpa=0, Tb=True)
    fig.savefig('%s/test_data_CO_plot_map2.png' % out_path)




plot_functions = [data_th_plot_T,
                  data_th_plot_T_log,
                  data_th_plot_0,
                  data_th_plot_0_contrib,
                  data_1_0_scattered_light,
                  data_1_0_polarisation,
                  data_1300_scattered_light,
                  data_CO_plot_line,
                  data_CO_plot_map1,
                  data_CO_plot_map2]
@pytest.mark.parametrize("plot_func", plot_functions)
def test_compare_data_th_plot_T(plot_func):
    plot_func()
    print('./%s/test_%s.png' % (out_path, plot_func.__name__))
    result = compare_images('%s/test_%s.png' % (out_path, plot_func.__name__),
                            '%s/test_%s.png' % (expected, plot_func.__name__),
                            tol=1e-6)
    assert result is None, f"Images not equal: {result}"



# @pytest.mark.mpl_image_compare
# def test_data_th_plot_T():
#     model = mcfost.SED(data_dir + "/data_th/", _mcfost_bin='None', _mcfost_utils='None')
#     model.disc = Disc(model.basedir)
#     fig, ax = plt.subplots()
#     model.plot_T()
#     return fig
#
# @pytest.mark.mpl_image_compare
# def test_data_th_plot_T_log():
#     model = mcfost.SED(data_dir + "/data_th/", _mcfost_bin='None', _mcfost_utils='None')
#     model.disc = Disc(model.basedir)
#     fig, ax = plt.subplots()
#     model.plot_T(log=True)
#     return fig
#
# @pytest.mark.mpl_image_compare
# def test_data_th_plot_0():
#     model = mcfost.SED(data_dir + "/data_th/", _mcfost_bin='None', _mcfost_utils='None')
#     model.disc = Disc(model.basedir)
#     fig, ax = plt.subplots()
#     model.plot(0)
#     return fig
#
# @pytest.mark.mpl_image_compare
# def test_data_th_plot_0_contrib():
#     model = mcfost.SED(data_dir + "/data_th/", _mcfost_bin='None', _mcfost_utils='None')
#     fig, ax = plt.subplots()
#     model.plot(0, contrib=True)
#     ax.set_ylim(1e-16,2e-12)   # we reduce the range on the y axis
#     return fig
#
# @pytest.mark.mpl_image_compare
# def test_data_1_0_scattered_light():
#     image_1mum = mcfost.Image(data_dir + "/data_1.0/")
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
#     cbar = False
#     no_ylabel = False
#     for i in range(3):
#         if i==2:
#             cbar=True
#         if i>0:
#             no_ylabel=True
#         image_1mum.plot(i, ax=axes[i], vmax=1e-15, colorbar=cbar, no_ylabel=no_ylabel)
#     return fig
#
# @pytest.mark.mpl_image_compare
# def test_data_1_0_polarisation():
#     image_1mum = mcfost.Image(data_dir + "/data_1.0/")
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
#     cbar = False
#     no_ylabel = False
#     for i in range(3):
#         if i>0:
#             no_ylabel=True
#         image_1mum.plot(i, ax=axes[i], type="Qphi", vmax=1e-15, colorbar=cbar,
#                         no_ylabel=no_ylabel, pola_vector=True, nbin=15)
#     return fig
#
#
# @pytest.mark.mpl_image_compare
# def test_data_1300_scattered_light():
#     image_1mm  = mcfost.Image(data_dir + "/data_1300/")
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
#     cbar = False
#     no_ylabel = False
#     for i in range(3):
#         if i==2:
#             cbar=True
#         if i>0:
#             no_ylabel=True
#         image_1mm.plot(i, ax=axes[i], Tb=True, colorbar=cbar, no_ylabel=no_ylabel, vmax=30)
#     return fig
#
#
#
# @pytest.mark.mpl_image_compare
# def test_data_CO_plot_line():
#     mol = mcfost.Line(data_dir + "/data_CO/")
#     fig, ax = plt.subplots()
#     mol.plot_line(2)
#     return fig
#
#
# @pytest.mark.mpl_image_compare
# def test_data_CO_plot_map1():
#     mol = mcfost.Line(data_dir + "/data_CO/")
#     fig, ax = plt.subplots()
#     mol.plot_map(2,v=0.5, Tb=True)
#     return fig
#
# @pytest.mark.mpl_image_compare
# def test_data_CO_plot_map2():
#     mol = mcfost.Line(data_dir + "/data_CO/")
#     fig, ax = plt.subplots()
#     mol.plot_map(2,v=0.5, bmaj=0.1, bmin=0.1, bpa=0, Tb=True)
#     return fig
