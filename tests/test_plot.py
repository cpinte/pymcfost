# import os
#
# working_dir = os.getcwd()

#
from pathlib import Path

test_dir = Path(__file__).parent

output_dir = test_dir / "artifacts"
input_dir = test_dir / "corpus"
ddc_dir = test_dir / "corpus" / "ccd"

print(output_dir)
print(input_dir)
print(ddc_dir)



# import matplotlib.pyplot as plt
# import pymcfost as mcfost
# from pymcfost.disc_structure import Disc
# import pytest
#
# import os
# working_dir = os.getcwd()
#
# data_dir = working_dir + "/test_ref4.0"
#
#
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
