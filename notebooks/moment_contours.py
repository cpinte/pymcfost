import pymcfost as pym
import matplotlib.pyplot as plt


data = pym.Line('data_CO')

nrows = 1
ncols = 4


plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True


fig, allAxes = plt.subplots(nrows=nrows, ncols=ncols, sharey='row', figsize=(4 * ncols, 5 * nrows), sharex='col', dpi=200)


limits = [-0.8, 0.8, -0.8, 0.8]
             
h1 = data.plot_map(moment=0, substract_cont=True, plot_stars=True, no_xlabel=False, iaz=0, colorbar=False, ax=allAxes[0], limits=limits, bmaj=0.05, bmin=0.05, bpa=0, no_ylabel=False, no_yticks=False, Tb=True, plot_beam=True, title="M0")

# M0 -> M0 contours -> M1 contours -> v=0

# Here we are replotting the data from the M0 map onto other axes. This means we can skip the recalculating of the moment each time

# replot(ax, map, no_xlabel=None, no_ylabel=None, title=None, cmap=None)
pym.replot(ax=allAxes[1], map=h1, no_ylabel=True, title="M0 with M0 contours")
pym.replot(allAxes[2], h1, no_ylabel=True, title="M0 with M1 contours")
pym.replot(allAxes[3], h1, no_ylabel=True, title="M0 with v=0 contour")


#plot_contours(map, moment, levels=4, ax=None, specific_values=[], colors='black', linewidths=0.25)  
pym.plot_contours(map=h1, moment=0, ax=allAxes[1], colors='white', linewidths=0.5) # M0 contours
pym.plot_contours(map=h1, moment=1, ax=allAxes[2], colors='white', linewidths=0.5) # M1 contours
pym.plot_contours(map=h1, moment=1, ax=allAxes[3], specific_values=[0], colors='white', linewidths=0.5) # M1 contours, but only the contour line for v-0


# Because all of the figures share the same fmin/fmax range, we can have a single colorbar for all of them, and place it on the axes container.

# create_colorbar(figure, map, ax, fontsize='18', rotation=270, tick_label_size=16)
pym.create_colorbar(fig, h1, allAxes, fontsize='16')


plt.savefig('moment_contours.png')
