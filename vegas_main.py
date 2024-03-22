# import the required libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------- Define the constants -------------------
# ------------------- Define the constants -------------------
# ------------------- Define the constants -------------------

cf = 3.894e+8 #pb per GeV^-2
Mz = 91.188 # GeV --> Z boson mass
Gz = 2.4414 # GeV  --> Gamma_Z=Z boson width
alpha= 1/(132.507)
Gf = 1.16639e-5 # GeV^-2
Wtheta = 0.222246 # Weinberg angle
kappa = np.sqrt(2)*Gf*(Mz**2)/(4*np.pi*alpha)

Qe = -1 # electron charge and muon charge
Vmu= -0.5 + 2*(Wtheta)
Ve = -0.5 + 2*(Wtheta)
Amu= -0.5
Ae = -0.5

XMIN = 10 # minimum energy in GeV
XMAX = 200 # maximum energy in GeV
YMIN = -1 # minimum costheta
YMAX = 1 # maximum costheta

# ------------------- Define the functions for the cross section -------------------
# ------------------- Define the functions for the cross section -------------------
# ------------------- Define the functions for the cross section -------------------

def chi1(s):
    num = kappa*s*(s-Mz**2)
    den = (s-Mz**2)**2 + (Gz**2)*(Mz**2)
    return num/den
def chi2(s):
    num = (kappa**2)*(s**2)
    den = (s-Mz**2)**2 + (Gz**2)*(Mz**2)
    return num/den

def A0(s):
    return Qe**2 - 2*Qe*Vmu*Ve*chi1(s) + (Amu**2 + Vmu**2)*(Ae**2 + Ve**2)*chi2(s)

def A1(s):
    return - 4*Qe*Amu*Ae*chi1(s) + 8*Amu*Vmu*Ae*Ve*chi2(s)

# option to calculate the cross section using the standard model, where the Z boson is present, or with QED where there is no Z boson
def cross_section(E,cost, method='SM'):
    # Define s as the square of the energy
    s = np.array(E**2)
    cost = np.array(cost)
    const = (alpha**2)/(4*s)
    if method=='SM':
        return const*(A0(s)*(1+cost**2) + A1(s)*cost)*cf # multiply by the conversion factor to convert to pb
    elif method=='QED':
        return const*(1+cost**2)*cf # multiply by the conversion factor to convert to pb
    

# ------------------- Define the functions for the VEGAS algorithm -------------------
# ------------------- Define the functions for the VEGAS algorithm -------------------
# ------------------- Define the functions for the VEGAS algorithm -------------------

# Define the function for the acceptance-rejection method
def brute_force(nPoints, seed=None, method='SM'):
    err=0
    if method=='SM':
        F_MAX=F_VAL_MAX
    elif method=='QED':
        F_MAX=FMAX2
    errs=0
    nFunctionEval = 0
    yy1_rej_method = []
    yy2_rej_method = []
    maxWeightEncounteredRej = -1.0e20
    generator = np.random.RandomState(seed=seed)
    while len(yy1_rej_method) < nPoints:
        rr = generator.uniform(size=3)
        yy1, yy2 = XMIN + rr[0] * (XMAX - XMIN), YMIN + rr[1] * (YMAX - YMIN)
        nFunctionEval += 1
        f_val = cross_section(yy1, yy2, method=method)
        if f_val > maxWeightEncounteredRej:
            maxWeightEncounteredRej = f_val
        if f_val > F_MAX:
            errs+=1
            print(
                f" f_val={f_val} exceeds F_VAL_MAX={F_MAX}, program will now exit. Error number {errs}"
            )
            exit(99)
        if (f_val / F_MAX) > rr[2]:
            yy1_rej_method.append(yy1)
            yy2_rej_method.append(yy2)
    return {"yy1": yy1_rej_method,
        "yy2": yy2_rej_method,
        "nFunEval": nFunctionEval,
        "maxWeightEncountered": maxWeightEncounteredRej}

# Define the function that sets up the grid for the VEGAS algorithm
def setup_intervals(NN=100, KK=2000, nIterations=4000, alpha_damp=1.5, seed=None, method='SM'):
    """
    Input:
        NN: Number of intervals in [XMIN, XMAX] or [YMIN, YMAX]
        KK: function evaluations per iteration
        nIterations: number of iterations
        alpha_damp: damping parameter in the Vegas algorithm
    Return:
        Intervals specified by xLow, yLow: each is a 1D numpy array of size NN+1, with
        xLow[0] = 0, xLow[NN] = ym; yLow[0] = 0, yLow[NN] = ym
    """

    # intitial intervals: uniform intervals between XMIN/YMIN and XMAX/YMAX
    xLow = XMIN + (XMAX - XMIN) / NN * np.arange(NN + 1)
    delx = np.ones(NN) * (XMAX - XMIN) / NN
    px = np.ones(NN) / (XMAX - XMIN)  # probability density in each interval
    yLow = YMIN + (YMAX - YMIN) / NN * np.arange(NN + 1)  # YMIN + (YMAX) / NN * np.arange(NN + 1) 
    dely = np.ones(NN) * (YMAX - YMIN) / NN
    py = np.ones(NN) / (YMAX - YMIN)

    generator = np.random.RandomState(seed=seed)
    for _ in range(nIterations):
        ixLow = generator.randint(0, NN, size=KK)
        xx = xLow[ixLow] + delx[ixLow] * generator.uniform(size=KK)
        iyLow = generator.randint(0, NN, size=KK)
        yy = yLow[iyLow] + dely[iyLow] * generator.uniform(size=KK)
        ff = cross_section(xx, yy,method=method)
        f2barx = np.array(
            [sum((ff[ixLow == i] / py[iyLow[ixLow == i]]) ** 2) for i in range(NN)]
        )
        fbarx = np.sqrt(f2barx)
        f2bary = np.array(
            [sum((ff[iyLow == i] / px[ixLow[iyLow == i]]) ** 2) for i in range(NN)]
        )
        fbary = np.sqrt(f2bary)
        fbardelxSum = np.sum(fbarx * delx)
        fbardelySum = np.sum(fbary * dely)
        logArgx = fbarx * delx / fbardelxSum
        logArgy = fbary * dely / fbardelySum
        mmx = KK * pow((logArgx - 1) / np.log(logArgx), alpha_damp)
        mmx = mmx.astype(int)
        mmx = np.where(mmx > 1, mmx, 1)
        mmy = KK * pow((logArgy - 1) / np.log(logArgy), alpha_damp)
        mmy = mmy.astype(int)
        mmy = np.where(mmy > 1, mmy, 1)
        xLowNew = [xLow[i] + np.arange(mmx[i]) * delx[i] / mmx[i] for i in range(NN)]
        xLowNew = np.concatenate(xLowNew, axis=0)
        yLowNew = [yLow[i] + np.arange(mmy[i]) * dely[i] / mmy[i] for i in range(NN)]
        yLowNew = np.concatenate(yLowNew, axis=0)
        nCombx = int(len(xLowNew) / NN)
        nComby = int(len(yLowNew) / NN)
        i = np.arange(NN)
        xLow[:-1] = xLowNew[i * nCombx]
        yLow[:-1] = yLowNew[i * nComby]
        delx = np.diff(xLow)
        dely = np.diff(yLow)
        px = 1.0 / delx / NN
        py = 1.0 / dely / NN

    return xLow, yLow, delx, dely

# Define the function for the VEGAS algorithm
def vegas(
    nPoints,
    vegasRatioFactor,
    NN=100,
    KK=2000,
    nIterations=4000,
    alpha_damp=1.5,
    seed=None,
    method='SM'
):
    errs=0
    if method=='SM':
        F_MAX=F_VAL_MAX
    elif method=='QED':
        F_MAX=FMAX2
    xLow, yLow, delx, dely = setup_intervals(NN, KK, nIterations, alpha_damp, seed,method=method)
    # vegasRatioMax = vegasRatioFactor * F_VAL_MAX * NN * NN * delx[NN - 2] * dely[NN - 2]  # we wanted to understand where the index NN-2 came from
    # vegasRatioMax = vegasRatioFactor * F_VAL_MAX * NN * NN * np.max(delx) * np.max(dely)  # in lab17 gaussian example, delx[NN-2] is the fourth largest delx
    vegasRatioMax = vegasRatioFactor * F_MAX * NN * NN * np.min(delx) * np.min(dely) # in the original code, delx[NN-2] is the smalles delx (where the maximum occurs)
    nFunctionEval = 0
    yy1_vegas_method = []
    yy2_vegas_method = []
    yy1_vrho_method = []
    yy2_vrho_method = []
    maxWeightEncountered = -1.0e20

    generator = np.random.RandomState(seed=seed)
    while len(yy1_vegas_method) < nPoints:
        ixLow = generator.randint(0, NN)
        xx = xLow[ixLow] + delx[ixLow] * generator.uniform()
        iyLow = generator.randint(0, NN)
        yy = yLow[iyLow] + dely[iyLow] * generator.uniform()
        yy1_vrho_method.append(xx)
        yy2_vrho_method.append(yy)
        nFunctionEval += 1
        f_val = cross_section(xx, yy, method=method)
        ratio = f_val * NN * NN * delx[ixLow] * dely[iyLow]
        if ratio > maxWeightEncountered:
            maxWeightEncountered = ratio
        if ratio > vegasRatioMax:
            errs+=1
            print(
                f"ratio={ratio} exceeds vegasRatioMax={vegasRatioMax}, yy={yy} program will now exit. Error number {errs}"
            )
            exit(99)
        if ratio / vegasRatioMax > generator.uniform():
            yy1_vegas_method.append(xx)
            yy2_vegas_method.append(yy)

    return {
        "yy1vrho": yy1_vrho_method,
        "yy2vrho": yy2_vrho_method,
        "yy1vegas": yy1_vegas_method,
        "yy2vegas": yy2_vegas_method,
        "nFunEval": nFunctionEval,
        "maxWeightEncountered": maxWeightEncountered,
        "vegasRatioMax": vegasRatioMax,
    }

# ------------------- Define the plotting function -------------------
# ------------------- Define the plotting function -------------------
# ------------------- Define the plotting function -------------------
def lego_plot(xAmplitudes, yAmplitudes, nBins, xLabel, yLabel, title, scale='linear'):
    x = np.array(xAmplitudes)  # turn x,y data into numpy arrays
    y = np.array(yAmplitudes)  # useful for regular matplotlib arrays

    fig = plt.figure(figsize=(9,9))  # create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection="3d")

    # make histograms - set bins
    hist, xedges, yedges = np.histogram2d(x, y, bins=(nBins, nBins))
    xpos, ypos = np.meshgrid(xedges[:-1] + xedges[1:], yedges[:-1] + yedges[1:])

    xpos = xpos.flatten() / 2.0
    ypos = ypos.flatten() / 2.0
    zpos = np.zeros_like(xpos)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    histt = np.transpose(hist) # need to transpose the array for it to take the organization we want when we flatten it
    dz = histt.flatten()
    dzlog = [np.log10(dzi) if dzi>=1 else 0 for dzi in dz]

    # cmap = mpl.colormaps.jet
    cmap = mpl.colormaps["jet"]
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]


    if scale=='linear':
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort="average")
    elif scale=='log':
        ax.bar3d(xpos, ypos, zpos, dx, dy, dzlog, color=rgba, zsort="average")
    plt.title(title)
    plt.xlabel(xLabel, fontsize=18)
    plt.ylabel(yLabel, fontsize=18)
    plt.xlim(XMIN, XMAX)
    plt.ylim(YMIN, YMAX)
    plt.show()

def _get_colors(hist):
    cmap = mpl.cm.jet
    max_height = np.max(hist)
    min_height = np.min(hist)
    rgba = [cmap((k - min_height) / max_height) for k in hist]
    return rgba

def lego_plot2d(xAmplitudes, yAmplitudes, nBins, xLabel, yLabel, title, scale='linear'):
    x = np.array(xAmplitudes)
    y = np.array(yAmplitudes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    xhist, xedges = np.histogram(x, bins=nBins)
    xpos = (xedges[:-1] + xedges[1:]) / 2
    if scale=='linear':
        axes[0].bar(xpos, xhist, width=np.diff(xedges), color=_get_colors(xhist), alpha=0.7)
    elif scale=='log':
        axes[0].bar(xpos, np.log10(xhist), width=np.diff(xedges), color=_get_colors(xhist), alpha=0.7)
    
    yhist, yedges = np.histogram(y, bins=nBins)
    ypos = (yedges[:-1] + yedges[1:]) / 2
    axes[1].bar(ypos, yhist, width=np.diff(yedges), color=_get_colors(yhist), alpha=0.7)

    axes[0].set_xlabel(xLabel, fontsize=18)
    # axes[0].set_ylabel('Frequency')
    # axes[0].set_title(title + ' - E_{cm}')

    axes[1].set_xlabel(yLabel, fontsize=18)
    # axes[1].set_ylabel('Frequency')
    # axes[1].set_title(title + ' - Y Amplitudes')

    plt.tight_layout()
    plt.show()

def plot_results(
    nPoints,
    vegasRatioFactor,
    bf,
    vg,
    nBins=50,
    scale='linear',
    histtype='3d'
):

    # brute force
    titleRej = r"Acceptance-rejection Method $f(x,y)$"
    if histtype=='3d':
        lego_plot(bf["yy1"], bf["yy2"], nBins, "$E_{cm}$", "$cos$"+r"$\theta$", titleRej,scale=scale)
    elif histtype=='2d':
        lego_plot2d(bf["yy1"], bf["yy2"], nBins, "$E_{cm}$", "$cos$"+r"$\theta$", titleRej,scale=scale)
    plt.show()

    # Vegas method
    titleVrho = r"Vegas Method $p(x,y)$"
    if histtype=='3d':
        lego_plot(vg["yy1vrho"], vg["yy2vrho"], nBins, "$E_{cm}$", "$cos$"+r"$\theta$", titleVrho,scale=scale)
    elif histtype=='2d':
        lego_plot2d(vg["yy1vrho"], vg["yy2vrho"], nBins, "$E_{cm}$", "$cos$"+r"$\theta$", titleVrho,scale=scale)
    plt.show()

    titleVegas = r"Vegas Method $f(x,y)$"
    if histtype=='3d':
        lego_plot(vg["yy1vegas"], vg["yy2vegas"], nBins, "$E_{cm}$", "$cos$"+r"$\theta$", titleVegas,scale=scale)
    elif histtype=='2d':
        lego_plot2d(vg["yy1vegas"], vg["yy2vegas"], nBins, "$E_{cm}$", "$cos$"+r"$\theta$", titleVegas,scale=scale)
    plt.show()

    print(
        f"Acceptance-rejection method nPoints={nPoints}, nFunctionEval={bf['nFunEval']}, maxWeightEncounteredRej={bf['maxWeightEncountered']}, F_VAL_MAX={F_VAL_MAX}"
    )
    print(
        f"Vegas method nPoints={nPoints}, nFunctionEval={vg['nFunEval']}, maxWeightEncountered={vg['maxWeightEncountered']}, vegasRatioMax={vg['vegasRatioMax']}, vegasRatioFactor={vegasRatioFactor}"
    )

def plot_bindist(xAmplitudes, yAmplitudes, nBins, xLabel, yLabel):
    x = np.array(xAmplitudes)
    y = np.array(yAmplitudes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    xhist, xedges = np.histogram(x, bins=nBins)
    xpos = (xedges[:-1] + xedges[1:]) / 2
    # normalized cummulative hist
    xhist = np.cumsum(xhist)
    xhist = xhist/xhist[-1]
    axes[0].plot(xpos, xhist, linewidth=4, color='b')
    # vertical line at x=Mz
    axes[0].axvline(x=Mz, color='k', linestyle='--', label='Z boson mass', linewidth=2)

    
    yhist, yedges = np.histogram(y, bins=nBins)
    ypos = (yedges[:-1] + yedges[1:]) / 2
    # normalized cummulative hist
    yhist = np.cumsum(yhist)
    yhist = yhist/yhist[-1]
    axes[1].plot(ypos, yhist, linewidth=4, color='r')

    axes[0].set_xlabel(xLabel, fontsize=18)
    axes[0].set_ylabel('Cumulative Bins Distribution')
    axes[0].legend(fontsize=12)
    axes[1].set_xlabel(yLabel, fontsize=18)
    axes[1].set_ylabel('Bins Distribution')

    plt.tight_layout()
    plt.show()