import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec as gs

plt.close('all')

plt.rcParams.update({'font.size': 6})
plt.rcParams['text.usetex'] = True
c = 3.0E8

def calc_Lxx(omega, n1, n2, theta_inc):
    theta_refr = np.arcsin((n1/n2)*np.sin(theta_inc))
    A = 2*n1*np.cos(theta_refr)
    B = n1*np.cos(theta_refr) + n2*np.cos(theta_inc)
    Lxx = A / B
    return Lxx

def calc_Lyy(omega, n1, n2, theta_inc):
    theta_refr = np.arcsin((n1/n2)*np.sin(theta_inc))
    A = 2*n1*np.cos(theta_inc)
    B = n1*np.cos(theta_inc) + n2*np.cos(theta_refr)
    Lyy = A / B
    return Lyy

def calc_Lzz(omega, n1, n2, n_int, theta_inc):
    theta_refr = np.arcsin((n1/n2)*np.sin(theta_inc))
    A = 2*n2*np.cos(theta_inc)
    B = n1*np.cos(theta_refr) + n2*np.cos(theta_inc)
    C = (n1 / n_int)**2
    Lzz = (A / B)*C
    return Lzz

def wavevector(wavelength):
    k = 1/wavelength
    return k 

def frequency(wavelength):
    omega = c/wavelength
    return omega

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


#JDP refractive indices for air and water from refractiveindex.info
#JDP n1 is air and n2 is water
n1 = 1.0
n2_800 = 1.3290
n2_6000 = 1.2650
n2_705 = 1.3308

#JDP interfacial refractive index from Voros 2004 Biophys. J. 87 p553-561
#JDP for proteins at the AWI 
n_int = 1.47

#JDP wavelengths and conversions to wavenumber/frequencies
lam_vis = 800E-9
lam_IR = 6000E-9
lam_SF = 705E-9

omega_vis = frequency(lam_vis)
omega_IR = frequency(lam_IR)
omega_SF = frequency(lam_SF)

k_vis = wavevector(lam_vis)
k_IR = wavevector(lam_IR)
k_SF = k_vis + k_IR

#JDP defining angles
theta_deg = np.linspace(0, 90, num=300, endpoint='True')
theta_rad = theta_deg * np.pi/180
theta_inc_IR, theta_inc_vis = np.meshgrid(theta_rad, theta_rad)
theta_SF_sin = (k_vis * np.sin(theta_inc_vis) + k_IR * np.sin(theta_inc_IR))/k_SF
theta_inc_SF = np.arcsin(theta_SF_sin)

#JDP defining local field factors for various things
Lyy_SF = calc_Lyy(omega_SF, n1, n2_705, theta_inc_SF)
Lyy_VIS = calc_Lyy(omega_vis, n1, n2_800, theta_inc_vis)
Lyy_IR = calc_Lyy(omega_IR, n1, n2_6000, theta_inc_IR )
Lxx_SF = calc_Lxx(omega_SF, n1, n2_705, theta_inc_SF)
Lxx_VIS = calc_Lxx(omega_vis, n1, n2_800, theta_inc_vis)
Lxx_IR = calc_Lxx(omega_IR, n1, n2_6000, theta_inc_IR )
Lzz_SF = calc_Lzz(omega_SF, n1, n2_705, n_int ,theta_inc_SF )
Lzz_VIS = calc_Lzz(omega_vis, n1, n2_800, n_int, theta_inc_vis )
Lzz_IR = calc_Lzz(omega_IR, n1, n2_6000, n_int, theta_inc_IR )

#JDP SSP fresnel factor
F_SSP = Lyy_SF * Lyy_VIS * Lzz_IR * np.sin(theta_inc_IR)

#JDP SPS fresnel factor
F_SPS = Lyy_SF * Lzz_VIS * Lyy_IR * np.sin(theta_inc_vis)

#JDP PSS fresnel factor
F_PSS = Lzz_SF * Lyy_VIS * Lyy_IR * np.sin(theta_inc_SF)

#JDP PPP fresnel factors (4 components)
F_PPP_XXZ = -Lxx_SF * Lxx_VIS * Lzz_IR * np.cos(theta_inc_SF) * np.cos(theta_inc_vis) * np.sin(theta_inc_IR)
F_PPP_XZX = -Lxx_SF * Lzz_VIS * Lxx_IR * np.cos(theta_inc_SF) * np.sin(theta_inc_vis) * np.cos(theta_inc_IR)
F_PPP_ZXX = Lzz_SF * Lxx_VIS * Lxx_IR * np.sin(theta_inc_SF) * np.cos(theta_inc_vis) * np.cos(theta_inc_IR)
F_PPP_ZZZ = Lzz_SF * Lzz_VIS * Lzz_IR * np.sin(theta_inc_SF) * np.sin(theta_inc_vis) * np.sin(theta_inc_IR)

#JDP making the big figure - should be in a loop but i was lazy and copy and pasting was faster
grid = gs.GridSpec(4, 2, hspace=-0.7, wspace=0.7)

ticks = np.array([0, 50, 100, 150, 200, 250, 300 ])
labels = (ticks/(300/90)).astype(int)

fig = plt.figure()
#fig.set_size_inches(246*(1/72)*0.5, 1.5*246*(1/72))
fig.set_size_inches(246*(1/72), 4*246*(1/72))

factor = 300/90

lower = 40*factor
upper = 60*factor

ax0 = fig.add_subplot(grid[0,0])
SSP = ax0.imshow(F_SSP, origin='lower', cmap='viridis', extent=(0, 300, 0, 300), interpolation='None')
ax0.plot([lower,lower],[0,300],color='w',ls=':',lw=0.5)
ax0.plot([upper,upper],[0,300],color='w',ls=':',lw=0.5)
ax0.plot([0,300],[lower,lower],color='w',ls=':',lw=0.5)
ax0.plot([0,300],[upper,upper],color='w',ls=':',lw=0.5)
ax0.set_xticks(ticks=ticks)
ax0.set_yticks(ticks=ticks)
ax0.set_xticklabels(labels=labels)
ax0.set_yticklabels(labels=labels)
ax0.set_xlabel('Incident Angle IR [deg]')
ax0.set_ylabel('Incident Angle VIS [deg]')
ax0.set_title(r'Fresnel Factor for $\chi^{(2)}_{YYZ}$ (SSP)')
colorbar(SSP)


ax1 = fig.add_subplot(grid[1,0])
SPS = ax1.imshow(F_SPS, origin='lower', cmap='viridis', extent=(0, 300, 0, 300), interpolation='None')
ax1.plot([lower,lower],[0,300],color='w',ls=':',lw=0.5)
ax1.plot([upper,upper],[0,300],color='w',ls=':',lw=0.5)
ax1.plot([0,300],[lower,lower],color='w',ls=':',lw=0.5)
ax1.plot([0,300],[upper,upper],color='w',ls=':',lw=0.5)
ax1.set_xticks(ticks=ticks)
ax1.set_yticks(ticks=ticks)
ax1.set_xticklabels(labels=labels)
ax1.set_yticklabels(labels=labels)
ax1.set_xlabel('Incident Angle IR [deg]')
ax1.set_ylabel('Incident Angle VIS [deg]')
ax1.set_title(r'Fresnel Factor for $\chi^{(2)}_{YZY}$ (SPS)')
colorbar(SPS)

ax2 = fig.add_subplot(grid[2,0])
PSS = ax2.imshow(F_PSS, origin='lower', cmap='viridis', extent=(0, 300, 0, 300), interpolation='None')
ax2.plot([lower,lower],[0,300],color='w',ls=':',lw=0.5)
ax2.plot([upper,upper],[0,300],color='w',ls=':',lw=0.5)
ax2.plot([0,300],[lower,lower],color='w',ls=':',lw=0.5)
ax2.plot([0,300],[upper,upper],color='w',ls=':',lw=0.5)
ax2.set_xticks(ticks=ticks)
ax2.set_yticks(ticks=ticks)
ax2.set_xticklabels(labels=labels)
ax2.set_yticklabels(labels=labels)
ax2.set_xlabel('Incident Angle IR [deg]')
ax2.set_ylabel('Incident Angle VIS [deg]')
ax2.set_title(r'Fresnel Factor for $\chi^{(2)}_{ZYY}$ (PSS)')
colorbar(PSS)



ax3 = fig.add_subplot(grid[0,1])
PPP_XXZ = ax3.imshow(F_PPP_XXZ, origin='lower', cmap='viridis', extent=(0, 300, 0, 300), interpolation='None')
ax3.plot([lower,lower],[0,300],color='w',ls=':',lw=0.5)
ax3.plot([upper,upper],[0,300],color='w',ls=':',lw=0.5)
ax3.plot([0,300],[lower,lower],color='w',ls=':',lw=0.5)
ax3.plot([0,300],[upper,upper],color='w',ls=':',lw=0.5)
ax3.set_xticks(ticks=ticks)
ax3.set_yticks(ticks=ticks)
ax3.set_xticklabels(labels=labels)
ax3.set_yticklabels(labels=labels)
ax3.set_xlabel('Incident Angle IR [deg]')
ax3.set_ylabel('Incident Angle VIS [deg]')
ax3.set_title(r'Fresnel Factor for $\chi^{(2)}_{XXZ}$ (PPP)')
colorbar(PPP_XXZ)

ax4 = fig.add_subplot(grid[1,1])
PPP_XZX = ax4.imshow(F_PPP_XZX, origin='lower', cmap='viridis', extent=(0, 300, 0, 300), interpolation='None')
ax4.plot([lower,lower],[0,300],color='w',ls=':',lw=0.5)
ax4.plot([upper,upper],[0,300],color='w',ls=':',lw=0.5)
ax4.plot([0,300],[lower,lower],color='w',ls=':',lw=0.5)
ax4.plot([0,300],[upper,upper],color='w',ls=':',lw=0.5)
ax4.set_xticks(ticks=ticks)
ax4.set_yticks(ticks=ticks)
ax4.set_xticklabels(labels=labels)
ax4.set_yticklabels(labels=labels)
ax4.set_xlabel('Incident Angle IR [deg]')
ax4.set_ylabel('Incident Angle VIS [deg]')
ax4.set_title(r'Fresnel Factor for $\chi^{(2)}_{XZX}$ (PPP)')
colorbar(PPP_XZX)

ax5 = fig.add_subplot(grid[2,1])
PPP_ZXX = ax5.imshow(F_PPP_ZXX, origin='lower', cmap='viridis', extent=(0, 300, 0, 300), interpolation='None')
ax5.plot([lower,lower],[0,300],color='w',ls=':',lw=0.5)
ax5.plot([upper,upper],[0,300],color='w',ls=':',lw=0.5)
ax5.plot([0,300],[lower,lower],color='w',ls=':',lw=0.5)
ax5.plot([0,300],[upper,upper],color='w',ls=':',lw=0.5)
ax5.set_xticks(ticks=ticks)
ax5.set_yticks(ticks=ticks)
ax5.set_xticklabels(labels=labels)
ax5.set_yticklabels(labels=labels)
ax5.set_xlabel('Incident Angle IR [deg]')
ax5.set_ylabel('Incident Angle VIS [deg]')
ax5.set_title(r'Fresnel Factor for $\chi^{(2)}_{ZXX}$ (PPP)')
colorbar(PPP_ZXX)

ax6 = fig.add_subplot(grid[3,1])
PPP_ZZZ = ax6.imshow(F_PPP_ZZZ, origin='lower', cmap='viridis', extent=(0, 300, 0, 300), interpolation='None')
ax6.plot([lower,lower],[0,300],color='w',ls=':',lw=0.5)
ax6.plot([upper,upper],[0,300],color='w',ls=':',lw=0.5)
ax6.plot([0,300],[lower,lower],color='w',ls=':',lw=0.5)
ax6.plot([0,300],[upper,upper],color='w',ls=':',lw=0.5)
ax6.set_xticks(ticks=ticks)
ax6.set_yticks(ticks=ticks)
ax6.set_xticklabels(labels=labels)
ax6.set_yticklabels(labels=labels)
ax6.set_xlabel('Incident Angle IR [deg]')
ax6.set_ylabel('Incident Angle VIS [deg]')
ax6.set_title(r'Fresnel Factor for $\chi^{(2)}_{ZZZ}$ (PPP)')
colorbar(PPP_ZZZ)

plt.tight_layout()
plt.savefig('fresnelfactors.pdf', dpi=300, transparent=True, bbox_inches='tight')

