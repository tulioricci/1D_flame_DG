""" Thu 29 Dec 2022 05:26:13 PM CST """

import numpy as np
import matplotlib.pyplot as plt

import grid.basis as basis
from filterfuncs import *

import os
import pickle
import time
import sys

np.set_printoptions(edgeitems=10,linewidth=132,suppress=True)

#import gc
#import tracemalloc
#tracemalloc.start()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sigma = 0.0002

#case = "1D_flame"
case = "cantera"

xRef = np.genfromtxt('grid/x_50um.dat')

integration = "euler"

# polynomial order
Np = 3

niter = 2000000

"""
#################################################################################
"""

import getopt

arg_list = sys.argv[1:]

rstfile = None
opts, args = getopt.getopt(arg_list,"hr:",["rstfile="])
for opt, arg in opts:
    if opt == '-h':
        print ('test.py -r <rstfile>')
        sys.exit()
    elif opt in ("-r", "--rstfile"):
        rstfile = arg
        print ('Restart file is ', rstfile)




N = Np + 1

CFL = 0.22/(2.0*N + 1)

phi = basis.LegendrePolynomials(N)
LGL = basis.GaussLobatto(N,phi)
Vnd, invVnd, Dr = basis.VandermondeMatrices(N,phi,LGL)

x, xF = basis.mapping(xRef,LGL)
dx = xF[-1,:] - xF[0,:]

Jac, invJac = basis.Jacobian(x, Dr)

Surf = Vnd@Vnd.T@invJac

discr = [x, invJac, Dr, Vnd, Surf]

zeros = x*0.0

Ncoll = x.shape[0]
Ngrid = x.shape[1]
print(Ncoll, Ngrid)

#============================

#from mechanisms.uiuc import Thermochemistry
from mechanisms.uiuc_mod import Thermochemistry
#from mechanisms.Davis2005_expanded import Thermochemistry

eos = Thermochemistry()
print(eos.species_names)

nspecies = eos.num_species

#data = np.genfromtxt('adiabatic_flame_Davis2005_expanded_phi1.00_p1.00_E:H0.0.csv',skip_header=1,delimiter=',')
data = np.genfromtxt('./mechanisms/adiabatic_flame_uiuc_mod_phi1.00_p1.00_E:H0.0.csv', skip_header=1, delimiter=',')

nEqs = 3 + nspecies

#################################################################################

def gamma(T=None, Y=None, eos=None):
    cp = eos.get_mixture_specific_heat_cp_mass(T, Y)
    rspec = eos.get_specific_gas_constant(Y)

    return cp / (cp - rspec)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def InitCond(x, eos):

    exist = False

    if (case == "cantera"):

        nspecies = 7
        from scipy.interpolate import CubicSpline

        data_ct = 1.0*data
        nGrid_ct = data_ct[:,0].shape[0]

        data_ct[:,0] = data_ct[:,0] - 0.007

        x_min = data_ct[ 0,0]
        x_max = data_ct[-1,0]

        x_aux = x.reshape((Ncoll*Ngrid),order='F')
         
        cs = CubicSpline(data_ct[:,0], data_ct[:,1], extrapolate=False)
        data_u = cs(x_aux)
        u = np.where(np.less(x_aux, x_min), data_ct[0,1], data_u)
        u = np.where(np.greater(x_aux, x_max), data_ct[-1,1], u)
        u = u.reshape((Ncoll,Ngrid),order='F')

        cs = CubicSpline(data_ct[:,0], data_ct[:,2], extrapolate=False)
        data_T = cs(x_aux)
        T = np.where(np.less(x_aux, x_min), data_ct[0,2], data_T)
        T = np.where(np.greater(x_aux, x_max), data_ct[-1,2], T)
        T = T.reshape((Ncoll,Ngrid),order='F')

        cs = CubicSpline(data_ct[:,0], data_ct[:,3], extrapolate=False)
        data_rho = cs(x_aux)
        rho = np.where(np.less(x_aux, x_min), data_ct[0,3], data_rho)
        rho = np.where(np.greater(x_aux, x_max), data_ct[-1,3], rho)
        rho = rho.reshape((Ncoll,Ngrid),order='F')

        X = np.zeros((nspecies),dtype=object)
        for ii in range(0, nspecies):
            cs = CubicSpline(data_ct[:,0], data_ct[:,4+ii], extrapolate=False)
            data_X = cs(x_aux)
            X[ii] = np.where(np.less(x_aux, x_min), data_ct[0,4+ii], data_X)
            X[ii] = np.where(np.greater(x_aux, x_max), data_ct[-1,4+ii], X[ii])
            X[ii] = X[ii].reshape((Ncoll,Ngrid),order='F')

        W_mean = 0.0
        for ii in range(0, nspecies):
            W_mean = W_mean + X[ii]*eos.wts[ii]

        Y = np.zeros((nspecies),dtype=object)
        for ii in range(0, nspecies):
            Y[ii] = X[ii]*eos.wts[ii]/W_mean

        p = 101325.0 + zeros
        rho = eos.get_density(p, T, Y)

        rhoE = rho*eos.get_mixture_internal_energy_mass(T, Y) + 0.5*rho*u**2

        cv = []
        cv.append(rho)
        cv.append(rho*u)
        cv.append(rhoE)
        for i in range(0, nspecies):
            cv.append(rho*Y[i])

        return np.asarray( cv )

    if (case == "1D_flame"):

        u_0 = data[ 0,1]
        u_N = data[-1,1]
        u = (u_N - u_0)*0.5*(1.0 + np.tanh(1.0/sigma*(x))) + u_0

        T_0 = data[ 0,2]
        T_N = data[-1,2]
        #T = (T_N - T_0)*0.5*(1.0 + np.tanh(1.0/sigma*(x-0.0001))) + T_0
        T = (T_N - T_0)*0.5*(1.0 + np.tanh(1.0/sigma*(x))) + T_0

        nspecies = int(data.shape[1] - 4)
        Y = []
        W_mean_0 = np.sum(data[ 0,4:]*eos.wts)
        W_mean_N = np.sum(data[-1,4:]*eos.wts)
        for i in range(0, nspecies):
            Y_0 = data[ 0,4+i]*eos.wts[i]/W_mean_0
            Y_N = data[-1,4+i]*eos.wts[i]/W_mean_N
            Y.append( (Y_N - Y_0)*0.5*(1.0 + np.tanh(1.0/sigma*(x))) + Y_0 )
        Y = np.asarray(Y)

        p = 101325.0 + zeros
        rho = eos.get_density(p, T, Y)

        rhoE = rho*eos.get_mixture_internal_energy_mass(T, Y) + 0.5*rho*u**2
        rhoU = rho*u
        rhoY = rho*Y

        cv = []
        cv.append(rho)
        cv.append(rhoU)
        cv.append(rhoE)
        for i in range(0, nspecies):
            cv.append(rhoY[i])

        return np.asarray( cv )

    if exist == False:
        sys.exit()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def boundary_condition(location, x, cv, p, Y, eos):

    if (case == "1D_flame" or case == "cantera"):

        if location == 'inlet':
            rhoref = 1.1698532681009575
            uref = 0.6542240968866195
            pref = 101325.0
            Tref = 300.0
            Yref = Y

        if location == 'outlet':
            rhoref = 0.143747172793783
            uref = 5.3242521589756695
            pref = 101325.0
            Tref = 2413.4068590583524
            Yref = Y

        Rref = eos.get_specific_gas_constant(Yref)

    #~~~~~
    rho = cv[0]
    rhoU = cv[1]

    c = np.sqrt( gamma(Tref, Yref, eos)*pref/rhoref )

    rtilde =      rho - rhoref
    utilde = rhoU/rho - uref
    ptilde =        p - pref

    if location == 'inlet':
        # non-reflecting inflow BC
        c1 = 0.0 #-c**2*rtilde + ptilde
        c3 = 0.0 #+rhoref*c*utilde + ptilde
        c4 = -rhoref*c*utilde + ptilde

    if location == 'outlet':
        # non-reflecting outflow BC
        c1 = -c**2*rtilde + ptilde
        c3 = +rhoref*c*utilde + ptilde
        c4 = 0.0 #-rhoref*c*utilde + ptilde

    rtilde = (-1./c**2)*c1 + (1./(2.*c**2    ))*c3 + (1./(2.*c**2    ))*c4
    utilde = (   0.   )*c1 + (1./(2.*rhoref*c))*c3 - (1./(2.*rhoref*c))*c4
    ptilde = (   0.   )*c1 + (1./(2.)         )*c3 + (1./(2.)         )*c4

    rho_bc = rtilde + rhoref
    u_bc   = utilde + uref
    p_bc   = ptilde + pref

    rhoU_bc = rho_bc*u_bc
    T_bc = p_bc/(Rref*rho_bc)
    rhoE_bc = rho_bc*eos.get_mixture_internal_energy_mass(T_bc, Yref) + 0.5*rho_bc*u_bc**2
    rhoY_bc = rho_bc*Yref

    return np.hstack(( rho_bc, rhoU_bc, rhoE_bc, rhoY_bc ))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def inviscid_flux(cv, dv):

    rho  = cv[0]
    rhoU = cv[1]
    rhoE = cv[2]
    rhoY = cv[3:]

    p = dv[0]

    flux = []
    flux.append(rhoU)
    flux.append(rhoU**2/rho + p)
    flux.append((rhoE + p)*rhoU/rho)
    for i in range(0,nspecies):
        flux.append(rhoU*rhoY[i]/rho)

    return flux

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def rhs_invc(cv, dv, discretization):

    x, invJac, Dr, Vand, Surf = discretization
    alpha = 0.0

    Np = x.shape[0]
    Ng = x.shape[1]

    jump_L = np.zeros(Ng,)
    jump_R = np.zeros(Ng,)
    d_cv = np.zeros((Np,Ng))
    cv_0 = np.zeros(Ng,)
    cv_N = np.zeros(Ng,)

    rho  = cv[0]
    rhoU = cv[1]
    Y = cv[3:]/rho

    p = dv[0]
    u = dv[1]
    T = dv[2]
    
    array_has_nan = np.isnan(np.sum(p))
    if (array_has_nan == True):
      return False, False

    c = np.sqrt(gamma(T, Y, eos)*p/rho)
    lm = np.abs(u) + c
    LFc = np.hstack((lm[0,0],np.maximum(lm[ 0,1:],lm[-1,0:-1]))) #XXX double check this

    array_has_nan = np.isnan(np.sum(c))
    if (array_has_nan == True):
      return False, False

    ############################

    bc = [boundary_condition( 'inlet', x[ 0, 0], cv[:, 0, 0], p[ 0, 0], Y[:, 0, 0], eos),
          boundary_condition('outlet', x[-1,-1], cv[:,-1,-1], p[-1,-1], Y[:,-1,-1], eos)]

    flux  = inviscid_flux(cv, dv)

    nhat_0 = -1.0 # normal vector on the left will point to the left (-)
    nhat_N = +1.0 # normal vector on the right will point to the right (+)

    ############################

    rhs_cv = []
    for ii in range(nEqs):

        cv_0[:] = cv[ii][ 0,:]  # first point, starts from the second element
        cv_N[:] = cv[ii][-1,:]  # last point, starts from the first element

        # compute the jump at internal interfaces
        jump_L[1:Ng-1] = cv_0[1:Ng-1]*nhat_0 + cv_N[0:Ng-2]*nhat_N #left face
        jump_R[1:Ng-1] = cv_0[2:Ng  ]*nhat_0 + cv_N[1:Ng-1]*nhat_N #right face

        # apply Dirichlet BCs and evaluate the jump at domain bnds
        cv_inlet  = bc[0][ii]
        cv_outlet = bc[1][ii]

        jump_L[ 0] = cv_0[ 0 ]*nhat_0 + cv_inlet*nhat_N
        jump_R[ 0] = cv_0[ 1 ]*nhat_0 + cv_N[ 0]*nhat_N

        jump_L[-1] = cv_0[-1 ]*nhat_0 + cv_N[-2]*nhat_N
        jump_R[-1] = cv_outlet*nhat_0 + cv_N[-1]*nhat_N

        # flux for the left face (i.e., first point)
        flux_int = np.zeros(Ng,)
        flux_ext = np.zeros(Ng,)
        flux_int[1:] = flux[ii][  0,  1:]
        flux_ext[1:] = flux[ii][ -1,0:-1]

        f_star = 0.5*((flux_int + flux_ext) + (1.0-alpha)*LFc*jump_L)
        d_cv[ 0,:] = (flux_int - f_star)*nhat_0
        
        # flux for the right face (i.e., last point)
        flux_int = np.zeros(Ng,)
        flux_ext = np.zeros(Ng,)
        flux_int[0:-1] = flux[ii][-1:,0:-1]
        flux_ext[0:-1] = flux[ii][  0,  1:]

        f_star = 0.5*((flux_int + flux_ext) + (1.0-alpha)*LFc*jump_R)
        d_cv[-1,:] = (flux_int - f_star)*nhat_N

        rhs_cv.append( invJac*np.matmul(Dr,flux[ii]) - Surf*d_cv )

    return -np.asarray(rhs_cv), True
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def grad(cv, u, discretization):

    x, invJac, Dr, Vand, Surf = discretization

    var_list = ["P", "u", "T", "X", "X", "X", "X", "X", "X", "X"]

    Np = x.shape[0]
    Ng = x.shape[1]

    nhat_0 = -1.0 # normal vector on the left will point to the left (-)
    nhat_N = +1.0 # normal vector on the right will point to the right (+)

    grad = [] 
    for ii in range(0, nEqs):

        du  = np.zeros((Np,Ng)) #note that "dq" is a full matrix now

        jump_L_u  = np.zeros(Ng,)
        jump_R_u  = np.zeros(Ng,)

        u_0  = np.zeros(Ng,)
        u_N  = np.zeros(Ng,)

        u_0[:] = u[ii, 0,:]  # first point, start from the second element
        u_N[:] = u[ii,-1,:]  #  last point, start from the first element

        # Dirichlet BC
        u_inlet  = +u_0[ 0]
        u_outlet = +u_N[-1]

        # compute the jump at internal interfaces
        jump_L_u[1:Ng-1] = u_0[1:Ng-1]*nhat_0 + u_N[0:Ng-2]*nhat_N #left face
        jump_R_u[1:Ng-1] = u_0[2:Ng  ]*nhat_0 + u_N[1:Ng-1]*nhat_N #right face

        # evaluate the jump at domain bnds
        jump_L_u[ 0] = u_0[ 0 ]*nhat_0 + u_inlet *nhat_N
        jump_R_u[ 0] = u_0[ 1 ]*nhat_0 + u_N[ 0 ]*nhat_N

        jump_L_u[-1] = u_0[-1 ]*nhat_0 + u_N[-2 ]*nhat_N
        jump_R_u[-1] = u_outlet*nhat_0 + u_N[-1 ]*nhat_N

        du[ 0,:] = +jump_L_u
        du[-1,:] = +jump_R_u

        grad.append( invJac*np.matmul(Dr, u[ii,:,:]) - (Surf)*du/2.0 )

    return np.asarray(grad)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def diff_fluxes(q, bc, discretization):

    x, invJac, Dr, Vand, Surf = discretization

    Np = x.shape[0]
    Ng = x.shape[1]

    nhat_0 = -1.0 # normal vector on the left will point to the left (-)
    nhat_N = +1.0 # normal vector on the right will point to the right (+)

    dq  = np.zeros((Np,Ng)) #note that "dq" is a full matrix now

    jump_L_q  = np.zeros(Ng,)
    jump_R_q  = np.zeros(Ng,)

    q_0  = np.zeros(Ng,)
    q_N  = np.zeros(Ng,)
             
    if len(q.shape) == 2:
        nVars = 1
        Np = q.shape[0]
        Ng = q.shape[1]
    else:
        nVars = nspecies
        Np = q[0].shape[0]
        Ng = q[0].shape[1]

    diffusive_flux = [] 
    for ii in range(0, nVars):

        if nVars == 1:
            q_0[:] = q[ 0,:]  # first point, start from the second element
            q_N[:] = q[-1,:]  # last point, start from the first element
        else:
            q_0[:] = q[ii, 0,:]  # first point, start from the second element
            q_N[:] = q[ii,-1,:]  # last point, start from the first element

        # compute the jump at internal interfaces
        jump_L_q[1:Ng-1] = q_0[1:Ng-1]*nhat_0 + q_N[0:Ng-2]*nhat_N #left face
        jump_R_q[1:Ng-1] = q_0[2:Ng  ]*nhat_0 + q_N[1:Ng-1]*nhat_N #right face

        # apply boundary conditions and evaluate the jump at domain bnds
        # XXX in this case, I am NOT imposing any BC....
        q_inlet  = q_0[ 0]
        q_outlet = q_N[-1]

        jump_L_q[0] = q_0[ 0 ]*nhat_0 +  q_inlet*nhat_N
        jump_R_q[0] = q_0[ 1 ]*nhat_0 + q_N[ 0 ]*nhat_N

        jump_L_q[-1] = q_0[-1 ]*nhat_0 + q_N[-2]*nhat_N
        jump_R_q[-1] = q_outlet*nhat_0 + q_N[-1]*nhat_N 

        dq[ 0,:] = +jump_L_q
        dq[-1,:] = +jump_R_q

        if nVars == 1:
            diffusive_flux.append(
                invJac*np.matmul(Dr, q) - (Surf)*dq/2.0
            )
        else:
            diffusive_flux.append(
                invJac*np.matmul(Dr, q[ii,:,:]) - (Surf)*dq/2.0
            )

    if nVars == 1:
        return np.asarray(diffusive_flux)[0]
    else:
        return np.asarray(diffusive_flux)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def rhs_visc(cv, dv, tv, grad_Q, discretization):

    rho = cv[0]
    Y = cv[3:]/rho
    u = dv[1]
    T = dv[2]

    #~~~ mass
    mass = grad_Q[0]*0.0 # zeros

    #~~~ momentum
    tau = 4.0/3.0*tv[1]*grad_Q[1] # shear stress: mu * du/dx
    momt = diff_fluxes(tau, None, discretization)

    #~~~ species
    W_mean = eos.get_mix_molecular_weight(Y)
    W_k = eos.wts

    V_corr = np.zeros(rho.shape)
    for ii in range(0,nspecies):
        V_corr = V_corr + tv[3+ii,:,:]*(W_k[ii]/W_mean)*grad_Q[3+ii,:,:]

    j_a = np.zeros(dv[3:].shape)
    for ii in range(0,nspecies):
        j_a[ii,:,:] = - rho*(W_k[ii]/W_mean)*tv[3+ii,:,:]*grad_Q[3+ii,:,:] + rho*Y[ii,:,:]*V_corr
    spec = diff_fluxes(j_a, None, discretization)

    #~~~ energy
    q_heat = -1.0*tv[2]*grad_Q[2] # thermal cond: - kappa * dT/dx

    h_a = eos.get_species_enthalpies_rt(T)
    spec_enrg = np.zeros(rho.shape)
    for ii in range(0,nspecies):
        j_a[ii,:,:] = - W_k[ii]/W_mean*tv[3+ii,:,:]*grad_Q[3+ii,:,:] + Y[ii,:,:]*V_corr
        spec_enrg = spec_enrg + j_a[ii] * (
            (eos.gas_constant/W_k[ii] * T) * (h_a[ii][:,:])
            )

    q_spec = rho*spec_enrg

    enrg = (
           + diff_fluxes(          u*tau, None, discretization)
           - diff_fluxes(q_heat + q_spec, None, discretization)
    )

    #~~~ output
    rhs = []
    rhs.append(mass)
    rhs.append(momt)
    rhs.append(enrg)
    for i in range(0, nspecies):
        rhs.append(-spec[i])

    #~~~ dummy
    check = np.zeros(rho.shape)
    for ii in range(0,nspecies):
        check = check - (W_k[ii]/W_mean)*tv[3+ii,:,:]*grad_Q[3+ii,:,:] + Y[ii,:,:]*V_corr

    return np.asarray(rhs), check, V_corr

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sponge_source_terms(x, cv, cv_ref):

    amplitude = 1000.0
    
    x_min = -0.25
    x_max = +0.25
    x_thickness = 0.15

    x0 = x_max - x_thickness
    dx = +((x - x0)/x_thickness)
    sponge_func = np.where(
      np.greater(x, x0),
          np.where(np.greater(x, x_max), 1.0, 3.0*dx**2 - 2.0*dx**3),
          0.0
    )

    x0 = x_min + x_thickness
    dx = -((x - x0)/x_thickness)
    sponge_func = sponge_func + np.where(
      np.less(x, x0),
          np.where(np.less(x, x_min), 1.0, 3.0*dx**2 - 2.0*dx**3),
          0.0
    )

    return amplitude * sponge_func * (cv_ref - cv)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def net_production_rates(rho, temperature, mass_fractions):

    concs = eos.get_concentrations(rho, mass_fractions)
    for i in range(nspecies):
        concs[i] = np.where(np.less(concs[i], 1e-12), zeros, concs[i])

    return eos.get_net_production_rates(rho, temperature, mass_fractions, concentrations=concs)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_species_source_terms(cv, dv, eos):

    rho = cv[0]
    T = dv[2]
    Y = cv[3:]/rho

    w_dot = eos.wts * net_production_rates(rho, T, Y)

    sources = []
    sources.append(zeros)
    sources.append(zeros)
    sources.append(zeros)
    for i in range(0, nspecies):
        sources.append(w_dot[i])

    return np.asarray(sources)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_transport_vars(cv, dv, eos):

    Lewis = 1.0
    rho = cv[0]
    P = dv[0]
    T = dv[2]

    Y = np.zeros((7),dtype=object)
    for ii in range(0,nspecies):
        Y[ii] = cv[3+ii]/rho

    mu = eos.get_mixture_viscosity_mixavg(T, Y)
    kappa = eos.get_mixture_thermal_conductivity_mixavg(T, Y)
    dij = eos.get_species_mass_diffusivities_mixavg(P, T, Y)

    tv = []
    tv.append(zeros)
    tv.append(mu)
    tv.append(kappa)
    for i in range(0, nspecies):
        tv.append(dij[i])

    return np.asarray(tv)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_dependent_vars(cv, eos, Tguess=300.0):

    rho  = cv[0]
    rhoU = cv[1]
    rhoE = cv[2]  

    Y = cv[3:]/rho
    X = np.zeros(Y.shape)
    W_k = eos.wts
    W_mean = eos.get_mix_molecular_weight(Y)

    e = (rhoE - 0.5*rhoU**2/rho)/rho
    T = eos.get_temperature(e, Tguess, Y, do_energy=True)
    p = eos.get_pressure(rho, T, Y)
    u = rhoU/rho

    dv = []
    dv.append(p)
    dv.append(u)
    dv.append(T)
    for i in range(0,nspecies):
        X[i] = Y[i]*W_mean/W_k[i]
        dv.append(X[i])

    return np.asarray( dv )

"""
##############################################################################
"""

if rstfile is None:
    cv = InitCond(x, eos)
    step = 0
    t = 0.0
else:
    filename = 'cv-' + str('%09d' % float(rstfile))
    with open('./output/' + filename + '.pickle', 'rb') as handle: 
        step, t, cv = pickle.load(handle)
    print('Starting step: ', step)
    print('Starting time: ', t)

dv = get_dependent_vars(cv, eos, Tguess=300.0)
tv = get_transport_vars(cv, dv, eos)

#============================

cv_ref = InitCond(x, eos)

#============================

print('niter = ', niter)
counter = time.time()
sub_counter = time.time()

counter_1s = 0.0
counter_2s = 0.0
counter_3s = 0.0
counter_4s = 0.0

try:
    os.mkdir('./figs')
except:
    print('"figs" already exist')

try:
    os.mkdir('./output')
except:
    print('"output" already exist')    

#============================

kk = step
#while t < tfinal:
while kk <= niter:

    counter_1a = time.time()
    dv = get_dependent_vars(cv, eos, Tguess=dv[2])
    tv = get_transport_vars(cv, dv, eos)
    counter_1s = counter_1s + time.time() - counter_1a

    c = np.sqrt( gamma(dv[2], cv[3:]/cv[0], eos)*dv[0]/cv[0] )
    dt_inv = np.min(CFL*dx/c)
    dt_vis = np.min(CFL*dx**2/tv[1][0,:])
    dt = min(dt_inv, dt_vis)

    # pre-step
    if (kk%1000 == 0):

        aux = np.zeros((nEqs+3,Ncoll*Ngrid))
        aux[  0,:] = x.reshape((Ncoll*Ngrid),order='F')
        aux[1:4,:] = cv[0:3].reshape((3,Ncoll*Ngrid),order='F')
        aux[  4,:] = dv[0].reshape((Ncoll*Ngrid),order='F')
        aux[  5,:] = dv[2].reshape((Ncoll*Ngrid),order='F')
        aux[ 6:,:] = (dv[3:].reshape((nspecies,Ncoll*Ngrid),order='F'))
        header = "x, rho, rhoU, rhoE, P, T, X_" + ", X_".join(eos.species_names)
        filename = 'cv-' + str('%09d' % kk)
        np.savetxt('./output/' + filename + '.csv', aux.T, delimiter=',', fmt='%15.6f', header=header, comments="")

    if (kk%10000 == 0):
        filename = 'cv-' + str('%09d' % kk)
        with open('./output/' + filename + '.pickle', 'wb') as handle:
            pickle.dump([kk, t, cv], handle, protocol=pickle.HIGHEST_PROTOCOL)

    if (integration == "euler"):

        # TODO: implement positivity preserving limiter
        # TODO: implement SSPRK

        #SSP RK Stage 1
        counter_2a = time.time()
        rhs_I, flag = rhs_invc(cv, dv, discr)
        counter_2s = counter_2s + time.time() - counter_2a

        counter_3a = time.time()
        grad_Q = grad(cv, dv, discr)
        counter_3s = counter_3s + time.time() - counter_3a

        counter_4a = time.time()
        rhs_V, check, V_corr = rhs_visc(cv, dv, tv, grad_Q, discr)
        counter_4s = counter_4s + time.time() - counter_4a

        if (flag == False): 
          break

        sponge = sponge_source_terms(x, cv, cv_ref)

        source = get_species_source_terms(cv, dv, eos)

        rhs = rhs_I + rhs_V + sponge + source

        cv = cv + dt*(rhs)

    #============================

    if (kk%1000 == 0):

        aux = np.zeros((nEqs+1,Ncoll*Ngrid))
        aux[ 0,:] = x.reshape((Ncoll*Ngrid),order='F')
        aux[1:,:] = grad_Q.reshape((nEqs,Ncoll*Ngrid),order='F')
        header = "x, rho, u, T, X_" + ", X_".join(eos.species_names)
        filename = 'grad-' + str('%09d' % kk)
        np.savetxt('./output/' + filename + '.csv', aux.T, delimiter=',', fmt='%20.11f', header=header, comments="")

        aux = np.zeros((nEqs+1,Ncoll*Ngrid))
        aux[ 0,:] = x.reshape((Ncoll*Ngrid),order='F')
        aux[1:,:] = rhs.reshape((nEqs,Ncoll*Ngrid),order='F')
        header = "x, rho, rhoU, rhoE, X_" + ", X_".join(eos.species_names)
        filename = 'rhs-' + str('%09d' % kk)
        np.savetxt('./output/' + filename + '.csv', aux.T, delimiter=',', fmt='%20.11f', header=header, comments="")


        aux = np.zeros((nEqs+1,Ncoll*Ngrid))
        aux[ 0,:] = x.reshape((Ncoll*Ngrid),order='F')
        aux[1:,:] = rhs_I.reshape((nEqs,Ncoll*Ngrid),order='F')
        header = "x, rho, rhoU, rhoE, X_" + ", X_".join(eos.species_names)
        filename = 'rhs_I-' + str('%09d' % kk)
        np.savetxt('./output/' + filename + '.csv', aux.T, delimiter=',', fmt='%20.11f', header=header, comments="")

        aux = np.zeros((nEqs+1,Ncoll*Ngrid))
        aux[ 0,:] = x.reshape((Ncoll*Ngrid),order='F')
        aux[1:,:] = rhs_V.reshape((nEqs,Ncoll*Ngrid),order='F')
        header = "x, rho, rhoU, rhoE, X_" + ", X_".join(eos.species_names)
        filename = 'rhs_V-' + str('%09d' % kk)
        np.savetxt('./output/' + filename + '.csv', aux.T, delimiter=',', fmt='%20.11f', header=header, comments="")

        aux = np.zeros((nEqs+1,Ncoll*Ngrid))
        aux[ 0,:] = x.reshape((Ncoll*Ngrid),order='F')
        aux[1:,:] = source.reshape((nEqs,Ncoll*Ngrid),order='F')
        header = "x, rho, rhoU, rhoE, X_" + ", X_".join(eos.species_names)
        filename = 'source-' + str('%09d' % kk)
        np.savetxt('./output/' + filename + '.csv', aux.T, delimiter=',', fmt='%20.11f', header=header, comments="")


    # post-step
    kk += 1
    t = t + dt

#    if (kk%1000 == 0):
#        snapshot = tracemalloc.take_snapshot()
#        top_stats = snapshot.statistics('lineno')
#        print("[ Top 10 ]")
#        for stat in top_stats[:10]:
#            print(stat)
#        gc.collect()

    if (kk%100 == 0):
        print('iter =', kk, ', '
              'iter_wall_time:', '{:.5f}'.format((time.time() - sub_counter)/100.0), 's ,',
              'dt =', '{:.5e}'.format(dt),'s ,',
              'sim_time =', '{:.5e}'.format(t),'s ,',
              'Y_C2H4 =', '{:.5e}'.format(np.min(dv[3])),
              end="\n")
        print('This should be zero:', np.amax(check), np.amax(V_corr), '...')
        sub_counter = time.time()

    if (flag == False): 
      break

##############################################################################

print('\n')
print('Total time of simulation: ', time.time() - counter)
print(counter_1s, 's')
print(counter_2s, 's')
print(counter_3s, 's')
print(counter_4s, 's')
print('niter = ', kk)   
print('t_final = ', t)
  
##############################################################################




#        plt.plot(x, Y[0])
#        plt.ylim(-0.001, 0.07)
#        plt.xlim(-0.01, 0.01)
#        plt.savefig(f'./figs/C2H4-{kk:09d}.png')
#        plt.close()

#        plt.plot(x, T)
#        plt.ylim(-250, 2500)
#        plt.xlim(-0.01, 0.01)
#        plt.savefig(f'./figs/T-{kk:09d}.png')
#        plt.close()

#        plt.plot(x, p)
#        plt.ylim(100000, 102000)
#        plt.xlim(-0.01, 0.01)
#        plt.savefig(f'./figs/P-{kk:09d}.png')
#        plt.close()

#        plt.plot(x, T)
#        plt.ylim(-250, 2500)
#        plt.savefig(f'./figs/T-full-{kk:09d}.png')
#        plt.close()

#        plt.plot(x, p)
#        plt.ylim(100000, 102000)
#        plt.savefig(f'./figs/P-full-{kk:09d}.png')
#        plt.close()
