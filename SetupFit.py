import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec#
import math
from astropy import units as u
import scipy.stats

plt.style.use(['science','ieee', 'no-latex'])
from multiprocessing import Pool
from astropy.modeling import models, fitting

import pandas as pd
from astropy.table import Table
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as IUSScipy
from astropy.modeling import Fittable1DModel
from astropy.modeling import Parameter
from astropy.modeling.functional_models import Gaussian1D
from astropy.modeling.physical_models import Drude1D
import os
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.special import gamma, factorial, legendre

from scipy.interpolate import interp1d
import tensorflow_probability as tfp


from jax.config import config

config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline as IUS

numpyro.set_platform("cpu")


# Ice + CH templates
ls_i, ice = np.loadtxt("./Ice Templates/IceExt.txt", unpack = True, usecols=[0,1])
ls_C, CH = np.loadtxt("./Ice Templates/CHExt.txt", unpack = True, usecols=[0,1])

# Nuclear Silicate Template
ls_S, SS = np.loadtxt("./Nuclear Templates/NGC4418SProfile.dat", unpack = True, usecols=[0,1])
ls_S, SS = np.loadtxt('./Nuclear Templates/IRAS08572SProfile.txt', unpack=True)


# Star forming extinction curve
ls, S = np.loadtxt('./Extinction Curves/KVT.txt', unpack=True)
S_np = interpolate.interp1d(ls,S/np.interp(9.8, ls, S), fill_value="extrapolate")



# Read in Star-forming continua
x,y = np.loadtxt("./ContTemplates/SFCont_"+str(0)+".dat", unpack = True, usecols=[0,1])
C_temps = np.empty((95, len(x)))
C_temps_j = jnp.empty((95, len(x)))
for i in range(95):
    x,y = np.loadtxt("./ContTemplates/SFCont_"+str(i)+".dat", unpack = True, usecols=[0,1])
    area = np.trapz(y[(x>=5.2) & (x<=14.2)], x[(x>=5.2) & (x<=14.2)])
    y/=area
    C_temps[i, :] = y
x_c = x
C_temp = [x, interp1d(np.linspace(0.0, 1.0, 95), C_temps, axis=0)]

    



class Fit():
    def __init__(self, filename, z):
    
    
        # Read in data
        ##################################

        try:
            lam, flux, flux_err = np.loadtxt(filename, unpack = True, usecols=[0,1,2,])
        except:
            lam, flux, flux_err = np.loadtxt(filename, unpack = True, usecols=[0,1,2,], skiprows=1)

            
      #  flux /= 1000
       # flux_err /=1000

        lam = lam/(1.0+z)
        scale = np.mean(flux[lam<=max(lam)])
        self.scale = scale
        
        flux = flux[lam<=14.2]/scale
        flux_err = flux_err[lam<=14.2]/scale
        lam=lam[lam<=14.2]
        flux = flux[lam>=5.2]
        flux_err = flux_err[lam>=5.2]
        lam=lam[lam>=5.2]


        
        self.ObjName = filename
        
########################################




        # Set up parameters dataframe
        self.parameters = pd.DataFrame(columns=['Section', 'Component','Name', 'Description','Value', '+Error', '-Error','Prior','Prior Type', 'Fixed'])
        
        # Initialise Model Components
        
        #Emission lines
        lines =["H2 S(7)","H2 S(6)","H2 S(5)","Ar II","NeVI", "H2 S(4)","AR III","H2 S(3)", "S IV", "H2 S(2)","Ne II","Ne IV ", "Ne III", "H2 S(1)", "S III", "Ne IV ", "O IV", "Fe II", "H2 S(0)", "S III", "Si II", "C2H2", "HCN"]
        cents =  [5.511, 6.109, 6.909, 6.985, 7.6524, 8.026, 8.991, 9.665, 10.511, 12.278, 12.813, 14.3217, 15.555, 17.035, 18.713, 24.3175, 25.910, 25.989, 28.221, 33.480, 34.815, 13.7, 14.0]
        widths = [0.053, 0.053,0.053,0.053,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.14,0.14,0.14,0.14,0.34,0.34,0.34,0.34, 0.34, 0.34, 0.1, 0.1] # Spitzer
        vals = [0.000757799755090427, 0.00029842545884589856, 0.00409849610889481, 0.045355842828490714, 0.030825202361984858, 0.013389637498470245, 0.023938848033417545, 0.01829149007506314, 0.04773620924836646, 0.01969443135828826, 0.39347772642591167, 0.01816892897538409, 0.2640011203300603, 0.035319393780289365, 0.5476647269591665, 0.003334490580185104, 0.0011990672414200704, 0.004236638465562518, 0.024856938574108832, 0.7463738764070731, 0.9867613111587851, -0.0031418002198900824, -0.0031418002198900824]

        self.Nlines = 0
        self.linecents=[]
        self.linenames=[]
        for i in range(len(lines)):
            if (np.min(lam) <= cents[i] <= np.max(lam)):
                self.Nlines += 1
                self.linecents.append(cents[i])
                self.linenames.append(lines[i])
                amp_lower = 0.0
                amp_upper = 5.0
                
                if (lines[i] == "C2H2" or lines[i] =="HCN"):
                    amp_lower = -5.0
                    amp_upper = -0.0

                self.parameters = self.parameters.append({ 'Section': 'Lines', 'Component': lines[i],'Name': "AMP("+lines[i]+")",'Description': 'Line flux', 'Value': vals[i], '+Error': 0.0, '-Error': 0.0,'Prior': [amp_lower, amp_upper],'Prior Type': 'Uniform', 'Fixed': False}, ignore_index=True)
                self.parameters = self.parameters.append({ 'Section': 'Lines', 'Component': lines[i],'Name': "CENT("+lines[i]+")",'Description': 'Line centre', 'Value': cents[i], '+Error': 0.0, '-Error': 0.0,'Prior': [cents[i] - 0.05, cents[i] + 0.05],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
                self.parameters = self.parameters.append({ 'Section': 'Lines', 'Component': lines[i],'Name': "FWHM("+lines[i]+")",'Description': 'Line width', 'Value': widths[i], '+Error': 0.0, '-Error': 0.0,'Prior': [widths[i] - 0.1*widths[i], widths[i] + 0.1*widths[i]],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

            
        # PAH features
        cents = [5.27, 5.70, 6.22, 6.69, 7.42, 7.60, 7.85, 8.33, 8.61, 10.68, 11.23, 11.33, 11.99, 12.62, 12.69, 13.48, 14.04, 14.19, 15.90, 16.45, 17.04, 17.375, 17.87, 18.92, 33.10]
        widths = [0.179, 0.2, 0.187, 0.468, 0.935, 0.334, 0.416, 0.417, 0.336, 0.214, 0.135, 0.363, 0.54, 0.53, 0.165, 0.539, 0.225, 0.355, 0.318, 0.230, 1.108, 0.209, 0.286, 0.359, 1.655]
        vals=[0.011682602057147044, 0.011647804266876815, 0.19360656656030342, 0.015532690753316257, 0.02699845206633171, 0.18595623096523864, 0.19356401038752674, 0.035393129492062275, 0.13514692087725702, 0.03408052383057687, 0.2344523396350007, 0.2384634688004408, 0.08393517025601546, 0.16927732197440273, 0.025177535029095217, 0.044238162113021375, 0.015171033322684534, 0.006057457240455767, 0.0012839631547702755, 0.09515971140918132, 0.09273141664105061, 0.04660970606762026, 0.06980243500425296, 0.05766352808288334, 0.3534249388595553]

        print(len(cents), len(vals))
        feats = ['{:.3f}'.format(x) for x in cents]
        self.Npah = 0
        self.pahcents=[]
        self.pahnames=[]
        for i in range(len(cents)):
            if (np.min(lam) <= cents[i] <= np.max(lam)+0.1):
                self.Npah +=1
                self.pahcents.append(cents[i])
                self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "AMP("+"PAH"+feats[i]+")",'Description': 'PAH flux', 'Value': vals[i], '+Error': 0.0, '-Error': 0.0,'Prior': [0.0, 5.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
                self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "CENT("+"PAH"+feats[i]+")",'Description': 'PAH centre', 'Value': cents[i], '+Error': 0.0, '-Error': 0.0,'Prior': [cents[i] - 0.05, cents[i] + 0.05],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
                self.parameters = self.parameters.append({ 'Section': 'PAH', 'Component': feats[i],'Name': "FWHM("+"PAH"+feats[i]+")",'Description': 'PAH width', 'Value': widths[i], '+Error': 0.0, '-Error': 0.0,'Prior': [widths[i] - 0.1*widths[i], widths[i] + 0.1*widths[i]],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

            
        # Continuum
        NBBs = 9
        Temps=[35.0, 40.0,50.0, 65.0, 90.0, 135.0, 200.0, 300.0, 5000.0]
        vals=[7.258475839089247, 0.05726475219959023, 0.00913500217227525, 5.47157007390233, 29.596107143863424, 0.0468312316896768, 2.5736281112212662, 0.35697683678906766, 0.0013024385471208525]

      #  Temps=[   200.0, 300.0]
      #  for i in range(NBBs):
          #  self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "A"+str(i+1),'Description': 'BB Amp', 'Value': vals[i], '+Error': 0.0, '-Error': 0.0,'Prior': [0.0	, 50.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
          #  self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "T"+str(i+1),'Description': 'BB temp', 'Value': Temps[i], '+Error': 0.0, '-Error': 0.0,'Prior': [35.0, 1500],'Prior Type': 'Uniform','Fixed': True}, ignore_index=True)
        self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "S",'Description': 'Continuum Scale', 'Value': 1.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0, 50.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
        self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "beta",'Description': 'Nuclear Fraction', 'Value': 0.0, '+Error': 0.0, '-Error': 0.0,'Prior': [ 0.0, 1.0, 0.0, 1.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
        self.parameters = self.parameters.append({ 'Section': 'Continuum', 'Component': 'Continuum','Name': "NTemp",'Description': 'Continuum Temp', 'Value': 0.5, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0, 1.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

        self.parameters = self.parameters.append({ 'Section': 'Extinction', 'Component': 'Sil','Name': "\u03C4_Sil(SF)",'Description': 'Silicate Opt Depth', 'Value': 0.001, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0, 50.0, 0.0, 1.1],'Prior Type': 'TruncatedNormal','Fixed': False}, ignore_index=True)

        #Extract sampled parameters
        sampled_parameters = self.parameters.loc[(self.parameters["Fixed"] == False)]
        init_pos = sampled_parameters['Value'].to_numpy()
        
        self.Npar = sampled_parameters.shape[0]
        self.nchains = 5
        self.Nwalkers = int(2.0*self.Npar + 1.0)
        
        # Generate arrays and indices for passing into prob funcs
        self.ps = self.parameters["Value"].to_numpy()
        self.priors = self.parameters["Prior"].to_numpy()
        # Section indices
        self.lines_indx = self.parameters.index[self.parameters["Section"] == "Lines"].to_numpy()
        self.pah_indx = self.parameters.index[self.parameters["Section"] == "PAH"].to_numpy()
        self.cont_indx = self.parameters.index[self.parameters["Section"] == "Continuum"].to_numpy()
        self.ext_indx = self.parameters.index[self.parameters["Section"] == "Extinction"].to_numpy()
        
        self.sampled_indx = self.parameters.index[self.parameters["Fixed"] == False].to_numpy()
        
        self.fixed =self.parameters["Fixed"].to_numpy()
        
        self.pos=init_pos
        
        self.data = [lam, flux, flux_err]
        
        self.samples_flat = None
        

####################################################################### Functions ###############################################################




def complexe_modulo(z):
    a = z.real
    b = z.imag
    return a**2+b**2
    

def ModifiedGauss3(lam, x0, fwhm, a, b, numpy):
   # a = a/fwhm
    

    if (numpy == False):
        fwhm = 2.0*fwhm/(1.0+tt.exp(a*(lam-x0)))
        sig = fwhm/2.355
        return tt.exp(-0.5*((lam-x0)/sig)**2)
    else:
        fwhm = 2.0*fwhm*(1./(1.0+np.exp(a*(lam-x0))))# +np.exp(-0.5*((lam-x0)/b)**2))
        sig = fwhm/2.355
        return np.exp(-0.5*((abs(lam-x0))/sig)**b)

def ModifiedGauss2(lam, x0, fwhm, a, b, numpy):
    
    b = a/b
    G = np.empty(len(lam))
   # fwhm1 = 2.0*fwhm*(1./(1.0+np.exp(a*(lam-x0))) )# +np.exp(-0.5*((lam-x0)/b)**2))
    sig = fwhm/2.355
   # fwhm2 = 2.0*fwhm*(1./(1.0+np.exp(b*(lam-x0))) )# +np.exp(-0.5*((lam-x0)/b)**2))
   # sig2 = fwhm2/2.355   
   # G[lam<=x0] = np.exp(-0.5*((abs(lam[lam<=x0]-x0))/sig1[lam<=x0])**2)
   # G[lam>x0] =  np.exp(-0.5*((abs(lam[lam>x0]-x0))/sig2[lam>x0])**2)
    G[lam<=x0] = np.exp(-0.5*((abs(lam[lam<=x0]-x0))/sig)**a)
    G[lam>x0] =  np.exp(-0.5*((abs(lam[lam>x0]-x0))/sig)**b)
    
    return G



        
        
        
    
    
def Ext(lam, ext_params, jax=False):
    tau_9 = ext_params[0]
    if (jax==False):
        if (tau_9 == 0.0):
            full_ext = np.ones(len(lam))
        else:
            full_ext = (1.0 - np.exp(-tau_9*S_np(lam)))/(tau_9*S_np(lam))
        #full_ext = np.exp(-tau_9*S_np(lam))
        return full_ext#, ext_ice*ext_CH, ext_S, tau_9*(1. - beta), tau_ice/(ice_np(6.0)+CH_np(6.0)), Psi
    else:
        full_ext = (1.0 - jnp.exp(-tau_9*jnp.interp(lam, ls, S)))/(tau_9*jnp.interp(lam, ls, S))
        #full_ext = np.exp(-tau_9*S_np(lam))
        return full_ext#, ext_ice*ext_CH, ext_S, tau_9*(1. - beta), tau_ice/(ice_np(6.0)+CH_np(6.0)), Psi
        







  
def Gauss(X,  l0, fwhm, A, jax=False):
    sig = fwhm/2.355
    
    if (jax==False):
        return A*np.exp(-0.5*((X-l0)/sig)**2)
    else:
        return A*jnp.exp(-0.5*((X-l0)/sig)**2)

def ModifiedGauss(lam, x0, fwhm, a, numpy):

    if (numpy == False):
        fwhm = 2.0*fwhm/(1.0+tt.exp(a*(lam-x0)))
        sig = fwhm/2.355
        return tt.exp(-0.5*((lam-x0)/sig)**2)
    else:
        fwhm = 2.0*fwhm/(1.0+np.exp(a*(lam-x0)))
        sig = fwhm/2.355
        return np.exp(-0.5*((lam-x0)/sig)**2)


def Drude(lam, x0, fwhm, A):
    gamma = fwhm/x0
    
    return A*(gamma**2)/((((lam/x0)- (x0/lam))**2) + gamma**2)

def ModifiedDrude(lam, x0, w, a, numpy):
    if (numpy ==True):
        fwhm = 2.0*w/(1.0+np.exp(a*(lam-x0)))
    else:
        fwhm = 2.0*w/(1.0+pm.math.exp(a*(lam-x0)))
    gamma = fwhm/x0
    return (gamma**2)/((((lam/x0)- (x0/lam))**2) + gamma**2)#/((gamma**2)#/((((9.8/x0)- (x0/9.8))**2) + gamma**2))
    
    
    

def B_nu(x, A, T):

    
    c=299792458
    h=6.62607004e-34
    k = 1.38064852e-23

        
    x = x*1e-6 #micron to metres
    l_peak = 2.897771e-3/T
    nu = c/x
    nu_peak = c/l_peak


    norm = ((nu_peak**3)/(np.exp(h*nu_peak/(k*T))-1.0))*((1./l_peak)**2)
    return (A* (nu**3)/(np.exp(h*nu/(k*T))-1.0))*((1./x)**2)/norm # Return in Jy
    

def B_n(x, A, T):

    
    c=299792458
    h=6.62607004e-34
    k = 1.38064852e-23

        
    x = x*1e-6 #micron to metres
    l_peak = 2.897771e-3/T
    nu = c/x
    nu_peak = c/l_peak


    norm = ((nu_peak**3)/(np.exp(h*nu_peak/(k*T))-1.0))
    return (A* (nu**3)/(np.exp(h*nu/(k*T))-1.0))/norm # Return in Jy
    
    


def PearIV(x, x_peak, w, m, nu, numpy=False):
    
    l0 = x_peak + 0.5*w*nu/m
    
    if (numpy == True):
        return ((1.+((x-l0)/w)**2)**(-1.*m))*np.exp(-1.*nu*np.arctan((x-l0)/w))
    else:

        return ((1.+((x-l0)/w)**2)**(-1.*m))*pm.math.exp(-1.*nu*tt.arctan((x-l0)/w))  
        
        
def IceExt(lam, ext_params, jax=False):
    tau_ice = ext_params[0]
    tau_CH = ext_params[0]
    
    ext = jnp.exp(-tau_ice*jnp.interp(lam, ls_i, ice))*jnp.exp(-tau_CH*jnp.interp(lam, ls_C, CH))
    return ext
        
        
def PowerCont(lam, dust_parameters):
    A1 = dust_parameters[0]
    A2 = dust_parameters[1]
    P1 = dust_parameters[2]
    P2 = dust_parameters[3]
    
    return A1*(lam/6.0)**P1 + A2*(lam/6.0)**P2 
        
        
        
# Cold dust blackbodies
norms = np.array([5.590528403707989e-07, 1.023889562397611e-05, 0.0005243018541441268, 0.016136087080743204, 0.2875392698161289, 2.3065280542734126, 6.0134433660306374, 8.100722521435312, 0.6379965132143024])
Temps= np.array([35.0, 40.0, 50.0, 65.0, 90.0, 135.0, 200.0, 300.0, 5000.0])
def DustCont(lam, dust_parameters, jax = False):
    #models=np.empty((len(lam), int(len(dust_parameters)/2.0))) # Store each of the blackbody components for plotting


    if (jax ==False):
        model = 0.0
        models=[]
        for i in range(int(len(dust_parameters)/2.0)):
            a = int(2.0*i) # Index for BB amps
            t =  int(2.0*i + 1.0) # Index for BB temps

            if (dust_parameters[t] == 5000):
                model += B_n(lam,  dust_parameters[a], dust_parameters[t])#/norms[Temps == dust_parameters[t]]
                models.append(B_n(lam, dust_parameters[a], dust_parameters[t]))#/norms[Temps == dust_parameters[t]])

            else:
                model += B_nu(lam,  dust_parameters[a], dust_parameters[t])#/BBNorm(dust_parameters[t])
                models.append(B_nu(lam, dust_parameters[a], dust_parameters[t]))#/BBNorm(dust_parameters[t]))
        return model, models#np.sum(models, axis=1), models

    else:
        model = jnp.zeros(len(lam))
        for i in range(int(len(dust_parameters)/2.0) - 1):
            a =int(2.0*i) # Index for BB amps
            t =  int(2.0*i + 1.0) # Index for BB temps

            #if (dust_parameters[t] == 5000):
               # model += B_n(lam,  dust_parameters[a], dust_parameters[t], jax =jax)/norms_J[Temps_J == dust_parameters[t]]

           # else:
            model += B_nu(lam,  dust_parameters[a], dust_parameters[t], jax =jax)#/norms_J[Temps_J == dust_parameters[t]]
    
        return model +  B_n(lam,  dust_parameters[-2], dust_parameters[-1], jax =jax)#/norms_J[Temps_J == dust_parameters[-1]]

        


    
def contTemp(lam, dust_parameters, jax = False):

    NTemp = dust_parameters[2]
      
    
    if (jax==False):
       # cont = C_temps[int(NTemp)]
        cont = tfp.substrates.jax.math.interp_regular_1d_grid(NTemp, 0.0, 1.0, C_temps, axis=0,fill_value='constant_extension', fill_value_below=None,fill_value_above=None, grid_regularizing_transform=None, name=None)
    
        cont = np.interp(lam, x_c, cont)
    else:
       # print(NTemp)
       # cont = C_temps[jnp.array([NTemp], int)]
        #cont = jnp.interp(NTemp, jnp.linspace(0.0, 1.0, 103), C_temps)
        cont = tfp.substrates.jax.math.interp_regular_1d_grid(NTemp, 0.0, 1.0, C_temps, axis=0,fill_value='constant_extension', fill_value_below=None,fill_value_above=None, grid_regularizing_transform=None, name=None)
        cont = jnp.interp(lam, x_c, cont)
    return cont


def NUCTemp(lam, nuc_params, jax = False):

  #  NTemp = int(dust_parameters[2])
    x1,x3 = nuc_params[0:2]
    tau_ice = nuc_params[2]
    tau_s =   nuc_params[3]
    

    
  #  ext_ice = np.exp(-1.0*tau_ice*ice_np(lam))*np.exp(-1.0*tau_ice*CH_np(lam))
    
  #  if (NTemp>0):
       # return ext_ice*N_temps[NTemp](lam), N_temps_Unobscured[NTemp](lam), N_temps_IceCorr[NTemp](lam)
   # else:

        #spec = N_temps[int(NTemp)]
       # spec = jnp.interp(lam, x_N, spec)
       # spec_Unobscured = N_temps_Unobscured[int(NTemp)]
        #spec_Unobscured = jnp.interp(lam, x_N, spec_Unobscured)       
        #spec_IceCorr = N_temps_IceCorr[int(NTemp)]
        #spec_IceCorr = jnp.interp(lam, x_N, spec_IceCorr)


    if (jax ==True):
        knots = jnp.array([x1, 1.0, x3])
        ext_ice = jnp.exp(-1.0*tau_ice*jnp.interp(lam, ls_i, ice))*jnp.exp(-1.0*tau_ice*jnp.interp(lam, ls_C, CH))
        ext_sil = jnp.exp(-1.0*tau_s*jnp.interp(lam, ls_S, SS))
           # spec = ext_ice*spec_IceCorr
        #power = (lam/9.8)**nuc_params[0]
        xknots = jnp.linspace(min(lam), max(lam), 3)
        
        
        power = IUS(xknots, knots, k=2)(lam)
           
        cont = ext_ice*ext_sil*power
        
        x = jnp.linspace(5.2, 14.2, 200)
        area = jnp.trapz(jnp.interp(x, lam, cont), x)
    else:
        knots = np.array([x1, 1.0, x3])
        ext_ice = np.exp(-1.0*tau_ice*np.interp(lam, ls_i, ice))*np.exp(-1.0*tau_ice*np.interp(lam, ls_C, CH))
        ext_sil = np.exp(-1.0*tau_s*np.interp(lam, ls_S, SS))
           # spec = ext_ice*spec_IceCorr
        #power = (lam/9.8)**nuc_params[0]
        xknots = np.linspace(min(lam), max(lam), 3)
        
        
        power = IUSScipy(xknots, knots, k=2)(lam)
           
        cont = ext_ice*ext_sil*power
        
        x = np.linspace(5.2, 14.2, 200)
        area = np.trapz(jnp.interp(x, lam, cont), x)
       
    spec = cont/area
    spec_Unobscured = power/area
    spec_IceCorr = ext_sil*power/area

    return spec, spec_Unobscured, spec_IceCorr


# Stellar SED
def Stellar(lam, stellar_parameters, numpy = False):
        return B_n(lam,  stellar_parameters, 5000.0, numpy)

# spectral lines
def Lines(lam, lines_parameters, jax=False):

    if (jax==False):
        model = 0.0
        for i in range(int(len(lines_parameters)/3.0)):
            a = int(3.0*i) # Index for amps
            c =  int(3.0*i + 1.0) # Index for centres
            w =  int(3.0*i + 2.0) # Index for widths

            model += Gauss(lam, lines_parameters[c], lines_parameters[w], lines_parameters[a],jax=jax)
    else:
        model = jnp.zeros(len(lam))
        for i in range(int(len(lines_parameters)/3.0)):
            a =int(3.0*i) # Index for amps
            c =  int(3.0*i + 1.0) # Index for centres
            w =  int(3.0*i + 2.0) # Index for widths

            model += Gauss(lam, lines_parameters[c], lines_parameters[w], lines_parameters[a], jax=jax)
    
    
    
    return model
    
    


def PAH(lam, pah_parameters, jax = False):

    if (jax == False):
        model = 0.0
        PAHS = np.empty((int(len( pah_parameters)/3.0), len(lam)))
        strengths = np.empty((int(len( pah_parameters)/3.0)))
        for i in range(int(len( pah_parameters)/3.0)):
            a = int(3.0*i) # Index for amps
            c =  int(3.0*i + 1.0) # Index for centres
            w =  int(3.0*i + 2.0) # Index for widths

            model+=Drude(lam,  pah_parameters[c],  pah_parameters[w], pah_parameters[a])
            PAHS[i, :] =Drude(lam,  pah_parameters[c],  pah_parameters[w], pah_parameters[a])
            strengths[i] = 2.9979246e14 * 0.5 * pah_parameters[a]*pah_parameters[w]*1.0e-9*np.pi/(pah_parameters[c])
        return model, PAHS#, ratioCheck

    else:
        model = jnp.zeros(len(lam))
        
        for i in range(int(len( pah_parameters)/3.0)):
            a = int(3.0*i) # Index for amps
            c = int(3.0*i + 1.0) # Index for centres
            w =  int(3.0*i + 2.0) # Index for widths

            model+=Drude(lam,  pah_parameters[c],  pah_parameters[w], pah_parameters[a])
        return model


