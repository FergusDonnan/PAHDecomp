


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec#
import math
from astropy.modeling.models import BlackBody
from astropy import units as u
import scipy.stats
import corner
import pickle
from astropy.io import fits
from astropy.io import ascii
plt.style.use(['science','ieee', 'no-latex'])

from multiprocessing import Pool
from astropy.modeling import models, fitting
from astropy.modeling import models, fitting
import time
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d
from tqdm import tqdm
import pickle
import SetupFit
from scipy.interpolate import CubicSpline
from scipy.misc import derivative
from scipy.special import gamma, factorial, legendre

from scipy.stats import uniform, loguniform


import sys

    
from astropy.table import Table
from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d
import pandas as pd
import os
from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax.config import config

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")


#############################################################################################################################################################
import warnings
warnings.filterwarnings("ignore")


    

def numpyro_model(data_list, parameters_list, lines_indx, pah_indx, cont_indx, ext_indx, sampled_indx, priors, fixed,  priors_lower, priors_upper, Nuclearparameters, priors_Nuc, priors_mean, priors_sig):
    
    
    
    # Set up nuclear parameters
    P=0
    ps_Nuc=[]
    for i in range(len(Nuclearparameters)):
        name = Nuclearparameters["Name"][i]

        value = Nuclearparameters["Value"][i]
        
        pr_type = Nuclearparameters['Prior Type'][i]
        
        if (Nuclearparameters["Fixed"][i] == False):
            P +=1
            if (pr_type!='TruncatedNormal'):
                value = numpyro.sample(name, dist.Uniform(priors_Nuc[i][0], priors_Nuc[i][1]))
            else:
                value = numpyro.sample(name, dist.TruncatedNormal(loc = priors_Nuc[i][0], scale = priors_Nuc[i][1], low = priors_Nuc[i][2], high = priors_Nuc[i][3]))
        ps_Nuc.append(value)
    ps_Nuc = jnp.array(ps_Nuc)
    
    
    
    
    # Set up parameters to be sampled and which are fixed
    y_data=[]
    err=[]
    Nuc_fraction_sum=0
    EWPr=0.0
    ratio_pr=0.0
    ratio_pr1=0.0
    ratio_pr2=0.0
    nuc_frac_diff_pr=0.0
    for k in range(len(data_list)):
        data = data_list[k]
        x_data = data[0]
        y_data+=data[1].tolist()
        err+=data[2].tolist()
        parameters = parameters_list[k]
        ps=[]
        for i in range(len(parameters)):
            name = parameters["Name"][i]+str(k+1)

            value = parameters["Value"][i]
            
            pr_type = parameters['Prior Type'][i]
            
            if (np.any(sampled_indx == i)):
                P +=1
                if (pr_type!='TruncatedNormal'):
                    value = numpyro.sample(name, dist.Uniform(priors_lower[i], priors_upper[i]))
                else:
                    value = numpyro.sample(name, dist.TruncatedNormal(loc = priors_lower[i], scale = priors_upper[i], low = priors_mean[i], high = priors_sig[i]))
            ps.append(value)
        ps = jnp.array(ps)
        Ext = SetupFit.Ext(x_data, ps[ext_indx], jax=True)
            # Line model
        lines = SetupFit.Lines(x_data, ps[lines_indx], jax=True) 
        pahs = SetupFit.PAH(x_data, ps[pah_indx], jax=True)
        lines+=pahs
            # Continuum
        cont = SetupFit.contTemp(x_data, ps[cont_indx], jax=True)

        Nuc_fraction_sum += ps[cont_indx][1]
            # Define model
        beta =ps[cont_indx][1]
        if (k==0):
            model = (lines + ps[cont_indx][0]*(1.0-beta)*cont)*Ext + ps[cont_indx][0]*beta*SetupFit.NUCTemp(x_data, ps_Nuc, jax=True)[0]
        else:
            model = jnp.concatenate((model, (lines + ps[cont_indx][0]*(1.0-beta)*cont)*Ext + ps[cont_indx][0]*beta*SetupFit.NUCTemp(x_data, ps_Nuc, jax=True)[0]))
            # Prior to ensure decreasing nuclear fraction with aperture size
            nuc_frac_diff_pr+= jnp.log(jnp.tanh(20*jnp.sqrt(beta_prev - beta)))
        beta_prev=beta
            
        # Define star-forming continuum
        cont_SF = (ps[cont_indx][0]*(1.0-beta)*cont)

        
        # Restrinct hcn line amplitude
        HCN_amp = ps[lines_indx][-3]
        C2H2_amp =ps[lines_indx][-6]
        ratio =HCN_amp/C2H2_amp
        ratio_pr2 +=jnp.log(jnp.exp(-0.5*((ratio - 0.9)/0.1)**2))

        
        
        # Normal prior on PAH flux to cont SF ratio
        area_cont = jnp.trapz(Ext*cont_SF, x_data)
        area_pah = jnp.trapz(Ext*pahs, x_data)
        ratio = area_pah/area_cont
        ratio_pr += jnp.log(jnp.exp(-0.5*((ratio - 1.92)/10.0)**2)) # Mean of 1.92, Std of 10.0 (Can change to be tighter prior)
            

            
            
    y_data = np.array(y_data)
    err = np.array(err)
   # numpyro.sample("obs", dist.Normal(model, err), obs=y_data)

    N = len(y_data)
    


        
    prob = numpyro.sample("obs", dist.Normal(model, err), obs=y_data)

    numpyro.factor("Penalty", prob +  ratio_pr2 + ratio_pr +  nuc_frac_diff_pr)



    
def RunFit(obj, z, datadir, apertures):


    parameters_list = []
    data = []
    ps=[]
    SCALES=[]
    ObjNames=[]
    ObjName = obj
    for i in range(len(apertures)):
        if (str(apertures[i])==""):
            filename = datadir+obj+".txt"
            ObjNames.append(obj)
        else:
            filename = datadir+obj+"_"+str(apertures[i])+".txt"
            ObjNames.append(obj+"_"+str(apertures[i]))

        setup = SetupFit.Fit(filename, z)
        parameters_list.append(setup.parameters)
        data.append(setup.data)
        ps.append(setup.ps)
        lines_indx = setup.lines_indx
        pah_indx = setup.pah_indx
        cont_indx = setup.cont_indx
        ext_indx = setup.ext_indx
        sampled_indx = setup.sampled_indx
        priors = setup.priors
        fixed = setup.fixed
        SCALES.append(setup.scale)
    
    
   # print('N_data = ',len(data[0]))
    #print('N_params = ',len(sampled_indx))
    # Add nuclear parameters
    Nuclearparameters = pd.DataFrame(columns=['Section', 'Component','Name', 'Description','Value', '+Error', '-Error','Prior','Prior Type', 'Fixed'])
   # Nuclearparameters = Nuclearparameters.append({ 'Section': 'Nucleus', 'Component': 'Nucleus Continuum','Name': "alpha",'Description': 'Power Law', 'Value': 0.5, '+Error': 0.0, '-Error': 0.0,'Prior': [1.3, 5.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
    Nuclearparameters = Nuclearparameters.append({ 'Section': 'Nucleus', 'Component': 'Nucleus Continuum','Name': "x1",'Description': 'Knots', 'Value': 0.5, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0, 1.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
    Nuclearparameters = Nuclearparameters.append({ 'Section': 'Nucleus', 'Component': 'Nucleus Continuum','Name': "x3",'Description': 'Knots', 'Value': 1.5, '+Error': 0.0, '-Error': 0.0,'Prior': [1.0, 10.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

    Nuclearparameters= Nuclearparameters.append({ 'Section': 'Nucleus', 'Component': 'Ice','Name': "\u03C4_Ice",'Description': 'Ice Opt Depth', 'Value': 0.0, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0, 2.0],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)
    Nuclearparameters = Nuclearparameters.append({ 'Section': 'Nucleus', 'Component': 'Sil','Name': "\u03C4_Sil(Nuc)",'Description': 'Nucleus Opt Depth', 'Value': 4.1, '+Error': 0.0, '-Error': 0.0,'Prior': [0.0, 10.0 ],'Prior Type': 'Uniform','Fixed': False}, ignore_index=True)

    ps_Nuc =  Nuclearparameters["Value"].to_numpy()
    priors_Nuc = Nuclearparameters["Prior"].to_numpy()

    # Setup priors
    prs=setup.priors#[sampled_indx]
    bounds=[]
    priors_lower = np.empty(len(prs))
    priors_upper = np.empty(len(prs))
    priors_mean = np.empty(len(prs))
    priors_sig = np.empty(len(prs))
    for i in range(len(prs)):
        bounds.append((prs[i][0], prs[i][1]))
        priors_lower[i] = prs[i][0]
        priors_upper[i] = prs[i][1]
        if (len(prs[i])>2):
            priors_mean[i] = prs[i][2]
            priors_sig[i]  = prs[i][3]
        
        
    
    
############ Run MCMC ###############
    nuts_kernel = NUTS(numpyro_model, dense_mass=False)
    mcmc = MCMC(nuts_kernel, num_warmup=4000, num_samples=4000)
    rng_key = random.PRNGKey(int(np.random.uniform(0.0, 1e9)))
    mcmc.run(rng_key, data, parameters_list, lines_indx, pah_indx, cont_indx, ext_indx, sampled_indx, priors, fixed,  priors_lower, priors_upper, Nuclearparameters, priors_Nuc, priors_mean, priors_sig)

    mcmc.print_summary()
    samples = mcmc.get_samples()






############ Read best fit parameters ###############

    Nuc_params=[]
    Nuc_uncs_up =[]
    Nuc_uncs_low=[]
    Nuc_all_params = np.empty((Nuclearparameters.shape[0], len(samples[Nuclearparameters["Name"][0]])))
    j=0
    
    for i in range(len(Nuclearparameters)):
        name = Nuclearparameters["Name"][i]

       # value = parameters["Value"][i]
        Nuc_all_params[i] =np.full( len(samples[Nuclearparameters["Name"][0]]), Nuclearparameters['Value'][i])

        if (Nuclearparameters["Fixed"][i] == False):

            p_samples = samples[name]
            value = np.percentile(p_samples, [16, 50, 84])[1]
            Nuc_params.append(np.percentile(p_samples, [16, 50, 84])[1])
            Nuc_uncs_up.append(np.percentile(p_samples, [16, 50, 84])[2] - np.percentile(p_samples, [16, 50, 84])[1])
            Nuc_uncs_low.append(np.percentile(p_samples, [16, 50, 84])[1] - np.percentile(p_samples, [16, 50, 84])[0])
            Nuc_all_params[i] = np.array(p_samples)
            
            
    Nuclearparameters.loc[Nuclearparameters['Fixed'] == False, 'Value'] = Nuc_params
    Nuclearparameters.loc[Nuclearparameters['Fixed'] == False, '+Error'] = Nuc_uncs_up
    Nuclearparameters.loc[Nuclearparameters['Fixed'] == False, '-Error'] = Nuc_uncs_low
    print("")
    print("")
    print("NUCLEAR PARAMETERS")
    print("")
    Table.pprint_all(Table.from_pandas(Nuclearparameters))
    ps_Nuc = Nuclearparameters["Value"].to_numpy()






    for k in range(len(ObjNames)):

        parameters = parameters_list[k]
        lam, flux, flux_err = data[k]
        SCALE = SCALES[k]

        params=[]
        uncs_up =[]
        uncs_low=[]
        all_params = np.empty((setup.parameters.shape[0], len(samples[parameters["Name"][0]+str(k+1)])))
        j=0

        for i in range(len(parameters)):
            name = parameters["Name"][i]+str(k+1)

           # value = parameters["Value"][i]
            all_params[i] =np.full( len(samples[parameters["Name"][0]+str(k+1)]), setup.parameters['Value'][i])

            if (np.any(sampled_indx == i)):
            
                p_samples = samples[name]
                value = np.percentile(p_samples, [16, 50, 84])[1]
                params.append(np.percentile(p_samples, [16, 50, 84])[1])
                uncs_up.append(np.percentile(p_samples, [16, 50, 84])[2] - np.percentile(p_samples, [16, 50, 84])[1])
                uncs_low.append(np.percentile(p_samples, [16, 50, 84])[1] - np.percentile(p_samples, [16, 50, 84])[0])
                all_params[i] = np.array(p_samples)
                
        # Generate Output Directory
        resultsdir ="./Results/"+ObjName+"/"+str(apertures[k])+"/"
        
        if not os.path.exists("./Results/"):
            os.mkdir("./Results/")
        if not os.path.exists("./Results/"+ObjName+"/"):
            os.mkdir("./Results/"+ObjName+"/")
            
        if not os.path.exists(resultsdir):
            os.mkdir(resultsdir)

     
        setup.parameters.loc[setup.parameters['Fixed'] == False, 'Value'] = params
        setup.parameters.loc[setup.parameters['Fixed'] == False, '+Error'] = uncs_up
        setup.parameters.loc[setup.parameters['Fixed'] == False, '-Error'] = uncs_low

        
        setup.ps = setup.parameters["Value"].to_numpy()

        ps = setup.ps
            
            
        Nuclearparameters.to_csv(resultsdir+ObjNames[k]+"Nuclearparameters.csv",  index=False) 
            
        # Print best fit parameters
        print("")
        print("")
        print("BEST FIT PARAMETERS")
        print("")
        Table.pprint_all(Table.from_pandas(setup.parameters))

        ################################### PLOTTING ######################################################

        plt.style.use(['science','ieee', 'no-latex'])
        
        #plt.figure(1)
        fig,ax = plt.subplots()


        wav = np.linspace(min(lam), max(lam), 1000)
        
        #flux, flux_err = ScaleModule(lam, flux, flux_err, ps[ext_indx], self.SL2Mask)
        # Plot data
        ax.errorbar(lam, flux*SCALE, color="black", yerr=flux_err*SCALE, ls='none',fmt='o', mfc='none', ms=.5)

        
        # Ext parameters
        ext = SetupFit.Ext(wav, ps[ext_indx])

        

        SFcont =   SetupFit.contTemp(wav, ps[cont_indx])
        NUCcont = SetupFit.NUCTemp(wav, ps_Nuc)[0]
        beta = np.exp(ps[cont_indx][1])#*np.tanh(20.0*ps[cont_indx][1])
        beta = ps[cont_indx][1]
        cont =  ps[cont_indx][0]*(beta*NUCcont + (1. - beta)*SFcont*ext)

        ax.plot(wav, cont*SCALE, color="goldenrod", ls="solid", lw=0.25, label="Continuum", zorder = 1)
        ax.plot(wav, ps[cont_indx][0]*(beta*NUCcont)*SCALE, color="goldenrod", ls="dashed", lw=0.25,  zorder = 0, label='Nucleus')
        ax.plot(wav, ps[cont_indx][0]*( (1. - beta)*SFcont*ext)*SCALE, color="goldenrod", ls="dashdot", lw=0.25,  zorder = 0, label='SF')

        ax.plot(wav, ps[cont_indx][0]*(beta*SetupFit.NUCTemp(wav, ps_Nuc)[2] + (1. - beta)*SFcont*ext)*SCALE, color="gray", ls="dashed", lw=0.25, label="Ice Correction", zorder = 0)
        

        # Plot Lines
        lin = SetupFit.Lines(wav, ps[lines_indx])
        ax.plot(wav,  ext*lin*SCALE, color="tab:cyan", ls="solid", lw=0.25, label="Lines")
        
        
        # Plot PAHs




        pah =  SetupFit.PAH(wav, ps[pah_indx])[0]
        ax.plot(wav,  ext*pah*SCALE, color="green", ls="solid", lw=0.25, label="PAH's")
        
        # Plot Full Model
        total =  ext*SetupFit.Lines(wav, ps[lines_indx]) + ext*SetupFit.PAH(wav, ps[pah_indx])[0] + cont
        ax.plot(wav, total*SCALE, color="red", ls="solid", lw=0.25, label="Full Model")
        
        ax.set_ylabel("Flux (Jy)")
        ax.set_xlabel("Rest Wavelength ($\mu m$)")
        
        
       # # Optical Depth Plot
       # ax2=ax.twinx()
        
        #ax2.plot(wav, ext, color="black", ls="dashed", lw=0.25, label='Extinction')
       # ax2.set_ylabel("Extinction")
      #  ax2.set_ylim(0.0, 1.0)
        #ax2.invert_yaxis()
        
        
        
        fig.legend(frameon=True, prop={'size': 5}, loc='center left', bbox_to_anchor=(0.9, 0.5))
        fig.savefig(resultsdir+ObjName+"_Plot.pdf")
        plt.close()
        
        #Save Components to file as scipy interpolate objects
        np.savetxt(resultsdir+ObjNames[k]+"_FullContinuum.txt", np.transpose([wav, SetupFit.Ext(wav, ps[ext_indx])*cont*SCALE]))
        np.savetxt(resultsdir+ObjNames[k]+"_IceCorrectedContinuum.txt", np.transpose([wav, SetupFit.Ext(wav, ps[ext_indx])*cont*SCALE]))
        np.savetxt(resultsdir+ObjNames[k]+"_UnObscuredContinuum.txt", np.transpose([wav, cont*SCALE]))
        np.savetxt(resultsdir+ObjNames[k]+"Lines.txt", np.transpose([wav,np.array(lin)*SCALE]))
        np.savetxt(resultsdir+ObjNames[k]+"PAHs.txt", np.transpose([wav, np.array(pah)*SCALE]))
        np.savetxt(resultsdir+ObjNames[k]+"FullModel.txt", np.transpose([wav,np.array(total)*SCALE]))
        np.savetxt(resultsdir+ObjNames[k]+"Nucleus.txt", np.transpose([wav, ps[cont_indx][0]*(beta*NUCcont)*SCALE]))




        
        ################################## LINE PROPERTIES #######################################################
        # Output csv
        output = pd.DataFrame(columns=['Name', 'Rest Wavelength (micron)','Strength (10^-17 W/m^2)', 'S_err+','S_err-', 'Continuum (10^-17 W/m^2/um)', 'C_err+','C_err-','Eqw (micron)', 'E_err+','E_err-'])
        
        
        
        #Find Continuum samples
        cont_samps=[]
        sil_samps=[]
        sil_Psi_samps=[]
        ice_samps=[]
        print("Extracting Continuum Samples...")#
        wav = np.linspace(5.2, 14.2, 1000)
        wav_forIntegral = np.linspace(min(lam), 15.0, 1000)
        nu = 2.9979246e14/wav
        eqws_samples = np.empty((5, len(all_params[0])))
        str_samples =  np.empty((5, len(all_params[0])))
        
        SF_eqws_samples = np.empty((5, len(all_params[0])))
        nuc_frac_samples=[]
        pahtoSF_ratio=[]
        for i in tqdm(range(len(all_params[0]))):
            ps = all_params[:,i]
           
            ps_Nuc = Nuc_all_params[:,i]

           
            SFcont =   SetupFit.contTemp(wav, ps[cont_indx])
            NUCcont = SetupFit.NUCTemp(wav, ps_Nuc)[2]
            beta = np.exp(ps[cont_indx][1])
            beta = ps[cont_indx][1]
            cont =  ps[cont_indx][0]*(beta*NUCcont + (1. - beta)*SFcont*ext)*SCALE
            cont_SF = ps[cont_indx][0]*( (1. - beta)*SFcont*ext)*SCALE
           
            # Ext parameters
            ext = SetupFit.Ext(wav, ps[ext_indx])
            
            PAHs = SetupFit.PAH(wav, ps[pah_indx])[1]*SCALE
            #6.2 EW samples
            PAH = PAHs[2]
           # eqws_samples.append(quad(SetupFit.Integrand, 5., 14., args=(ext*cont, PAH, wav)))
            eqws_samples[0, i] = np.trapz(y = (ext*PAH)/(cont), x=wav)
            str_samples[0, i] = np.trapz(y = (ext*PAH), x=nu)
            SF_eqws_samples[0, i] = np.trapz(y = (ext*PAH)/(cont_SF), x=wav)



            #7.7 EW samples
            PAH = PAHs[4] + PAHs[5] + PAHs[6]
           # eqws_samples.append(quad(SetupFit.Integrand, 5., 14., args=(ext*cont, PAH, wav)))
            eqws_samples[1, i] = np.trapz(y = (ext*PAH)/(cont), x=wav)
            str_samples[1, i] = np.trapz(y = (ext*PAH), x=nu)
            SF_eqws_samples[1, i] = np.trapz(y = (ext*PAH)/(cont_SF), x=wav)

            PAH = PAHs[8]
           # eqws_samples.append(quad(SetupFit.Integrand, 5., 14., args=(ext*cont, PAH, wav)))
            eqws_samples[2, i] = np.trapz(y = (ext*PAH)/(cont), x=wav)
            str_samples[2, i] = np.trapz(y = (ext*PAH), x=nu)
            SF_eqws_samples[2, i] = np.trapz(y = (ext*PAH)/(cont_SF), x=wav)

            #11.3 EW samples
            PAH= PAHs[10] + PAHs[11]
           # eqws_samples.append(quad(SetupFit.Integrand, 5., 14., args=(ext*cont, PAH, wav)))
            eqws_samples[3, i] = np.trapz(y = (ext*PAH)/(cont), x=wav)
            str_samples[3, i] = np.trapz(y = (ext*PAH), x=nu)
            SF_eqws_samples[3, i] = np.trapz(y = (ext*PAH)/(cont_SF), x=wav)

            #12.7 EW samples
            PAH = PAHs[13] + PAHs[14]
           # eqws_samples.append(quad(SetupFit.Integrand, 5., 14., args=(ext*cont, PAH, wav)))
            eqws_samples[4, i] = np.trapz(y = (ext*PAH)/(cont), x=wav)
            str_samples[4, i] = np.trapz(y = (ext*PAH), x=nu)
            SF_eqws_samples[4, i] = np.trapz(y = (ext*PAH)/(cont_SF), x=wav)


            PAHs = SetupFit.PAH(wav, ps[pah_indx])[0]*SCALE
            area_cont = np.trapz(cont_SF, wav)
            area_pah = np.trapz(ext*PAHs, wav)
            pahtoSF_ratio.append(area_pah/area_cont)
           

            nuc_frac_samples.append(beta)



            # Sil depth
            NUCcont = SetupFit.NUCTemp(wav, ps_Nuc)[1]
            unobscured_continuum =ps[cont_indx][0]*(beta*NUCcont + (1. - beta)*SFcont)*SCALE
            sil_samps.append(-1.0*np.log(np.interp(9.8, wav, unobscured_continuum*SCALE)/np.interp(9.8, wav, cont*SCALE)))
           # sil_Psi_samps.append(-1.0*Psi_peak)
           # ice_samps.append(SetupFit.Ext(wav, ps[ext_indx])[4])
        
            cont_samps.append(interp1d(wav, cont, fill_value = "extrapolate"))
            
            
            
            

            
        
        

        #Lines
        line_params = all_params[lines_indx]
        for i in range(setup.Nlines):
            a = int(3.0*i) # Index for amps
            c =  int(3.0*i + 1.0) # Index for centres
            w =  int(3.0*i + 2.0) # Index for widths
            
            lam =line_params[c]
            strength =SCALE*2.9979246e14 * 0.5 * line_params[a]*line_params[w]*1.0e-9*np.sqrt(np.pi/np.log(2))/(line_params[c]) # 10^-17 W/m^2

                
            strr = np.percentile(strength, [16,50,84])
            str_med = strr[1]
            str_l = strr[1]-strr[0]
            str_u = strr[2]-strr[1]
            # Upper limits
           # if (str_med/str_l < 3.0):
             #   str_med = "<" + str(np.max(strength))
             #   str_l = "-"
             #   str_u = "-"
                

            continuum = np.empty(len(lam))
            for j in range(len(lam)):
                continuum[j] =  cont_samps[j](lam[j])*1.0e-9*2.9979246e14/(lam[j]**2) # 10^-17 W/m^2/um#
            
            cont = np.percentile(continuum, [16,50,84])
            cont_med = cont[1]
            cont_l = cont[1]-cont[0]
            cont_u = cont[2]-cont[1]
            # Upper limits
            if (cont_med/cont_l < 3.0):
                cont_med = "<" + str(np.max(continuum))
                cont_l = "-"
                cont_u = "-"
                
                
            eqw = strength/continuum # micron
            Eqw = np.percentile(eqw, [16,50,84])
            Eqw_med = Eqw[1]
            Eqw_l = Eqw[1]-Eqw[0]
            Eqw_u = Eqw[2]-Eqw[1]
            # Upper limits
            if (Eqw_med/Eqw_l < 3.0):
                Eqw_med = "<" + str(np.max(eqw))
                Eqw_l = "-"
                Eqw_u = "-"
                
                
            output = output.append({ 'Name': setup.linenames[i], 'Rest Wavelength (micron)': setup.linecents[i],'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}, ignore_index=True)

        
        
        #PAH's
       # eqws_samples=[]
       # str_samples = []
        cont_samples=[]
        pah_params = all_params[pah_indx]
        for i in range(setup.Npah):
            a = int(3.0*i) # Index for amps
            c =  int(3.0*i + 1.0) # Index for centres
            w =  int(3.0*i + 2.0) # Index for widths
            
            lam = pah_params[c]
            strength = SCALE*2.9979246e14 * 0.5 * pah_params[a]*pah_params[w]*1.0e-9*np.pi/(pah_params[c]) # 10^-17 W/m^2

                
            strr = np.percentile(strength, [16,50,84])
            str_med = strr[1]
            str_l = strr[1]-strr[0]
            str_u = strr[2]-strr[1]
            # Upper limits
         #   if (str_med/str_l < 3.0):
              #  str_med = "<" + str(np.max(strength))
              #  str_l = "-"
              #  str_u = "-"
            

            continuum = np.empty(len(lam))
            for j in range(len(lam)):
                continuum[j] =   cont_samps[j](lam[j])*1.0e-9*2.9979246e14/(lam[j]**2) # 10^-17 W/m^2/um#
            
            
            
            cont = np.percentile(continuum, [16,50,84])
            cont_med = cont[1]
            cont_l = cont[1]-cont[0]
            cont_u = cont[2]-cont[1]
            # Upper limits
            #if (cont_med/cont_l < 3.0):
           #     cont_med = "<" + str(np.max(continuum))
            #    cont_l = "-"
             #   cont_u = "-"
            
            eqw =  strength/continuum # micron
            Eqw = np.percentile(eqw, [16,50,84])
            Eqw_med = Eqw[1]
            Eqw_l = Eqw[1]-Eqw[0]
            Eqw_u = Eqw[2]-Eqw[1]
            # Upper limits
           # if (Eqw_med/Eqw_l < 3.0):
                #Eqw_med = "<" + str(np.max(eqw))
                #Eqw_l = "-"
                #Eqw_u = "-"
            
            output = output.append({ 'Name': 'PAH', 'Rest Wavelength (micron)': setup.pahcents[i],'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}, ignore_index=True)


            # PAH complex's
            if (setup.pahcents[i] == 6.22):
                output = output.append({ 'Name': 'PAH 6.2 Complex', 'Rest Wavelength (micron)': 6.2,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}, ignore_index=True)
             #   eqws_samples.append(eqw)
               # str_samples.append(strength)
                cont_samples.append(continuum)
                
            if (setup.pahcents[i] == 8.61):
                output = output.append({ 'Name': 'PAH 8.6 Complex', 'Rest Wavelength (micron)': 8.61,'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}, ignore_index=True)
              #  eqws_samples.append(eqw)
               # str_samples.append(strength)
                cont_samples.append(continuum)

            if  (setup.pahcents[i] == 11.33 or setup.pahcents[i] == 12.69):
                lam = np.full(len(pah_params[c]), np.round(setup.pahcents[i], 1))

                strength +=  SCALE*2.9979246e14 * 0.5 * pah_params[a - 3]*pah_params[w - 3]*1.0e-9*np.pi/(pah_params[c - 3]) # 10^-17 W/m^2

                strr = np.percentile(strength, [16,50,84])
                str_med = strr[1]
                str_l = strr[1]-strr[0]
                str_u = strr[2]-strr[1]
                # Upper limits
              #  if (str_med/str_l < 3.0):
                  #  str_med = "<" + str(np.max(strength))
                   # str_l = "-"
                   # str_u = "-"
                
      
                continuum = np.empty(len(lam))
                for j in range(len(lam)):
                    continuum[j] =   cont_samps[j](lam[j])*1.0e-9*2.9979246e14/(lam[j]**2) # 10^-17 W/m^2/um#
            
                cont = np.percentile(continuum, [16,50,84])
                cont_med = cont[1]
                cont_l = cont[1]-cont[0]
                cont_u = cont[2]-cont[1]
                # Upper limits
             #   if (cont_med/cont_l < 3.0):
                 #   cont_med = "<" + str(np.max(continuum))
                 #   cont_l = "-"
                 #   cont_u = "-"
                    
                    
                eqw = strength/continuum # micron
                Eqw = np.percentile(eqw, [16,50,84])
                Eqw_med = Eqw[1]
                Eqw_l = Eqw[1]-Eqw[0]
                Eqw_u = Eqw[2]-Eqw[1]
                # Upper limits
               # if (Eqw_med/Eqw_l < 3.0):
                   # Eqw_med = "<" + str(np.max(eqw))
                   # Eqw_l = "-"
                    #Eqw_u = "-"
              #  eqws_samples.append(eqw)
                #str_samples.append(strength)
                cont_samples.append(continuum)
                output = output.append({ 'Name': 'PAH' +str(lam[0]) +'Complex', 'Rest Wavelength (micron)': lam[0],'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}, ignore_index=True)
                
                
                
            if (setup.pahcents[i] == 7.85):
                lam = np.full(len(pah_params[c]), 7.7)

                strength +=  SCALE*2.9979246e14 * 0.5 * pah_params[a - 3]*pah_params[w - 3]*1.0e-9*np.pi/(pah_params[c - 3]) + SCALE*2.9979246e14 * 0.5 * pah_params[a - 6]*pah_params[w - 6]*1.0e-9*np.pi/(pah_params[c - 6])# 10^-17 W/m^2

                strr = np.percentile(strength, [16,50,84])
                str_med = strr[1]
                str_l = strr[1]-strr[0]
                str_u = strr[2]-strr[1]
                # Upper limits
              #  if (str_med/str_l < 3.0):
                  #  str_med = "<" + str(np.max(strength))
                   # str_l = "-"
                   # str_u = "-"
                
      
                continuum = np.empty(len(lam))
                for j in range(len(lam)):
                    continuum[j] =   cont_samps[j](lam[j])*1.0e-9*2.9979246e14/(lam[j]**2) # 10^-17 W/m^2/um#
            
                cont = np.percentile(continuum, [16,50,84])
                cont_med = cont[1]
                cont_l = cont[1]-cont[0]
                cont_u = cont[2]-cont[1]
                # Upper limits
             #   if (cont_med/cont_l < 3.0):
                 #   cont_med = "<" + str(np.max(continuum))
                 #   cont_l = "-"
                 #   cont_u = "-"
                    
                    
                eqw = strength/continuum # micron
                Eqw = np.percentile(eqw, [16,50,84])
                Eqw_med = Eqw[1]
                Eqw_l = Eqw[1]-Eqw[0]
                Eqw_u = Eqw[2]-Eqw[1]
                # Upper limits
               # if (Eqw_med/Eqw_l < 3.0):
                   # Eqw_med = "<" + str(np.max(eqw))
                   # Eqw_l = "-"
                    #Eqw_u = "-"
                #eqws_samples.append(eqw)
              #  str_samples.append(strength)
                cont_samples.append(continuum)
                
                
                
                output = output.append({ 'Name': 'PAH' +str(lam[0]) +'Complex', 'Rest Wavelength (micron)': lam[0],'Strength (10^-17 W/m^2)': str_med,'S_err+': str_u, 'S_err-': str_l, 'Continuum (10^-17 W/m^2/um)': cont_med, 'C_err+': cont_u,'C_err-': cont_l,'Eqw (micron)': Eqw_med, 'E_err+': Eqw_u, 'E_err-': Eqw_l}, ignore_index=True)
                
                
        # 12.7/11.3 PAH EQW ratio
        ratio = eqws_samples[4]/eqws_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
       # if (ratio_med/ratio_l < 3.0):
          #  ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'EW(12.7)/EW(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)
        
        print(np.mean(eqws_samples[4]))
        # 6.2/11.3 PAH EQW ratio
        ratio = eqws_samples[0]/eqws_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
       # if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'EW(6.2)/EW(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)':ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)

        # 7.7/11.3 PAH EQW ratio
        ratio = eqws_samples[1]/eqws_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
       # if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'EW(7.7)/EW(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)':ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)


        # 7.7/8.6 PAH EQW ratio
        ratio = eqws_samples[1]/eqws_samples[2]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
       # if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'EW(7.7)/EW(8.6)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)':ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)


        # 8.6/11.3 PAH EQW ratio
        ratio = eqws_samples[2]/eqws_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
       # if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'EW(8.6)/EW(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)':ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)







        # 12.7/11.3 cont ratio
        ratio = cont_samples[4]/cont_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
     #   if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
            #ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'cont(12.7)/cont(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)
        
        # 6.2/11.3 cont ratio
        ratio = cont_samples[0]/cont_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
      #  if (ratio_med/ratio_l < 3.0):
          #  ratio_med = "<" + str(np.max(ratio))
          #  ratio_l = "-"
          #  ratio_u = "-"
        output = output.append({ 'Name': 'cont(6.2)/cont(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)



        # 7.7/11.3 cont ratio
        ratio = cont_samples[1]/cont_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
      #  if (ratio_med/ratio_l < 3.0):
          #  ratio_med = "<" + str(np.max(ratio))
          #  ratio_l = "-"
          #  ratio_u = "-"
        output = output.append({ 'Name': 'cont(7.7)/cont(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)


        # 7.7/8.6 cont ratio
        ratio = cont_samples[1]/cont_samples[2]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
      #  if (ratio_med/ratio_l < 3.0):
          #  ratio_med = "<" + str(np.max(ratio))
          #  ratio_l = "-"
          #  ratio_u = "-"
        output = output.append({ 'Name': 'cont(7.7)/cont(8.6)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)

        # 8.6/11.3 cont ratio
        ratio = cont_samples[2]/cont_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
      #  if (ratio_med/ratio_l < 3.0):
          #  ratio_med = "<" + str(np.max(ratio))
          #  ratio_l = "-"
          #  ratio_u = "-"
        output = output.append({ 'Name': 'cont(8.6)/cont(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)








        
        # 12.7/11.3 PAH ratio
        ratio = str_samples[4]/str_samples[3] #eqws_samples[4]/eqws_samples[3] * (cont_samples[4]/cont_samples[3])
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
      #  if (ratio_med/ratio_l < 3.0):
            #ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
          #  ratio_u = "-"
        output = output.append({ 'Name': 'PAH(12.7)/PAH(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)
        
        # 6.2/11.3 PAH ratio
        ratio = str_samples[0]/str_samples[3] #eqws_samples[0]/eqws_samples[3] * (cont_samples[0]/cont_samples[3])
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
      #  if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'PAH(6.2)/PAH(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)
        
        
        # 7.7/11.3 PAH ratio
        ratio = str_samples[1]/str_samples[3] #eqws_samples[1]/eqws_samples[3] * (cont_samples[1]/cont_samples[3])
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
      #  if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'PAH(7.7)/PAH(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)
        
        
        # 7.7/8.6 PAH ratio
        ratio = str_samples[1]/str_samples[2] #eqws_samples[1]/eqws_samples[2] * (cont_samples[1]/cont_samples[2])
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
      #  if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'PAH(7.7)/PAH(8.6)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)
        
        # 8.6/11.3 PAH ratio
        ratio = str_samples[2]/str_samples[3] #eqws_samples[2]/eqws_samples[3] * (cont_samples[2]/cont_samples[3])
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
      #  if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'PAH(8.6)/PAH(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)
        



        # 12.7/11.3 PAH EQW ratio
        ratio = SF_eqws_samples[4]/SF_eqws_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
       # if (ratio_med/ratio_l < 3.0):
          #  ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'SF EW(12.7)/EW(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)
        
        print(np.mean(SF_eqws_samples[4]))
        # 6.2/11.3 PAH EQW ratio
        ratio = SF_eqws_samples[0]/SF_eqws_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
       # if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'SF EW(6.2)/EW(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)':ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)

        # 7.7/11.3 PAH EQW ratio
        ratio = SF_eqws_samples[1]/SF_eqws_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
       # if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'SF EW(7.7)/EW(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)':ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)


        # 7.7/8.6 PAH EQW ratio
        ratio = SF_eqws_samples[1]/SF_eqws_samples[2]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
       # if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'SF EW(7.7)/EW(8.6)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)':ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)


        # 8.6/11.3 PAH EQW ratio
        ratio = SF_eqws_samples[2]/SF_eqws_samples[3]
        Ratio =np.percentile(ratio, [16,50,84])
        ratio_med = Ratio[1]
        ratio_l = Ratio[1]-Ratio[0]
        ratio_u = Ratio[2]-Ratio[1]
        # Upper limits
       # if (ratio_med/ratio_l < 3.0):
           # ratio_med = "<" + str(np.max(ratio))
           # ratio_l = "-"
           # ratio_u = "-"
        output = output.append({ 'Name': 'SF EW(8.6)/EW(11.3)', 'Rest Wavelength (micron)': 0.0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)':ratio_med, 'E_err+': ratio_l, 'E_err-': ratio_u}, ignore_index=True)










        # Silicates
        # 10 micron

        Sil =np.percentile(sil_samps, [16,50,84])
        sil_med = Sil[1]
        sil_l = Sil[1]-Sil[0]
        sil_u = Sil[2]-Sil[1]
        output = output.append({ 'Name': 'Sil Depth 10', 'Rest Wavelength (micron)': 9.7,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': sil_med, 'E_err+': sil_l, 'E_err-': sil_u}, ignore_index=True)
        
        
        Sil =np.percentile(nuc_frac_samples, [16,50,84])
        sil_med = Sil[1]
        sil_l = Sil[1]-Sil[0]
        sil_u = Sil[2]-Sil[1]
        output = output.append({ 'Name': 'Nuclear Fraction', 'Rest Wavelength (micron)': 0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': sil_med, 'E_err+': sil_l, 'E_err-': sil_u}, ignore_index=True)
        
        
                    
                    
        Sil =np.percentile(pahtoSF_ratio, [16,50,84])
        sil_med = Sil[1]
        sil_l = Sil[1]-Sil[0]
        sil_u = Sil[2]-Sil[1]
        output = output.append({ 'Name': 'PAHtoSF Ratio', 'Rest Wavelength (micron)': 0,'Strength (10^-17 W/m^2)': 0.0,'S_err+': 0.0, 'S_err-': 0.0, 'Continuum (10^-17 W/m^2/um)': 0.0, 'C_err+': 0.0,'C_err-': 0.0,'Eqw (micron)': sil_med, 'E_err+': sil_l, 'E_err-': sil_u}, ignore_index=True)
        
        
        #print(output )
        output.to_csv(resultsdir+ObjNames[k]+"Output.csv",  index=False)
        
        print("")
        print("")
        print("Line and PAH properties")
        Table.pprint_all(Table.from_pandas(output))
        
        
        
        
    
    


############ Run Fit #################################################################################################################################################################################################################################

from os import listdir
from os.path import isfile, join
if __name__ == '__main__':


    # Set data directory
    datadir="./Data/"



    # Example of fitting multiple apertures simultaneously
    apertures = [ 2.5,4.5,6.5, 8.5 ] # List of aperture sizes where file = 'name_aperturesize.txt'
    objs = ["ESO320-G030"] # Source name
    zs = [0.01078] # Source redshift


    # Example of fitting a single spectrum

   # objs=['Arp 220']
  #  zs=np.zeros(len(objs)) # Data is already rest frame




 #################################################################################################################################################################################################################################


    
    for i in range(len(objs)):
        obj = objs[i]
        z = zs[i]
        print(obj, z)

        try:
            RunFit(obj, z, datadir, apertures)
        except:
            try:
                RunFit(obj, z, datadir, [2.5, 4.5, 6.5])
            except:
                try:
                    RunFit(obj, z, datadir, [2.5, 4.5])
                except:
                    try:
                        RunFit(obj, z, datadir, [2.5])
                    except:
                        try:
                            RunFit(obj, z, datadir, [""])
                        except:
                            print("Skipping Object - no files found")
    
    
    
    
    
    
    
    
    
    
    
    
