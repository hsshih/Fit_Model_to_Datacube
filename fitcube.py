#! /Library/Frameworks/Python.framework/Versions/2.7/bin/python

### Use MPFIT to fit the NII + Halpha, SII, OIII, and Hbeta lines in data cube, produce line-ratio maps and map spaxels to BPT diagram. 

import numpy as np
import matplotlib.pyplot as plt
import pyfits
import scipy.optimize as optimize
import copy
import pandas
from mpfit import mpfit

import peakutils
from peakutils.plot import plot as pplot



def gauss(x, p):
    #p = [Peak, mean, sigma]
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss_area(A,sig):
    #A = peak value
    #sig = sigma

    area = A * np.sqrt(2.*np.pi) * sig
    return area



def gauss_n2ha(x,p):
    #### p = [const, peak_ha, z_ha, sigma_ha, peak_n2_1, z_n2]
    # Constrains all sigmas to be the same
    halpha_rest = 6562.8
    n2_1_rest = 6548.0
    n2_2_rest = 6583.3
    n2_21_ratio = 2.94442   # n2_6583 is ~2.9444 times stronger than n2_6548
    
    n2ha = p[0] + gauss(x,[p[1],(1.+p[2])*halpha_rest,p[3]]) + gauss(x,[p[4],(1.+p[5])*n2_1_rest,p[3]]) \
           + gauss(x,[p[4]*n2_21_ratio,(1.+p[5])*n2_2_rest,p[3]])
    return n2ha


def gauss_n2ha_2(x,p):
    #### p = [const, peak_ha, z_ha, sigma_ha, peak_n2_1, z_n2, sigma_n2]
    halpha_rest = 6562.8
    n2_1_rest = 6548.0
    n2_2_rest = 6583.3
    n2_21_ratio = 2.94442   # n2_6583 is ~2.9444 times stronger than n2_6548
    
    n2ha = p[0] + gauss(x,[p[1],(1.+p[2])*halpha_rest,p[3]]) + gauss(x,[p[4],(1.+p[5])*n2_1_rest,p[6]]) \
        + gauss(x,[p[4]*n2_21_ratio,(1.+p[5])*n2_2_rest,p[6]])
    return n2ha


def gauss_s2(x,p):
    #### p = [const, peak_s2_1, peak_s2_2, z_s2, sigma_s2]
    s2_1_rest = 6716.3
    s2_2_rest = 6730.7
    
    s2 = p[0] + gauss(x,[p[1],(1.+p[3])*s2_1_rest,p[4]]) + gauss(x,[p[2],(1.+p[3])*s2_2_rest,p[4]])
    return s2


def gauss_o3(x,p):
    #### p = [const, peak_o3, z_o3, sigma_o3]
    o3_rest = 5006.9
    
    o3 = p[0] + gauss(x,[p[1],(1.+p[2])*o3_rest,p[3]])
    return o3

def gauss_hbeta(x,p):
    #### p = [const, peak_hbeta, z_hbeta, sigma_hbeta]
    hbeta_rest = 4861.3
    
    hbeta = p[0] + gauss(x,[p[1],(1.+p[2])*hbeta_rest,p[3]])
    return hbeta


def myfunctgauss_hbeta(p, fjac=None, x=None, y=None, err=None):
    # Fits H-beta gaussian profile
    #### p = [const, peak, wave, sigma]
    model = gauss_hbeta(x,p)
    vals = (y-model)/err
    
    status = 0
    return [status, vals]


def myfunctgauss_o3(p, fjac=None, x=None, y=None, err=None):
    # Fits [OIII] 5007 gaussian profile
    #### p = [const, peak, wave, sigma]
    model = gauss_o3(x,p)
    vals = (y-model)/err
    
    status = 0
    return [status, vals]


def myfunctgauss_n2ha(p, fjac=None, x=None, y=None, err=None):
    # Fits the N[II] + H alpha profiles
    #### p = [const, peak_ha, z_ha, sigma_ha, peak_n2_1, z_n2, sigma_n2]
    
    model = gauss_n2ha(x,p)
    vals = (y-model)/err
    
    status = 0
    return [status, vals]

def myfunctgauss_s2(p, fjac=None, x=None, y=None, err=None):
    # Fits the S[II] 6716,6730
    #### p = [const, peak_s2_1, peak_s2_2, z_s2, sigma_s2]
    
    model = gauss_s2(x,p)
    vals = (y-model)/err
    
    status = 0
    return [status, vals]


def myfunctgauss1(p, fjac=None, x=None, y=None, err=None):
    # Fits single gaussian profile
    # p = [const, peak, wave, sigma]
    model = p[0] + gauss(x,p[1:4])
    vals = (y-model)/err
    
    status = 0
    return [status, vals]


def mpfit_all(casenum = 0):
    """
    Fit the emission lines using mpfit, and use mkfits_obj to make the important fits files
    Manually check the data cube to determine fit regions (startpix, npix) and initial guesses (p_o3, p_hb, p_n2ha, p_s2) for each line
    p_o3 = [const_o3, peak_o3, z_o3, sigma_o3]
    p_hb= [const_hb, peak_hb, z_hb, sigma_hb]
    p_n2ha= [const_n2ha, peak_ha, z_ha, sigma_ha, peak_n2_6548, z_n2, sigma_n2]
    p_s2= [const_s2, peak_s2_6716, peak_s2_6731, z_s2, sigma_s2]
    """

    if casenum == 0:
        
        ### At pixel [18,22] H-alpha/H-beta ~ 5.47, which means E(B-V) ~ 0.626 and Av ~ 1.9
        ### At pixel [16,25] H-alpha/H-beta ~ 4.96
        ### At pixel [18,22] [OIII]/H-beta ~ 3.08
        ### At pixel [16,25] [OIII]/H-beta ~ 2.93
        
        filename = 'SDSSJ143642_20150615_3D.fits'
        objname = 'SDSSJ143642'
        
        f1 = fit_object(filename,objname)
        
        p_o3 = [0.005,0.005,0.0204,2.]
        p_hb = [0.005,0.003,0.0204,1.5]
        p_n2ha = [0.002,0.00218,0.0203,4.,0.0127,0.0203]
        p_s2 = [0.001,0.0015,0.0015,0.0203,1.5]
        
        df1_o3 = f1.fit_o3(p_o3,startpix=675, npix=60)
        df1_hb = f1.fit_hb(p_hb,startpix=520, npix=40)
        df1_n2ha = f1.fit_n2ha(p_n2ha,startpix=2360, npix=136)
        df1_s2 = f1.fit_s2(p_s2,startpix=2530, npix=130, limidx=[4], limiteds=[[1,1]], limits=[[0.,4.]])
        
        df_file = objname + '_fitresult.pkl'
        df_csv = objname + '_fitresult.csv'
        df1_join = pandas.concat([df1_o3, df1_hb, df1_n2ha, df1_s2], axis=1)
        df1_join.to_pickle(df_file)
        df1_join.to_csv(df_csv)
        
        ######## To read the dataframe file: df = pandas.read_pickle(df_file)

        mkfis_obj(df_file,objname,f1.fitsdim)
        mkcontmed(filename, objname)
        mkbpt(df_file,objname)

        #get_ulim(filename, df_file,startpix=520, npix=40, col=['Hbeta_ulim'], new_df_filename='new_df_file.pkl')



    if casenum == 1:
        
        
        filename = 'SDSSJ151505_20150411-12_3D.fits'
        objname = 'SDSSJ151505'
        
        f1 = fit_object(filename,objname)
        
        p_o3 = [0.002,0.001,0.01807,2.]
        p_hb = [0.002,0.0005,0.01750,2.]
        p_n2ha = [0.003,0.00218,0.01807,2.,0.001,0.01807]
        p_s2 = [0.002,0.0015,0.0015,0.01807,1.5]
        
        df1_o3 = f1.fit_o3(p_o3,startpix=660, npix=55)
        df1_hb = f1.fit_hb(p_hb,startpix=515, npix=40)
        df1_n2ha = f1.fit_n2ha(p_n2ha,startpix=2300, npix=200)
        df1_s2 = f1.fit_s2(p_s2,startpix=2500, npix=160, limidx=[4], limiteds=[[1,1]], limits=[[0.,4.]])
        
        df_file = objname + '_fitresult.pkl'
        df_csv = objname + '_fitresult.csv'
        df1_join = pandas.concat([df1_o3, df1_hb, df1_n2ha, df1_s2], axis=1)
        df1_join.to_pickle(df_file)
        df1_join.to_csv(df_csv)

        ######## To read the dataframe file: df = pandas.read_pickle(df_file)
        
        mkfis_obj(df_file,objname,f1.fitsdim)
        mkcontmed(filename, objname)
        mkbpt(df_file,objname)
        
        get_ulim(filename, df_file,startpix=520, npix=40, col=['Hbeta_ulim'], new_df_filename='new_df_file.pkl')


    if casenum == 2:
    
    
        filename = 'SDSSJ134640_20150411_3D.fits'
        objname = 'SDSSJ134640'
        
        f1 = fit_object(filename,objname)
        
        p_o3 = [0.01,0.04,0.029,2.]
        p_hb = [0.01,0.005,0.029,2.]
        p_n2ha = [0.01,0.01,0.029,2.4,0.02,0.029]
        p_s2 = [0.01,0.01,0.009,0.029,3.4]
        
        df1_o3 = f1.fit_o3(p_o3,startpix=725, npix=65)
        df1_hb = f1.fit_hb(p_hb,startpix=565, npix=60)
        df1_n2ha = f1.fit_n2ha(p_n2ha,startpix=2420, npix=150)
        df1_s2 = f1.fit_s2(p_s2,startpix=2630, npix=80, limidx=[4], limiteds=[[1,1]], limits=[[0.,4.]])
        
        df_file = objname + '_fitresult.pkl'
        df_csv = objname + '_fitresult.csv'
        df1_join = pandas.concat([df1_o3, df1_hb, df1_n2ha, df1_s2], axis=1)
        df1_join.to_pickle(df_file)
        df1_join.to_csv(df_csv)
        
        ######## To read the dataframe file: df = pandas.read_pickle(df_file)
        
        mkfis_obj(df_file,objname,f1.fitsdim)
        mkcontmed(filename, objname)
        mkbpt(df_file,objname)
        
        #get_ulim(filename, df_file,startpix=520, npix=40, col=['Hbeta_ulim'], new_df_filename='new_df_file.pkl')





def mkfis_obj(df_file,objname,fitsdim):

    df = pandas.read_pickle(df_file)
    
    fit_fha = np.array(gauss_area(df['peak_ha'],df['sig_han2']))
    fitz_ha = np.array(df['z_ha'])
    fitsig_ha = np.array(df['sig_han2'])
    fit_n2ha_ratio = np.array(gauss_area(df['peak_n2_1'],df['sig_han2']) * 3.94442 / gauss_area(df['peak_ha'],df['sig_han2']))
    fit_o3hb_ratio = np.array(gauss_area(df['peak_o3'],df['sig_o3']) / gauss_area(df['peak_hb'],df['sig_hb']))
    fit_s2_ratio = np.array(gauss_area(df['peak_s2_1'],df['sig_s2']) / gauss_area(df['peak_s2_2'],df['sig_s2']))
    fit_hahb_ratio = np.array(gauss_area(df['peak_ha'],df['sig_han2']) / gauss_area(df['peak_hb'],df['sig_hb']))

    
    fitarrays = [fit_fha, fitz_ha, fitsig_ha, fit_n2ha_ratio, fit_o3hb_ratio, fit_s2_ratio, fit_hahb_ratio]
    fit_filenames = ['fit_fha.fits', 'fitz_ha.fits','fitsig_ha.fits','_n2ha_ratio.fits','_o3hb_ratio.fits','_s2_6716_6730_ratio.fits','_hahb_ratio.fits']
    
    xc = df.index.get_level_values(0)
    yc = df.index.get_level_values(1)
    
    for i in range(len(fit_filenames)):
        try:
            mkfits(xc,yc,fitarrays[i],objname+fit_filenames[i],xsize=fitsdim[2],ysize=fitsdim[1])
        except IOError:
            print 'IOError: ', objname+fit_filenames[i], ' probably already exist'



def mkbpt(df_file,objname):
    
    df = pandas.read_pickle(df_file)
    
    # Using [NII]6584/Halpha, not both of the [NII] lines
    
    fit_n2ha_ratio = np.array(gauss_area(df['peak_n2_1'],df['sig_han2']) * 2.94442 / gauss_area(df['peak_ha'],df['sig_han2']))
    fit_o3hb_ratio = np.array(gauss_area(df['peak_o3'],df['sig_o3']) / gauss_area(df['peak_hb'],df['sig_hb']))
    fit_s2ha_ratio = np.array((gauss_area(df['peak_s2_1'],df['sig_s2']) + gauss_area(df['peak_s2_2'],df['sig_s2'])) / gauss_area(df['peak_ha'],df['sig_han2']))
    
    ln2ha = np.log10(fit_n2ha_ratio)
    lo3hb = np.log10(fit_o3hb_ratio)
    ls2ha = np.log10(fit_s2ha_ratio)

    n2ha_x = np.arange(25)*0.1 - 2.
    s2ha_x1 = np.arange(23)*0.1 - 2.
    s2ha_x2 = np.arange(23)*0.1 - 0.3

    o3hb_star_n2 = 0.61 / (n2ha_x[0:20] - 0.05) + 1.3
    o3hb_comp_n2 = 0.61 / (n2ha_x - 0.47) + 1.19

    o3hb_agn_s2 = 0.72 / (s2ha_x1 - 0.32) + 1.3
    o3hb_liner_s2 = 1.89 * (s2ha_x2) + 0.76

    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    plt.xlim((-1,0.5))
    plt.ylim((-2,2))

    ax[0].plot(n2ha_x[0:20],o3hb_star_n2,'b', n2ha_x,o3hb_comp_n2,'b', ln2ha, lo3hb, 'ro')
    ax[0].set_xlabel('[NII]/H-alpha')
    ax[0].set_ylabel('[OIII]/H-beta')
    ax[0].set_title(objname+' BPT plots')
    
    ax[1].plot(s2ha_x1,o3hb_agn_s2,'b', s2ha_x2,o3hb_liner_s2,'b', ls2ha, lo3hb, 'ro')
    ax[1].set_xlabel('[SII]/H-alpha')
    ax[1].set_ylabel('[OIII]/H-beta')
    
    pdfname = objname + '_bpt.pdf'
    fig.savefig(pdfname, format='pdf')
    


class fit_object:
    
    def __init__(self, filename, objname):
        self.filename=filename
        self.objname=objname
        self.fitsdim=self.getfitsdim()

    
    def getfitsdim(self):
        fits = pyfits.open(self.filename)
        fitsdata = fits[1].data
        fitsdim = fitsdata.shape
        fits.close()
        return fitsdim
    
    def fit_o3(self,p_o3,startpix=675, npix=60, limidx=None, limiteds=None, limits=None):
        df = mpfitgauss(self.filename, p_o3, self.fitsdim, function=myfunctgauss_o3, startpix=startpix, npix=npix,\
                         cols = ['const_o3','peak_o3','z_o3','sig_o3', \
                                 'consto3_err','po3_err','zo3_err','sigo3_err'],
                        limidx=limidx, limiteds=limiteds, limits=limits, pdfname=self.objname+'_O3_plot.pdf', line='o3')
        return df
    
    def fit_hb(self,p_hb,startpix=520, npix=40, limidx=None, limiteds=None, limits=None):
        df = mpfitgauss(self.filename, p_hb, self.fitsdim, function=myfunctgauss_hbeta, startpix=startpix, npix=npix,\
                        cols = ['const_hb','peak_hb','z_hb','sig_hb', \
                                'consthb_err','phb_err','zhb_err','sighb_err'],
                        limidx=limidx, limiteds=limiteds, limits=limits, pdfname=self.objname+'_Hbeta_plot.pdf', line='hb')
        return df

    def fit_n2ha(self,p_n2ha,startpix=2360, npix=136, limidx=None, limiteds=None, limits=None):
        df = mpfitgauss(self.filename, p_n2ha, self.fitsdim, function=myfunctgauss_n2ha, startpix=startpix, npix=npix,\
                        cols = ['const_n2ha', 'peak_ha', 'z_ha', 'sig_han2', 'peak_n2_1', 'z_n2', \
                                'constn2ha_err', 'pha_err','zha_err','sighan2_err','pn2_err','zn2_err'],
                        limidx=limidx, limiteds=limiteds, limits=limits, pdfname=self.objname+'_N2Halpha_plot.pdf', line='n2ha')
        return df

    def fit_s2(self,p_s2,startpix=2530, npix=130, limidx=None, limiteds=None, limits=None):
        df = mpfitgauss(self.filename, p_s2, self.fitsdim, function=myfunctgauss_s2, startpix=startpix, npix=npix,\
                        cols = ['const_s2','peak_s2_1','peak_s2_2','z_s2','sig_s2', \
                                'consts2_err','ps2_1_err','ps2_2_err','zs2_err','sigs2_err'],
                        limidx=limidx, limiteds=limiteds, limits=limits, pdfname=self.objname+'_S2_plot.pdf', line='s2')
        return df




class getparinfo:
    
    def __init__(self, p0):
        self.parinfo=self.copyinfo(p0)
                            
    def copyinfo(self,p0):
        
        parinfo=[]
        parbase={'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}
    
        for i in range(len(p0)):
            parinfo.append(copy.deepcopy(parbase))
                            
        for i in range(len(p0)):
            # All values should be above zero, maybe except for the constant, fix this later....
            parinfo[i]['value']=p0[i]
            parinfo[i]['limited'] = [1,0]
            parinfo[i]['limits'] = [0.,10.]
                           
        return parinfo
                            
    
    def put_limit(self,index,limited,limit):
        self.parinfo[index]['limited'] = limited
        self.parinfo[index]['limits'] = limit





def mpfitgauss(filename, p0, fitsdim, function=myfunctgauss1,startpix=700, npix=100,
               cols = ['Const','Peak','Wave','Sigma','Cerr','Perr','Werr','Serr'],
               limidx=None, limiteds=None, limits=None, pdfname='plot.pdf', line='o3'):
    
    fits = pyfits.open(filename)
    fitsdata = fits[1].data
    fitsvar = fits[2].data
    scihdr = fits[1].header

    crval3 = scihdr['CRVAL3']
    crpix3 = scihdr['CRPIX3']
    cd3_3 = scihdr['CD3_3']
    
    startwav = crval3 + ((startpix - crpix3)*cd3_3)
    wav = startwav + np.arange(npix)*cd3_3
    
    fitpar=[]
    xc=[]
    yc=[]


    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    fig, ax = plt.subplots(fitsdim[1],fitsdim[2])

    #Get estimated continuum
    for i in range(fitsdim[1]):
        for j in range(fitsdim[2]):
            
            spec = fitsdata[startpix:startpix+npix, i, j]
            ey = np.sqrt(fitsvar[startpix:startpix+npix, i, j])
            
            # If error is zero or nan, it will cause problems for the fit later, replacing with 1
            zeroidx = np.where(ey == 0)[0]
            ey[zeroidx] = 1.0
            ey[np.isnan(ey)] = 1.0
            
            fa = {'x': wav, 'y':spec, 'err':ey}
            
            
            est_cont = np.median(np.concatenate([spec[0:9],spec[npix-11:npix-1]]))
            indexes = peakutils.indexes(spec, thres=0.1, min_dist=10)
            maxspec = max(spec)
            #print(i,j,wav[indexes], spec[indexes])
            
            p0[0] = est_cont
            
            if line is 'o3' and len(indexes) > 0:
                p0[1] = maxspec
            
            if line is 'hb' and len(indexes) > 0:
                p0[1] = maxspec
            
            if line is 'n2ha' and len(indexes) > 2:
                
                # Sort peaks from highest to lowest, pick the three highest peak and then sort by wavelength
                peaks = spec[indexes]
                wavpks = wav[indexes]
                
                sortpks = sorted(peaks, reverse=True)
                sortidx = sorted(range(len(indexes)),key=lambda x:peaks[x], reverse=True)

                topwav = wavpks[sortidx[0:3]]
                toppks = peaks[sortidx[0:3]]
                
                sortidx2 = sorted(range(len(topwav)), key=lambda x: topwav[x])
                sortpks2 = toppks[sortidx2]
                
                p0[1] = sortpks2[1] - p0[0]
                p0[4] = sortpks2[0] - p0[0]
                
                initmodel = gauss_n2ha(wav,p0)
                
            if line is 's2' and len(indexes) > 1:
                
                peaks = spec[indexes]
                wavpks = wav[indexes]
                
                sortpks = sorted(peaks, reverse=True)
                sortidx = sorted(range(len(indexes)),key=lambda x:peaks[x], reverse=True)
                
                topwav = wavpks[sortidx[0:2]]
                toppks = peaks[sortidx[0:2]]
                
                sortidx2 = sorted(range(len(topwav)), key=lambda x: topwav[x])
                sortpks2 = toppks[sortidx2]

                p0[1] = sortpks2[0] - p0[0]
                p0[2] = sortpks2[1] - p0[0]
            
            gparinfo = getparinfo(p0)

            if limidx != None:
                for idx in range(len(limidx)):
                    gparinfo.put_limit(limidx[idx],limiteds[idx],limits[idx])
                
            parinfo = gparinfo.parinfo


            m = mpfit(function, p0, parinfo=parinfo,functkw=fa,quiet=1)
            
            if (m.status <= 0):
                print("status = ", m.status)
                print("error message = ", m.errmsg)
            
            if m.perror == None:
                m.perror=np.zeros(len(m.params))
            me = np.concatenate([m.params,m.perror])
            fitpar.append(me)
            xc.append(str(i))
            yc.append(str(j))

            if line is 'o3':
                fitmodel = gauss_o3(wav,m.params)

            if line is 'hb':
                fitmodel = gauss_hbeta(wav,m.params)

            if line is 'n2ha':
                fitmodel = gauss_n2ha(wav,m.params)

            if line is 's2':
                fitmodel = gauss_s2(wav,m.params)

            
            
            ax[fitsdim[1]-i-1,j].plot(wav,spec,'b', wav,fitmodel,'r', linewidth=0.2)
            ax[fitsdim[1]-i-1,j].axis('off')

    fig.savefig(pdfname, format='pdf', dpi=1000)

    fits.close()

    idx_arr = [np.array(xc),np.array(yc)]
    df = pandas.DataFrame(fitpar, index=idx_arr, columns=cols)
    return df



def mkcontmed(filename, objname, contcoord=[100,3000]):
    try:
        fits = pyfits.open(filename)
    except IOError:
        print 'Cannot open', filename
    else:
        fitsdata = fits[1].data
        
        contcube = fitsdata[contcoord[0]:contcoord[1],:,:]
        contmed = np.median(fitsdata,axis=0)
        
        try:
            hdu_c = pyfits.PrimaryHDU(contmed)
            hdu_c.writeto(objname+'contmed.fits')
        except IOError:
            print 'IOError: ', objname+'contmed.fits probably already exist'

        fits.close()


def mkfits(xc,yc,vals,fitsfilename,xsize=33,ysize=49):
    
    fitsval = np.zeros([ysize,xsize])
    
    for a in xrange(xsize*ysize):
        i,j = xc[a],yc[a]
        fitsval[i,j] = vals[a]

    hdu_f = pyfits.PrimaryHDU(fitsval)
    hdu_f.writeto(fitsfilename)



def get_ulim(filename, df_file, startpix=1, npix=2000, col=['ulim'], new_df_filename='new_df_file.pkl', width_pix=3.):
    
    # Get the 3 sigma detection limit of a line in a given wavelength range, with a given line-width in pixel
    
    fits = pyfits.open(filename)
    fitsvar = fits[2].data
    scihdr = fits[1].header
    
    fitsdim = fitsvar.shape

    df1 = pandas.read_pickle(df_file)
    siglim_3 = []
    xc=[]
    yc=[]

    for i in range(fitsdim[1]):
        for j in range(fitsdim[2]):
            ey = np.sqrt(fitsvar[startpix:startpix+npix, i, j])
            tmp_err = np.median(ey)
            siglim_3.append(3 * np.sqrt((tmp_err**2.) * width_pix))
            xc.append(str(i))
            yc.append(str(j))
    fits.close()

    idx_arr = [np.array(xc),np.array(yc)]
    df2 = pandas.DataFrame(siglim_3, index=idx_arr, columns=col)

    df_join = pandas.concat([df1, df2], axis=1)
    df_join.to_pickle(new_df_filename)
    df_join.to_csv('new_df_file.csv')

    mkfits(xc,yc,np.array(siglim_3),col[0]+'_map.fits')





if __name__ == '__main__':
    mpfit_all(casenum=2)



