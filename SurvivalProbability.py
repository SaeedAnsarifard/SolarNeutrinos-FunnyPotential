import numpy as np
from scipy.optimize import curve_fit


r_sun  = 6.96e10 #cm
r_sun  = 3.53e21 #Mev^-1
rsun_l = 6.96e8/1.5e11

#h_{bar}c : 1.97 10^{-11} Mev.cm
hbarc  = 1.97


def main(m,g,theta12):
    #g in 10^{-30}
    #m in R^{-1}
    
    #Neutrino production point weight function :http://www.sns.ias.edu/~jnb/
    load_phi = np.loadtxt('./bs2005agsopflux1.txt', unpack = True)
    phi      = {'pp' : load_phi[5,:],
                'Be7': load_phi[10,:],
                'pep': load_phi[11,:],
                'B8' : load_phi[6,:]}


    
    deltam12  = 1.
    enu       = np.logspace(-1,np.log10(20),300)
    
    load_phi1 = np.loadtxt('./nele_bs05op.txt')
    #n(X) in 10^{25} cm^{-3}
    xdata     = load_phi1[:,0]
    ydata     = 6e-2*10**load_phi1[:,1]
    popt, pcov= curve_fit(Nx, xdata, ydata)
   
    
    #V_eff in 10^{-18} MeV
    x0     = load_phi[0,:]
    gamma0 = np.linspace(0,180,100)
    fx     = np.zeros(x0.shape)
    alpha  = np.zeros((x0.shape[0],gamma0.shape[0]))
    v_eff  = np.zeros((enu.shape[0],x0.shape[0],gamma0.shape[0]))
    theta12m = np.zeros((enu.shape[0],x0.shape[0],gamma0.shape[0]))
    survival_probablity_polar = np.zeros((enu.shape[0],x0.shape[0],gamma0.shape[0]))
    for i in range(x0.shape[0]):
        fx[i] = IntegralXthetaprime(x0[i],m,popt)
        for j in range(gamma0.shape[0]):
            alpha[i,j] = Alpha(x0[i],gamma0[j])
            v_eff[:,i,j] = 1.26*Nx(x0[i],*popt) + g*67.5*np.cos((alpha[i,j]+gamma0[j])*np.pi/180)*fx[i]/enu
            theta12m[:,i,j] = ThetaM12(deltam12,enu,theta12,v_eff[:,i,j])
            survival_probablity_polar[:,i,j] = np.cos(np.pi*theta12/180.)**2 * np.cos(np.pi*theta12m[:,i,j]/180.)**2 + np.sin(np.pi*theta12/180.)**2 * np.sin(np.pi*theta12m[:,i,j]/180.)**2

    survival_probablity_x0 = np.trapz(survival_probablity_polar,gamma0,axis=2)/180.
    survival_probablity    = np.sum(survival_probablity_x0*phi['B8'],axis=1)
    return enu,survival_probablity
    
def Nx(x, a, b, c):
    return a * np.exp(-b * x**c) 

def Rprime(x,y,t):
    return np.sqrt(x**2 + y**2 + 2*x*y*t)

def IntegralXthetaprime(x,m,popt):
    #Integrating over x^prime to compute the F(X) in 2.7 \times 10^{-14}
    t  = np.logspace(-4,np.log10(2),200)-1
    y  = np.logspace(-4,0,200)
    ny = Nx(y,*popt)
    
    integral = np.zeros(len(y))
    for i in range(len(y)):
        f = Rprime(x,y[i],t)
        a = (x+t*y[i])/f
        b = np.exp(-m*f)
        c = m/f
        d = 1/f**2
        integral[i] = np.trapz(ny[i] * y[i]**2 * b * a * (c + d),t)
    return np.trapz(integral,y)

def Alpha(x0,gamma0):
    #The angle between neutrino production point and sun center seen by detector
    return 180.*np.arctan(x0*np.sin(np.pi*gamma0/180.)/((1/rsun_l) - x0*np.cos(np.pi*gamma0/180.)))/np.pi

def EllNu(deltam12,enu):
    #10^{-18} MeV , deltam12/7.5e-5
    return 75.0 * deltam12/(2*enu)

def ThetaM12(deltam12,enu,theta12,v_eff):
    ell_nu   = EllNu(deltam12,enu)
    thetam12 = 180*0.5*np.arctan(np.sin(np.pi*2*theta12/180.)/(np.cos(np.pi*2*theta12/180.)-(v_eff/ell_nu)))/np.pi
    thetam12[thetam12<0] = thetam12[thetam12<0] + 90.
    return thetam12
