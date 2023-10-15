import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats
from scipy.interpolate import interp1d

#from scipy.integrate import solve_ivp

from scipy.optimize import curve_fit


#plt.rcParams['text.usetex']=True
#plt.style.use('classic')

#plt.rcParams['font.size'] = 18

#load_phi  = np.loadtxt('./bs2005agsopflux1.txt', unpack = True)

def main(m):
    load_phi1 = np.loadtxt('./nele_bs05op.txt')

    r_sun  = 6.96e10 #cm
    r_sun  = 3.53e21 #Mev^-1
    rsun_l = 6.96e8/1.5e11
    
    
    #h_{bar}c : 1.97 10^{-11} Mev.cm
    hbarc = 1.97
    
    xtest = np.logspace(-3,0,100)
    xdata = load_phi1[:,0]
    ydata = 6e-2*10**load_phi1[:,1]

    popt, pcov = curve_fit(Nx, xdata, ydata)
    perr  = np.sqrt(np.diag(pcov))
    
    fnew  = Fnew(np.logspace(-4,0,300),m,popt)
    dfnew = DfnewDx(np.logspace(-4,0,300),m,popt)
    
def Nx(x, a, b, c):
    return a * np.exp(-b * x**c) 

def DnDx(x,a,b,c):
    return -b * c * x**(c-1) * a * np.exp(-b * x**c)

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


def DintegralXthetaprime(x,m,popt):
    #Integrating over x^prime to compute the dF(X)/dX in 2.7 \times 10^{-14}
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
        integral[i] = np.trapz(ny[i] * y[i]**2 * b * (-c*a*(c+d) + ((1/f)*(c+d)) - ((1/f)*a**2*(2*c + 3*d))) ,t)
                               
    return np.trapz(integral,y)

def Fnew(x,m,popt):
    # F(X) in 2.7 \times 10^{-14}
    vx = np.zeros(len(x))
    for i in range(len(x)):
        vx[i] = IntegralXthetaprime(x[i],m,popt) 
    return interp1d(x,vx)

def DfnewDx(x,m,popt):
    # dF(X)/dX in 2.7 \times 10^{-14}
    vx = np.zeros(len(x))
    for i in range(len(x)):
        vx[i] = DintegralXthetaprime(x[i],m,popt) 
    return interp1d(x,vx)
                               
def Veff(x,gamma,popt,enu,fnew,alpha,g):
    #V_eff in 10^{-18} MeV
    return  1.26*Nx(x,*popt) + g*67.5*np.cos((alpha+gamma)*np.pi/180)*fnew(x)/enu 
                               
def DveffDx(x,gamma,popt,enu,dfnew,alpha,g):
    #dV_eff/dX in 10^{-18} MeV
    return 1.26*DnDx(x,*popt) + g*67.5*np.cos((alpha+gamma)*np.pi/180)*dfnew(x)/enu
                               
def DveffDg(x,gamma,enu,fnew,alpha,g):
    #dV_eff/dgamma in 10^{-18} MeV
    return -g*67.5*np.sin((alpha+gamma)*np.pi/180)*fnew(x)/enu
    
def DveffDt(x0,gamma0,x,gamma,popt,enu,fnew,dfnew,alpha,g):
    a = DveffDx(x,gamma,popt,enu,dfnew,alpha,g) * DxDt(x0,gamma0,gamma)
    b = DveffDg(x,gamma,enu,fnew,alpha,g) * DgDt(x0,gamma0,x,gamma)
    return a + b

def TtoXSolver(x0,gamma0):
    #Finding the polar cordinate of a Neutrino passing through the Sun from the production point (X_0,gamma_0) to the surface of the Sun
    t = np.logspace(-5,np.log10(2),500)
    a = x0*np.sin(gamma0*np.pi/180)*(1-t*rsun_l)
    b = x0*np.cos(gamma0*np.pi/180) + t
    x = np.abs(np.sqrt(a**2 + b**2))
    gamma = 180.*np.arctan(a/(b+1e-10))/np.pi 
    gamma[gamma<0] = gamma[gamma<0] + 180
    gamma[x<1e-6] = 0.
    return t[(x>=1e-4)&(x<=1.)],x[(x>=1e-4)&(x<=1.)],gamma[(x>=1e-4)&(x<=1.)]

def Alpha(x0,gamma0):
    #The angle between neutrino production point and sun center seen by detector
    return 180.*np.arctan(x0*np.sin(np.pi*gamma0/180.)/((1/rsun_l) - x0*np.cos(np.pi*gamma0/180.)))/np.pi

def DxDt(x0,gamma0,gamma):
    #dX/dT
    return np.cos(np.pi*gamma/180.) - (x0 * np.sin(np.pi*gamma0/180.) * rsun_l * np.sin(np.pi*gamma/180.))

def DgDt(x0,gamma0,x,gamma):
    #dgamma/dT
    return (-np.sin(np.pi*gamma/180.) - (x0 * np.sin(np.pi*gamma0/180.) * rsun_l * np.cos(np.pi*gamma/180.)))/x

def EllNu(deltam12,enu):
    #10^{-18} MeV , deltam12/7.5e-5
    return 75.0 * deltam12/(2*enu)

def ThetaM12(deltam12,enu,theta12,v_eff):
    ell_nu   = EllNu(deltam12,enu)
    thetam12 = 180*0.5*np.arctan(np.sin(np.pi*2*theta12/180.)/(np.cos(np.pi*2*theta12/180.)-(v_eff/ell_nu)))/np.pi
    thetam12[thetam12<0] = thetam12[thetam12<0] + 90.
    return thetam12

def SurvivalProbablity(phi, enu, n_e, f_c, hbarc, param, ls):
    pel   = np.zeros((ls.shape[0],enu.shape[0]))
    psl   = np.zeros((ls.shape[0],enu.shape[0]))
    
    util= np.ones((n_e.shape[0],enu.shape[0]))
    ne  = np.reshape(n_e ,(n_e.shape[0],1))*util
    e   = np.reshape(enu ,(1,enu.shape[0]))*util

    ve  = 2 * np.sqrt(2) * f_c * ne * hbarc**3 * 1e-9 * e
    den = np.sqrt((param['M12'] * np.cos(2*(np.pi/180) * param['T12'])- ve)**2 + (param['M12'] * np.sin(2*(np.pi/180) * param['T12']))**2)
    nom = param['M12'] * np.cos(2*(np.pi/180) * param['T12']) - ve
    tm  = 0.5*np.arccos(nom/den)

    sin = np.sin((np.pi/180) * param['T12'])**2 * np.cos((np.pi/180) * param['T13'])**4
    cos = np.cos((np.pi/180) * param['T12'])**2 * np.cos((np.pi/180) * param['T13'])**4

    for j,l in enumerate(ls):
        ae1 = cos * np.cos(tm)**2  * np.cos(10*param['mum1']*l/(hbarc*2*e))**2
        ae2 = sin * np.sin(tm)**2  * np.cos(10*param['mum2']*l/(hbarc*2*e))**2
        ae3 = np.sin((np.pi/180)*param['T13'])**4 * np.cos(10*param['mum3']*l/(hbarc*2*e))**2

        pee = ae1 + ae2 + ae3
        pel[j]  = np.sum(np.reshape(phi,(n_e.shape[0],1))*pee,axis=0)

        as1 = np.cos((np.pi/180) * param['T13'])**2 * np.cos(tm)**2  * np.sin(10*param['mum1']*l/(hbarc*2*e))**2
        as2 = np.cos((np.pi/180) * param['T13'])**2 * np.sin(tm)**2  * np.sin(10*param['mum2']*l/(hbarc*2*e))**2
        as3 = np.sin((np.pi/180) * param['T13'])**2 * np.sin(10*param['mum3']*l/(hbarc*2*e))**2

        pes = as1 + as2 + as3
        psl[j]  = np.sum(np.reshape(phi,(n_e.shape[0],1))*pes,axis=0)

    return pel, psl

