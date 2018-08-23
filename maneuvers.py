import numpy as np
from coordinates import kep2cart
from models import atmosDensity, cubesat
from scipy import integrate
from constants import constants

class Maneuvers:

  def __init__(self,com):
    self._PERTURBATION_ATMDRAG_ = 0
    self._PERTURBATION_J2_ = 0
    self._PERTURBATION_SOLARPRESS_ = 0
    self._IMPULSIVE_FORCE_ = 0
    
    self.current_r = 0 
    self.current_v = 0
    self.current_t = 0

    self.rTrace = np.array([])
    self.vTrace = np.array([])
    self.tTrace = np.array([])

    self.segments = []

    self.com = com
    self.current_r, self.current_v = kep2cart(com)
    self.rTrace = self.current_r
    self.vTrace = self.current_v
    self.tTrace = self.current_t

  def addSegment(self,segment):
    self.segments.append(segment)

  def addPerturbation(self,perturbation):
    if perturbation == "atmosphere":
      self._PERTURBATION_ATMDRAG_ = 1
    if perturbation == "J2":
      self._PERTURBATION_J2_ = 1
    if perturbation == "solar_pressure":
      self._PERTURBATION_SOLARPRESS_ = 1

  def conwell(self, stateVector,t):
    r = stateVector[0:3]
    v = stateVector[3:6]
    z = np.linalg.norm(r)-constants.Re
    if t % (60*60*24) < 100:
      print("Day:"+str(t/60/60/24)+"\tHeight: "+str(z/1000)+" km")

    if(z > 100e3):
      p = 0
      if self._IMPULSIVE_FORCE_>0:
        #Impulsive maneuver
        p = p - self._IMPULSIVE_FORCE_*np.cross(self.current_r,self.current_v)/(np.linalg.norm(np.cross(self.current_r,self.current_v)))
        
        
      if self._PERTURBATION_ATMDRAG_:
        #Atmospheric Drag
        vrel = v - np.cross(constants.wE,r)
        Fd = -0.5*atmosDensity(z/1000)*np.linalg.norm(vrel)*vrel*(1/cubesat.BC)
        p = p + Fd
      if self._PERTURBATION_J2_:
        J2 = 0.00108263

        zz = r[2]

        rNorm = np.linalg.norm(r)
        bigTerm = np.array([1/rNorm*(5*zz**2/rNorm**2-1),
                            1/rNorm*(5*zz**2/rNorm**2-1),
                            1/rNorm*(5*zz**2/rNorm**2-3)])

        #J2 Gravity Gradient
        FJ2 = (3/2)*J2*constants.mu_E*constants.Re**2/rNorm**4*bigTerm*r
        p = p + FJ2

      if self._PERTURBATION_SOLARPRESS_:
        #Solar Radiation Pressure
        PSR = 4.56e-6
        #Absorbing Area
        As = np.pi*(cubesat.dimensions[0]/2)**2
        #Shadow function TODO
        nu = 1
        #Radiation Pressure
        FR = -nu*PSR*CR*As/cubesat.mass
        p = p + FR
    
      E = np.linalg.norm(v)**2/2-constants.mu_E/np.linalg.norm(r)
      if t % (60*60*24) < 100:
        print("Orbital Energy:"+str(E));
      #Differential Equations
      
      drdt = v
      dvdt = -constants.mu_E*(r/(np.linalg.norm(r)**3))+p

      return np.append(drdt,dvdt)
    else:
      return np.append(-r,-v)

  def propagate(self,time,r0=0):
    if r0==0:
        print("Propagating...from day ",self.current_t/60/60/24," to ",(self.current_t+time)/60/60/24)
    #Integrate
    discretisation = 60*2
    y0 = np.append(self.current_r,self.current_v)
    t = np.linspace(self.current_t,self.current_t+time,time/discretisation)
    y = integrate.odeint(self.conwell,y0,t)
    # Update traces
    self.tTrace = np.append(self.tTrace,t)
    self.rTrace = np.vstack((self.rTrace, y[:,0:3]))
    self.vTrace = np.vstack((self.vTrace, y[:,3:6]))
    # Update last values
    self.current_r = self.rTrace[-1,:]
    self.current_v = self.vTrace[-1,:]
    self.current_t = self.current_t+time
    if r0!=0 and self._IMPULSIVE_FORCE_<=0 and (np.linalg.norm(self.current_r)-constants.Re)>r0*1000+1:
        if time<=60:
            time = 2*60
        self.propagate(time,r0)
    elif r0!=0 and self._IMPULSIVE_FORCE_>0 and (np.linalg.norm(self.current_r)-constants.Re)<r0*1000-1:
        if time<=1:
            time = 1
        self.propagate(time,r0)

  def impulsive_maneuver(self,dva,f=0):
    if f==0:
        self.current_v = self.current_v+dva
        print("---------------------current v: ",self.current_v)
    else:
        self._IMPULSIVE_FORCE_ = dva

