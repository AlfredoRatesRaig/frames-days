# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate as integrate
from scipy.signal import argrelextrema
import matplotlib as mpl
import matplotlib.pyplot as  plt
import mplcursors
from mpl_toolkits.mplot3d import Axes3D

from models import atmosDensity
from maneuvers import Maneuvers
from coordinates import cart2kep

#==============================================================================
#==============================================================================

#Conwell's Method
# ----UNIVERSAL CONSTANTS----
# Universal Gravity
G = 6.67e-11
# Earth Mass
Me = 5.97e24
#Earth Radius
Re = 6378e3
#Earth Angular Speed
wE = np.array([0,0,7.2921159e-5])
#---------------------------
mu = G*Me
rp = Re+370e3
ra = Re+370e3
#rp = Re+350e3
#ra = Re+350e3
Omega = 340*np.pi/180
i = 65.1*np.pi/180
omega = 58*np.pi/180
M = 332*np.pi/180
#-------------------
e = (ra-rp)/(ra+rp)
a = (ra+rp)/2
h = (mu*a*(1-e**2))**0.5
T = 2*np.pi/mu**0.5*a**(3/2)
Aceleration = 0.8e-4
#--------------------
print("Starting propagations...")
maneuvers = Maneuvers([a,e,i,omega,Omega,M,0,mu])
maneuvers.addPerturbation("atmosphere")

maneuvers.propagate(60*60,350)
print("=============================================================")
print("First propagation ended...")
#maneuvers.impulsive_maneuver(maneuvers.current_v/np.linalg.norm(maneuvers.current_v)*5.7)
maneuvers.impulsive_maneuver(Aceleration,1)
maneuvers.propagate(60*5,370)
maneuvers.impulsive_maneuver(0,1)
print("=============================================================")
print("Second propagation ended...")
#maneuvers.propagate(60*60*(1.52/2))
#maneuvers.impulsive_maneuver(maneuvers.current_v/np.linalg.norm(maneuvers.current_v)*5.7)
maneuvers.propagate(60*60,350)
print("=============================================================")
print("Last propagation ended.")


#==============================================================================
#==============================================================================


#Cleanup
validPoints = np.all(abs(maneuvers.rTrace) > 1e-10,axis=1)
rTrace = maneuvers.rTrace[validPoints]
vTrace = maneuvers.vTrace[validPoints]
tTrace = maneuvers.tTrace[validPoints]

#Get Perigees and Apogees
perigees = []
apogees = []
for i in range(0,len(tTrace)):
    a,e = cart2kep(rTrace[i,:],vTrace[i,:])
    rp = (1-e)*a
    ra = (1+e)*a
    perigees.append(rp)
    apogees.append(ra)
perigees = np.asarray(perigees)
apogees = np.asarray(apogees)


#==============================================================================
#==============================================================================


#PLOTTING
mpl.rcParams['toolbar'] = 'None'

plt.figure(figsize=(8,2))
plt.plot(tTrace[:-2:1000]/60/60/24,(apogees[:-2:1000]-Re)/1e3);
plt.plot(tTrace[:-2:1000]/60/60/24,(perigees[:-2:1000]-Re)/1e3);
plt.grid();
plt.title("Caída de satélite con impulso")
plt.ylabel("Altitude [km]")
plt.xlabel("time [days]")
mplcursors.cursor(hover=True);


#==============================================================================
#==============================================================================


a = (350e3+Re+370e3+Re)/2
T = 2*np.pi*(a**3/mu)**0.5
print(T/60/60)
A = np.array([1, 2, 3])
B = np.array([3, 2, 1])
C = A*B
print(C)


#==============================================================================
#==============================================================================


Z = np.array([])
X3 = np.array([])
Y3 = np.array([])
Z3 = np.array([])
for i in range(len(maneuvers.rTrace)):
    X3 = np.append(X3,maneuvers.rTrace[i][0])
    Y3 = np.append(Y3,maneuvers.rTrace[i][1])
    Z3 = np.append(Z3,maneuvers.rTrace[i][2])
    Z = np.append(Z,np.linalg.norm(maneuvers.rTrace[i])-Re)
mpl.rcParams['toolbar'] = 'None'

plt.figure(figsize=(8,2))
plt.plot(maneuvers.tTrace/60/60/24,(Z/1e3));
plt.grid();
plt.title("Altitud |r|")
plt.xlabel("time [days]")
plt.ylabel("Altitude [km]")


#==============================================================================
#==============================================================================

#mpl.rcParams['toolbar'] = 'None'
#plt.figure(figsize=(8,2))
ax = plt.axes(projection='3d')
ax.set_aspect("equal")
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = Re*np.cos(u)*np.sin(v)
y = Re*np.sin(u)*np.sin(v)
z = Re*np.cos(v)
ax.plot_wireframe(x, y, z, color="r")
ax.plot3D(X3, Y3, Z3, 'gray')
#plt.grid();


#==============================================================================
#==============================================================================

plt.figure(figsize=(8,2))
plt.plot(maneuvers.tTrace/60/60/24,(X3/1e3));
plt.plot(maneuvers.tTrace/60/60/24,(Y3/1e3));
plt.plot(maneuvers.tTrace/60/60/24,(Z3/1e3));
plt.grid();
plt.title("Altitud |r|")
plt.xlabel("time [days]")
plt.ylabel("Altitude [km]")
