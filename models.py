import numpy as np
def atmosDensity(z):
    i=0
    h = [ 0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000];
    rho = [1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5, 1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9, 2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12, 1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15];
    H = [ 7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661, 5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322, 21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019, 60.980, 65.654, 76.377, 100.587, 147.203, 208.020];
    for j in range(0,27):
        if z >= h[j] and z < h[j+1]:
            i = j
    if z == 1000:
        i = 27
    density = rho[i]*np.exp(-(z-h[i])/H[i]);
    return density
class curtisSat:
  m = 100
  A = np.pi/4
  Cd = 2.2
  BC = m/(Cd*A)
class cubesat:
  #Satellite Data
  m = 3
  dimensions = [10e-2,10e-2,30e-2]
  A = dimensions[0]*dimensions[1]
  Cd = 2.2
  BC = m/(Cd*A)
  
  #Solid fuel
  #https://www.publicacionesfac.com/index.php/cienciaypoderaereo/article/view/502/702
  fuel_mass = 1#Kg
  burning_rate =  1/(90*24*60*60)#Kg/s