
import numpy as np
import sys
import matplotlib.pyplot as plt
from pylab import *

# Default values
n=100
M=5
nk=500
u0=5
p=0.25
Msig=2

if(len(sys.argv)!=6):
    print("PLease enter correct number of params. Default set of params will be used.")
else:
    n = sys.argv[1]  # spatial grid size.
    M = sys.argv[2]  # number of electrons injected per turn.
    nk = sys.argv[3]  # number of turns to simulate.
    u0 = sys.argv[4]  # threshold velocity.
    p = sys.argv[5]  # probability that ionization will occur
    Msig = sys.argv[6]  # deviation of elctrons injected per turn

xx=np.zeros(n*M) # Position
u=np.zeros(n*M)  # Velocity
dx=np.zeros(n*M) # Displacement

# Params to be plotted
I = []   #Intensity of emitted light
X = []   #Electron position
V = []   #Electron velocity

# Iterations for simulation
for iter in range(nk):
    # No of electrons to be injected
    m = int(randn() * Msig + M)
    EmptySpace=np.where(xx==0)[0] # All empty locations
    t=min(len(EmptySpace),m) # Max empty that can be filled

    # Initializing values
    xx[EmptySpace[:t]] = 1
    u[EmptySpace[:t]] = 0
    dx[EmptySpace[:t]] = 0

    # Finding positions and speed of electrons and add to X and V
    params = np.where(xx > 0)[0]
    X.extend(xx[params].tolist())
    V.extend(u[params].tolist())

    # Updating pos,velocity and displacement
    ii = np.where(xx > 0)[0]
    dx[ii] = u[ii] + 0.5
    xx[ii] += dx[ii]
    u[ii] += 1

    # If it has reached the end, set all params to zero.
    ii = np.where(xx > n)[0]
    dx[ii] = 0
    xx[ii] = 0
    u[ii] = 0

    # Upon velocity reaching threshold, set electrons will collide.
    kk = np.where(u >= u0)[0]                         #Electrons with velocities greater than threshold
    ll = np.where(np.random.rand(len(kk)) <= p)[0]    #Electrons collide with a probability 'p'
    kl = kk[ll]                                       #Contains the indices of energetic electrons that collide

    # Collision happens at time 't' for each colliding electron
    t = np.random.rand(len(kl))

    # resetting values of position and velocity upon collision
    xx[kl] = xx[kl] - dx[kl] + (u[kl] - 1) * t + 0.5 * t ** 2 + 0.5 * (1 - t) ** 2
    u[kl] = 0 + (1 - t)

    # A poorer approximation of position and velocity
    #xx[kl] = xx[kl] - dx[kl] * np.random.rand()
    #u[kl] = 0

    # The excited atoms at this location resulted in emission from that point. So we have to add a photon at that point.
    I.extend(xx[kl].tolist())


#histogram for light intensity
figure(0)
pops, bins, temp= hist(I, bins = np.arange(0,n+1,1), edgecolor='white', rwidth=1, color='black') #draw histogram
xpos = 0.5*(bins[0:-1] + bins[1:])
title("Light Intensity")
xlabel(r'Position$\rightarrow$')
ylabel(r'Intensity$\rightarrow$')
show()

#Tabulate results
print("Intensity data:")
print("position     count")
for i in range(len(pops)):
    print(str(bins[i]) + "   " + str(pops[i]))

#histogram for electron density
figure(1)
hist(X, bins=np.arange(0, n + 1, 1), edgecolor='white', rwidth=1, color='black')
title("Electron Density")
xlabel(r'Position$\rightarrow$')
ylabel(r'Number of Electrons$\rightarrow$')
show()

#phase space diagram
figure(2)
plot(X,V,'o', color='red')
title("Electron Phase Space")
xlabel(r'Position$\rightarrow$')
ylabel(r'Velocity$\rightarrow$')
show()








