from pylab import *
import scipy.signal as sp
import numpy as np

# Section 1

# Question 1 & 2 - Defining the function.

def FunctionQ1_Q2(decay,Sno):
    # time
    t=np.linspace(0,50,1000)

    # not useful but just in case -
    # F = sp.lti([1,decay],[1,2*decay,2.25+decay*decay])

    # X(s) using np.poly functions
    X = sp.lti([1,decay],np.polymul([1,2*decay,2.25+decay*decay],[1,0,2.25]))
    t,x = sp.impulse(X,None,t)
    # Plotting
    figure(Sno)
    ylabel("x(t)")
    xlabel("time")
    title("Question No - " + str(Sno+1))
    plot(t, x, label="Decay = " + str(decay))
    grid(True)
    legend()
    show()

# Question 1 & 2 - plotted using the function
FunctionQ1_Q2(0.5,0)
FunctionQ1_Q2(0.05,1)

# Question 3 - Finding time responses as we vary the frequency of the cosine term.

t = np.linspace(0,50,1000)
H = sp.lti([1], [1, 0, 2.25])
t, h = sp.impulse(H, None, t)

figure(2)
title("Output with varying frequency.")
ylabel("$x(t)$")
xlabel("$t$")
grid(True)

for i in linspace(1.4,1.6,5):
    func = np.cos(i*t) * np.exp(-0.05*t)
    t, x, svec = sp.lsim(H, func, t)
    plot(t, x, label = "Frequency = " + str(i))

legend()
show()

# Section 2

# Question 4 - Solve and plot the functions x and y.
t = np.linspace(0,20,1000)

X = sp.lti([0.5,0,1,0],[0.5,0,1.5,0,0])
Y = sp.lti([1,0],[0.5,0,1.5,0,0])

t,x = sp.impulse(X,None,t)
t,y = sp.impulse(Y,None,t)

figure(3)
title("Plots of x(t) and y(t)")
ylabel("$x(t)$ and $y(t)$")
xlabel("$t$")
grid(True)
plot(t, x, label = "x(t)")
plot(t, y, label = "y(t)")
legend()
show()

# Section 3

# Question 5 - Find the transfer function for the given circuit and show the bode plots for magnitude and phase.

H = sp.lti([1], [1e-12, 1e-4, 1])
w, S, phi = H.bode()

figure(4)
# Plotting functions
semilogx(w, S)
title("Magnitude Plot")
ylabel("Magnitude(in dB)")
xlabel("$\\omega$")
grid(True)
show()

figure(5)
semilogx(w, phi)
title("Phase Plot")
ylabel("Phase")
xlabel("$\\omega$")
grid(True)
show()

# Question 6 - Find Vo(t) from the given Vi(t)

#For short term response
t = np.linspace(0, 20e-6, 10000)
#Input signal
Vin = np.cos(1e3*t)-np.cos(1e6*t)

#Convolution
t, Vout, svec = sp.lsim(H, Vin, t)
#Plotting functions
figure(6)
plot(t, Vout)
xlabel("$t$")
ylabel("$V_{out}$")
title("$V_{out}$ vs $t$ (Initial response)")
grid(True)
show()

#For long term response
t = np.linspace(0, 2e-2, 10000)
#Convolution
t, Vout, svec = sp.lsim(H, Vin, t)
#PLotting functions
figure(7)
plot(t, Vout)
xlabel("$t$")
ylabel("$V_{out}$")
title("$V_{out}$ vs $t$(Steady State response)")
grid(True)
show()



