import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

import scipy

# Question 1
x=np.arange(-2*np.pi,4*np.pi,0.1)
y=np.exp(x)
plt.figure(1)
plt.grid(True)
plt.semilogy(x,y)
plt.xlabel('x')
plt.ylabel('exp(x) in semilog')
plt.title('exp(x) in semilog')
plt.show()

x=np.arange(-2*np.pi,4*np.pi,0.1)
y=np.cos(np.cos(x))
plt.figure(2)
plt.grid(True)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('Cos(Cos(x))')
plt.title('Cos(Cos(x))')
plt.show()

# Defining functions
def expo(x):
    return np.exp(x)

def coscos(x):
    return np.cos(np.cos(x))

# Note: separate functions for u and v are not defined. Instead they have been defined inside the ComputeCoeffs function.

# Question 2
# Computes Coefficients using integration.
def ComputeCoeffs(f,numberOfCoeffs=51):
    coscoeff = []
    sincoeff = []
    u=lambda x,k:f(x)*np.cos(x*k) # Effectively u(x,k)
    v=lambda x,k:f(x)*np.sin(x*k) # Effectively v(x,k)
    for i in range((numberOfCoeffs+1)//2):
        result = (integrate.quad(u, 0, 2*np.pi,args=(i)))[0]/np.pi
        coscoeff.append(result)
    coscoeff[0]/=2 # For a0 term
    for i in range(1,(numberOfCoeffs+1)//2):
        result = (integrate.quad(v, 0, 2*np.pi,args=(i)))[0]/np.pi
        sincoeff.append(result)
    return [coscoeff,sincoeff]

# Question 3
# Plots 4 plots, which are the coeffs of the exp and coscos in both semilog and loglog scale.
dum=(ComputeCoeffs(expo))
cexp=np.zeros(51) # Fourier coeffs of exp
cexp[0] = dum[0][0]
for i in range(1, 51, 2):
    cexp[i] = dum[0][(i + 1) // 2]
    cexp[i + 1] = dum[1][i // 2]

plt.figure(3)
plt.grid(True)
plt.title('semilogy of exp')
plt.semilogy(abs(cexp),"ro")
plt.ylabel('semilogy of exp')
plt.grid(True)
plt.show()

plt.figure(4)
plt.title('loglog of exp')
plt.loglog(abs(cexp),"ro")
plt.ylabel('loglog of exp')
plt.grid(True)
plt.show()

dum=(ComputeCoeffs(coscos))
ccoscos=np.zeros(51) # Fourier coeffs of coscos
ccoscos[0] = dum[0][0]
for i in range(1, 51, 2):
    ccoscos[i] = dum[0][(i + 1) // 2]
    ccoscos[i + 1] = dum[1][i // 2]

plt.figure(5)
plt.grid(True)
plt.title('semilogy of coscos')
plt.semilogy(abs(ccoscos),"ro")
plt.ylabel('semilogy of coscos')
plt.show()

plt.figure(6)
plt.grid(True)
plt.title('loglog of coscos')
plt.loglog(abs(ccoscos),"ro")
plt.ylabel('loglog of coscos')
plt.show()

# Question 4
x=np.linspace(0,2*np.pi,401)
x=x[:-1]
bcoscos=coscos(x)
bexp=expo(x)
A=np.zeros((400,51))
A[:,0]=1
for k in range(1,26):
    A[:, 2 * k - 1] = np.cos(k * x)
    A[:, 2 * k] = np.sin(k * x)
c1coscos=scipy.linalg.lstsq(A,bcoscos)[0]
c1exp=scipy.linalg.lstsq(A,bexp)[0]

# Question 5
# Plots 4 plots, each of which compare the coefficients obtained by lstsq method with previous graphs.
plt.figure(7)
plt.grid(True)
plt.title('semilogy of exp with lstsq method')
plt.semilogy(abs(cexp),"ro")
plt.semilogy(abs(c1exp),"go",markersize=4)
plt.ylabel('semilogy of exp with lstsq method')
plt.legend(["Actual function","lstsq Approx"],loc="upper right")
plt.show()

plt.figure(8)
plt.grid(True)
plt.title('loglog of exp with lstsq method')
plt.loglog(abs(cexp),"ro")
plt.loglog(abs(c1exp),"go",markersize=4)
plt.ylabel('loglog of exp with lstsq method')
plt.legend(["Actual function","lstsq Approx"],loc="upper right")
plt.show()

plt.figure(9)
plt.grid(True)
plt.title('semilogy of coscos with lstsq method')
plt.semilogy(abs(ccoscos),"ro")
plt.semilogy(abs(c1coscos),"go",markersize=4)
plt.ylabel('semilogy of coscos with lstsq method')
plt.legend(["Actual function","lstsq Approx"],loc="upper right")
plt.show()

plt.figure(10)
plt.grid(True)
plt.title('loglog of coscos with lstsq method')
plt.loglog(abs(ccoscos),"ro")
plt.loglog(abs(c1coscos),"go",markersize=4)
plt.ylabel('loglog of coscos with lstsq method')
plt.legend(["Actual function","lstsq Approx"],loc="upper right")
plt.show()

# Question 6
# Computes errors. Error is small in coscos case as the function is periodic.
Error_exp=np.max(np.abs(c1exp-cexp))
Error_coscos=np.max(np.abs(c1coscos-ccoscos))
print("Error in exp is",Error_exp)
print("Error in coscos",Error_coscos)

# Question 7
# Compares the actual graph with the one obtained by lstsq method
plt.figure(1)
plt.grid(True)
plt.title('actual exp vs exp obtained with coeffs from lstsq method')
b=np.dot(A,c1exp)
plt.semilogy(x,b,"go",markersize=2)
plt.semilogy(x,expo(x))
plt.legend(["lstsq Approx","Actual function"],loc="upper right")
plt.show()

plt.figure(2)
plt.grid(True)
plt.title('actual coscos vs coscos obtained with coeffs from lstsq method')
b=np.dot(A,c1coscos)
plt.plot(x,b,"go",markersize=2)
plt.plot(x,coscos(x))
plt.legend(["lstsq Approx","Actual function"],loc="upper right")
plt.show()


