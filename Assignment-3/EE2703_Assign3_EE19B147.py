import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special as sp

# Defining sig
sig = np.logspace(-1,-3,9)

# Q2. Extract data
file_data = np.loadtxt("fitting.dat")
time = file_data[:, 0]
cols = file_data[:, 1:]

# g function for Q4.
def gfunc(t, A, B):
	return A * sp.jn(2, t) + B * t

# Plot all columns 3.
labels = np.append(sig, "true Value")

# Assigning labels for Q3, followed by plotting it
for i in range(len(labels) - 1):
	labels[i] = "σ" + "$_" + str(i + 1) + "$" + "=" + str(round(sig[i],3))

plt.figure(0)
plt.plot(time,cols)
plt.grid(True)
plt.title(r'Q4: A Plot of f(t) vs t for each noise Levels')
plt.xlabel('$t$')
plt.ylabel('$f(t)+noise$')

# Plot function  both r being plotted in the same figure

#plt.title("Original Plot for true value")
plt.plot(time,gfunc(time, 1.05, -0.105), 'k', linewidth=3)
plt.legend(labels)
plt.show()

##################################################
# Plot Errorbar and original 5.
y_true = gfunc(time, 1.05, -0.105)
sigma = np.std(cols[:, 0] - y_true)
plt.plot(time, y_true, label='f(t)')
plt.grid(True)
plt.title('Q5: Data points for σ = ' + str(sig[0]) + ' Along with exact function')
plt.xlabel('$t$')
plt.errorbar(time[::5],cols[::5,0],sigma,fmt='ro',label='errorbar')
plt.legend()
plt.show()
##################################################

# Construction for Q6.

M = np.c_[sp.jn(2, time), time]

# Q7

A = np.linspace(0,2,21)
B = np.linspace(-0.2,0,21)

MeanSquareError = np.zeros((21,21))

for ii in range(21):
	for jj in range(21):
		# Taking average.
		MeanSquareError[ii, jj] = sum((cols[:, 0] - gfunc(time,A[ii],B[jj]))**2)/101

# Contour plot Q8

CS = plt.contour(np.meshgrid(A,B)[0], np.meshgrid(A,B)[1], MeanSquareError, np.linspace(0.025, 0.5, 20))
plt.clabel(CS,CS.levels[:4], inline=1, fontsize=10)
plt.title("Q8: Contour Plot of ε" + "$_" + "i" + "$" + "$_" + "j" + "$")
plt.xlabel('$A$')
plt.ylabel('$B$')
plt.plot(1.05, -0.105, 'ro')
plt.annotate('Exact Value', (1.05, -0.105))
plt.show()

# Estimation Q9

EstimationArray = []
ErrorArray = []

for j in range(np.size(cols,1)):
	Estimation_i,_,_,_ = np.linalg.lstsq(M, cols[:, j], rcond=None)
	Error_i = [abs(Estimation_i[0] - 1.05), abs(Estimation_i[1] + 0.105)]

	EstimationArray.append(Estimation_i)
	ErrorArray.append(Error_i)

EstimationArray = np.array(EstimationArray)
ErrorArray = np.array(ErrorArray)

# Error Analysis

plt.plot(sig,ErrorArray[:, 0],linestyle='--',marker='o',color='r',label='A error')
plt.plot(sig,ErrorArray[:, 1],linestyle='--',marker='o',color='b',label='B error')
plt.xlabel("$Noise\ Standard\ Deviation$")
plt.ylabel("$Mean\ Squared\ Error$")
plt.title("Q10: Variation of error w.r.t noise")
plt.legend()
plt.grid(True)
plt.show()

# Variation of error with noise in log scale
plt.loglog(sig,ErrorArray[:, 0],linestyle='',marker='o',color='r',label='Aerr')
plt.loglog(sig,ErrorArray[:, 1],linestyle='',marker='o',color='b',label='Berr')
plt.xlabel("$Noise\ Standard\ Deviation$")
plt.ylabel("$Mean\ Squared\ Error$")
plt.title("Q11 Variation of error with noise in log scale")
plt.grid(True)
plt.stem(sig,ErrorArray[:, 0],'ro');
plt.stem(sig,ErrorArray[:, 1],'b');
plt.show()







