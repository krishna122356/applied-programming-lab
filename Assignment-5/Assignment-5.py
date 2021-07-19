# Importing libraries
import numpy as np
import os, sys
import scipy.linalg as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

# Note: size of plate has been assumed to be 1cm by 1cm to solve the question.
if(len(sys.argv)!=5):
    print("Incorrect number of params have been given, exactly 4 should be given. Try giving 25 25 8 1500 If unsure.")
    exit()
else:
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
    radius = float(sys.argv[3])
    Niter = int(sys.argv[4])

print("These are the params:")
print("Nx:",Nx)
print("Ny:",Ny)
print("radius:",radius)
print("iterations:",Niter)
print("Size of conductor is 1cm x 1cm")

# Todo later - convert radius to appropriate units.
# Done - radius is normalized to per cm from given units
radius = radius * (1/2) * ((1/Nx) + (1/Ny))

phi=np.zeros((Nx,Ny),dtype=float)
x = np.linspace(-0.5, 0.5, num = Nx, dtype = float)
y = np.linspace(-0.5, 0.5, num = Ny, dtype = float)
Y,X = np.meshgrid(y,x)
phi1points=np.where(X**2+Y**2<=radius**2)
phi[phi1points]=1.0

plt.figure(1)
plt.title("Plot of Initial potentials")
plt.xlabel("X coords")
plt.ylabel("Y coords")
plt.contourf(X,Y,phi,colors=['#481b6c', 'blue', 'green', 'yellow', 'orange', 'red', 'white'])
plt.plot(x[phi1points[0]], y[phi1points[1]], 'ro')
plt.colorbar()
plt.show()

# Now to iterate Niter times to get accurate potential values.
# Also, we will store errors in an array as well.
errors=np.zeros(Niter)
for i in range(Niter):
    oldphi=phi.copy()
    phi[1:-1,1:-1]=0.25*(oldphi[1:-1,0:-2]+oldphi[1:-1,2:]+oldphi[0:-2,1:-1]+oldphi[2:,1:-1])
    phi[:,0]=phi[:,1]
    phi[:,-1]=phi[:,-2]
    phi[0,:]=phi[1,:]
    phi[-1,:]=0
    phi[phi1points]=1.0
    errors[i]=np.max(np.abs(oldphi-phi))

# Plot the errors in loglog scale
plt.figure(2)
plt.title("Errors in loglog scale")
plt.xlabel("No of iters")
plt.ylabel("Errors")
plt.loglog((np.asarray(range(Niter))+1),errors)
plt.loglog((np.asarray(range(Niter)) + 1)[::50], errors[::50],'ro')
plt.legend(["All errors","every 50th value"])
plt.show()

# Plot errors in semilog scale
plt.figure(3)
plt.title("Plot of errors in semilog scale")
plt.xlabel("Number of iterations")
plt.ylabel("Errors")
plt.semilogy(np.asarray(range(Niter)), errors)
plt.semilogy((np.asarray(range(Niter)) + 1)[::50], errors[::50],'ro')
plt.legend(["All errors","every 50th value"])
plt.show()

# There may be errors after 500 points, so lets check for errors here as well.
def Error_Fit(errors, initial = 0):
    n_iter = errors.shape[0]
    log_errors = np.log(errors)
    return sp.lstsq(np.c_[np.ones((1, n_iter)).reshape(n_iter, 1), np.array(range(initial, n_iter + initial))], log_errors)[0]

## It is possible that there might be more errors when we take the error values from the 1st iteration so we identify the parameters for the entire vector and for errors after 500 iterations
Acoeff1, Bcoeff1 = Error_Fit(errors)
Acoeff2, Bcoeff2 = Error_Fit(errors[500:], 500)
K = np.array(range(Niter)) + 1

print("""\nIteration completed and errors have been fitted into an exponential plot of the form 'exp(A+B*x)'.
    Fitting based on all errors: exp({:.5} {:.5}*x)
    Fitting based on all errors after 500 iterations: exp({:.5} {:.5}*x)""".format(Acoeff1, Bcoeff1, Acoeff2, Bcoeff2), sep='')

# Plotting of errors in semilog scale and loglog scale
plt.figure(4)
plt.title("Plot of actual errors vs fitted errors - semilog scale")
plt.xlabel("Number of iterations")
plt.ylabel("Errors")
plt.semilogy(K, errors)
plt.semilogy(K[::100], np.exp(Acoeff1 + Bcoeff1 * (K-1))[::100], 'ro')
plt.semilogy(K[::100], np.exp(Acoeff2 + Bcoeff2 * (K-1))[::100], 'go', markersize = 4)
plt.legend(["Actual errors","fit1","fit2"])
plt.show()

plt.figure(5)
plt.title("Plot of actual errors vs fitted errors in loglog scale")
plt.xlabel("Number of iterations")
plt.ylabel("Errors")
plt.loglog(K, errors)
plt.loglog(K[::100], np.exp(Acoeff1 + Bcoeff1 * (K-1))[::100], 'ro')
plt.loglog(K[::100], np.exp(Acoeff2 + Bcoeff2 * (K-1))[::100], 'go', markersize = 4)
plt.legend([" Actual errors", "fit using all values", "fit after 500 values"])
plt.show()

# Computing cumulative error

## We shall compute the sum of all the errors after Niter iterations so that we have an idea about how much error we might
N_iterations = np.arange(200, 2001, 10)
Error_iteration = -(Acoeff1/Bcoeff1) * np.exp(Bcoeff1 * (N_iterations + 0.5))

## Plotting Cumulative error in loglog scale against number of iterations
plt.figure(6)
plt.title("Plot of Cumulative error in loglog scale")
plt.xlabel("Number of iterations")
plt.ylabel("Maximum error in computation")
plt.loglog(N_iterations, np.abs(Error_iteration), 'ro', markersize = 3)
plt.grid(True)
plt.show()

# 3D plot for potential
fig4=plt.figure(7)
ax=p3.Axes3D(fig4)
plt.title("The 3D surface plot of potential")
surf=ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=plt.cm.jet,linewidth=0, antialiased=False)
plt.show()

# 2D plot for contour of potentials
plt.figure(8)
plt.title("2-D Contour Plot of Potentials")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x[phi1points[0]], y[phi1points[1]],'ro')
plt.contourf(Y, X[::-1], phi, cmap='magma')
plt.colorbar()
plt.show()

# Computing current density distribution
Jx = 1/2 * (phi[1:-1, 0:-2] - phi[1:-1, 2:])
Jy = 1/2 * (phi[:-2, 1:-1] - phi[2:, 1:-1])

# Plotting of current density
plt.figure(9)
plt.title("vector plot for current flow")
plt.quiver(Y[1:-1, 1:-1], -X[1:-1, 1:-1], -Jx[:, ::-1], -Jy)
plt.plot(x[phi1points[0]], y[phi1points[1]],'ro')
plt.show()

# Compute Temperatures and plotting it
T = 300 * np.ones((Nx,Ny), dtype = float)

for k in range(Niter):
    oldT = T.copy()
    T[1:-1, 1:-1] = 0.25 * (oldT[1:-1, 0:-2] + oldT[1:-1, 2:] + oldT[0:-2, 1:-1] + oldT[2:, 1:-1] + (Jx)**2 + (Jy)**2)
    T[:, 0] = T[:, 1]
    T[:, Nx-1] = T[:, Nx-2]
    T[0, :] = T[1, :]
    T[Ny-1, :] = 300.0
    T[phi1points] = 300.0

# Plotting 2-D Contour of Temperature distribution
plt.figure(10)
plt.title("2-D Contour plot of Temperature distribution")
plt.xlabel("X")
plt.ylabel("Y")
plt.contourf(Y, X[::-1], T, cmap='magma')
plt.colorbar()
plt.show()





