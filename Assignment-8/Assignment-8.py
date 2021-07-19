from pylab import *

def Solver(func,Time=0,N=2048):
    # Defining each function and its limits
    if(func=="sin^3x"):
        Rleft = -2 * pi
        Rright = 2 * pi
        Xlimit = 10
    elif(func=="cos^3x"):
        Rleft = -4 * pi
        Rright = 4 * pi
        Xlimit = 10
    elif (func == "cos(20 * x + 5 * cos(x))"):
        Rleft = -4 * pi
        Rright = 4 * pi
        Xlimit = 40
    elif (func == "Gaussian"):
        Rleft = -Time * pi
        Rright = Time * pi
        Xlimit = 10
    # x is the Time axis
    x = linspace(Rleft, Rright, N + 1)
    x = x[:-1]

    # Here we calculate the values the functions take in the t and w domain. Gaussian is handled separately.
    if (func == "sin^3x"):
        y = (sin(x)) ** 3
    elif (func == "cos^3x"):
        y = (cos(x)) ** 3
    elif (func == "cos(20 * x + 5 * cos(x))"):
        y = cos(20 * x + 5 * cos(x))
    elif (func == "Gaussian"):
        y = exp((-x ** 2) / 2)

    Y = fftshift(fft(y)) / N
    # Normalization of frequency axis values
    normalization = N * (pi / (Rright - Rleft))
    w = linspace(-normalization, normalization, N+1)
    w = w[:-1]
    if (func == "Gaussian"):
        Y = fftshift(fft((y))) / N
        Y =Y* sqrt(2 * pi) / max(Y)
        Yotherway = exp(-w**2/2) * sqrt(2 * pi)
        print("We get the max error for the time range",Time,"as",abs(Y-Yotherway).max())

    # Magnitude and Phase plots of DFT curves.

    figure()
    subplot(2, 1, 1)
    plot(w, abs(Y), lw=2)
    xlim([-Xlimit, Xlimit])
    ylabel(r"$|Y|$", size=16)
    title("Magnitude of "+func)
    grid(True)
    subplot(2, 1, 2)
    scatter(w,angle(Y),marker='o',color='#ADD8E6')
    ii = where(abs(Y) > 1e-3)
    plot(w[ii], angle(Y[ii]),'ro', lw = 2)
    xlim([-Xlimit, Xlimit])
    ylabel(r"Phase of $Y$", size=16)
    xlabel("Phase of "+func, size=16)
    grid(True)
    show()
# Part 1
Solver("sin^3x")
Solver("cos^3x")
# Part 2
Solver("cos(20 * x + 5 * cos(x))")
# Part 3
for __ in range(10):
    Solver("Gaussian",__+1)


