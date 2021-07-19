from pylab import *


# Function to calculate and plot FFt curves as described in the document.
def FFTfunction(func, HEAD, rleft=-pi, rright=pi, N=64, Xlim=10, typeOfGraph=False, isOdd=False):
    # x is the time variable. w is for frequency.
    x = linspace(rleft, rright, N + 1)
    x = x[:-1]

    tempo = N * (pi / (rleft - rright))
    w = linspace(-tempo, tempo, N + 1)
    w = w[:-1]

    y = func(x)
    # for odd function, set value at 0 to 0.
    if isOdd:
        y[0] = 0
    Y = fftshift(fft(fftshift(y))) / N

    ## Plotting work.
    figure()
    if typeOfGraph:
        subplot(2, 1, 1)
        xlim([1, 10])
        ylim([-20, 0])
        title("Freq Spectrum " + HEAD)
        semilogx(w, 20 * log10(abs(Y)), lw=2)
        ylabel(r"$|Y|$ in deciBels", size=16)
        grid(True)

    else:
        subplot(2, 1, 1)
        xlim([-Xlim, Xlim])
        title("Freq Spectrum " + HEAD)
        plot(w, abs(Y), lw=2)
        ylabel(r"$|Y|$", size=16)
        grid(True)

    ii = where(abs(Y) > 1e-3)

    subplot(2, 1, 2)
    xlim([-Xlim, Xlim])
    scatter(w, angle(Y), marker='o', color='#D9D9D9')
    plot(w[ii], angle(Y[ii]), 'go', lw=2)
    ylabel(r"angle in rad", size=16)
    xlabel(r"w in rad/s", size=16)
    grid(True)

    show()


## Function to return Hamming Window sequence
def HammingWindow(l, r, lenght,n):
    return fftshift(l + r * cos((2 * pi * n) / (lenght - 1)))

# Question 2
# Func for cos(0.86 * x)**3
def CosCube(x):
    return (cos(0.86 * x)) **3

def CosCubeHamming(x):
    return CosCube(x) * HammingWindow(0.54, 0.46, len(x),arange(len(x)))

#Analysis
FFTfunction(CosCube, "$cos^{3}(0.86*t)$ wout windowing", -4 * pi, 4 * pi, N=512)

FFTfunction(CosCubeHamming, "$cos^{3}(0.86*t)$ with windowing", -4 * pi, 4 * pi, N=512)

# Question 3

## Func for cos(0.75x+0.3). We will first calculate it, and then try to estimate it.
def CosEst(x):
    return cos(0.75 * x + 0.3)

## With windowing
def CosEstHamming(x):
    return CosEst(x) * HammingWindow(0.54, 0.46, len(x) ,arange(len(x)))

def Expected(w, mag, phase):
    actual_mag = where(mag > 0.2)

    w_avg = sum((mag[actual_mag]**2) * abs(w[actual_mag]))/sum(mag[actual_mag]**2)

    phase_avg = mean(abs(phase[actual_mag]))

    print("Expected w0:", w_avg)
    print("Expected delta:", phase_avg)

x = linspace(-4*pi, 4*pi, 1024 + 1)
x = x[:-1]

tempo = 1024 * (pi / (4*pi + 4*pi))
w = linspace(-tempo, tempo, 1024 + 1)
w = w[:-1]

y = CosEst(x)
Y = fftshift(fft(fftshift(y))) / 1024
print(Expected(w,abs(Y),angle(Y)))

# Analysis
FFTfunction(CosEst, "$cos(0.75t+0.3)$ wout Windowing", -4 * pi, 4 * pi, 1024,4)

FFTfunction(CosEstHamming, "$cos(0.75t+0.3)$ with Windowing", -4 * pi, 4 * pi, 1024, 4)


# Question 4

# Adding white gaussian noise
def CosWithNoise(x):
    return cos(0.75 * x + 0.3) + 0.1 * randn(len(x))

# And Windowing
def CosWithNoiseHamming(x):
    return CosWithNoise(x) * HammingWindow(0.54, 0.46, len(x),arange(len(x)))

y = CosWithNoise(x)
Y = fftshift(fft(fftshift(y))) / 1024
print(Expected(w,abs(Y),angle(Y)))

#Analysis
FFTfunction(CosWithNoise, "noisy $cos(0.75t+0.3)$ wout Windowing", -4 * pi, 4 * pi, 512, 4)

FFTfunction(CosWithNoiseHamming, "noisy $cos(0.75t+0.3)$ with Windowing", -4 * pi, 4 * pi, 512, 4)

# Question 5

# Chirp function
def Chirp(x):
    return cos(16 * x * (1.5 + x/(2 * pi)))

# With Windowing
def ChirpHamming(x):
    return Chirp(x) * HammingWindow(0.54, 0.46, len(x),arange(len(x)))

## Passing function definitions for DFT Analysis
FFTfunction(Chirp, "chirp function wout Windowing",-pi,pi, 1024, 100)

FFTfunction(ChirpHamming, "chirp function with Windowing",-pi,pi, 1024, 100)

# Question 6

# Time vector and breaking it up using reshape.
t = linspace(-pi, pi, 1025)
t=t[:-1]
t_split = reshape(t, (16, 64))
mags = []
phases = []
# Here, we loop through various values of t to compute dft.
for i in t_split:
    y = Chirp(i)
    y[0] = 0
    y = fftshift(y)
    Y = fftshift(fft(y))/64
    mags=mags+[abs(Y)]
    phases=phases+[angle(Y)]

magnitudes = array(mags)
phases = array(phases)

X = linspace(-512, 512, 65)
X = X[:-1]
Y = linspace(-pi, pi, 17)
Y=Y[:-1]

X, Y = meshgrid(X, Y)

## Plotting 3d plot of Frequency response vs frequency and time
fig = figure(1)

ax = axes(projection = "3d")
ax.set_xlabel("w in rad/s")
ax.set_ylabel("$t$ in s")
Surface = ax.plot_surface(X, Y, magnitudes, cmap = cm.plasma)
fig.colorbar(Surface, shrink = 0.5)
ax.set_title("3D Surface plot of Mag Response vs Freq and Time")

show()

fig = figure(2)
ax = axes(projection = '3d')
ax.set_xlabel("w in rad/s")
ax.set_ylabel("$t$ in s")
Surface = ax.plot_surface(X, Y, phases, cmap = cm.coolwarm_r)
fig.colorbar(Surface, shrink = 0.5)
ax.set_title("3D Surface plot of Phase Response vs Freq and Time")
ax.set_zlabel("Phase of Y in rad")

show()