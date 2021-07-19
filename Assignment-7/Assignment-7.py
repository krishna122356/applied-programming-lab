
# Importing libraries
import scipy.signal as sp
import sympy as sy
from sympy.abc import s
from pylab import *

# Function to plot response for a given input signal
def input_sim(H, Vi, t_range, heading, type):
    x = sp.lsim(H, Vi, t_range)[1]
    plot(t_range, Vi, label = 'Input signal')
    plot(t_range, x, label = 'Output response')
    title(heading + " input signal vs output response of " + type)
    xlabel("time (in seconds)")
    ylabel("Voltage (in V)")
    legend()
    grid(True)
    show()

# Function to plot Magnitude Response (Bode plot)
def bode_plotting(H, heading):
    w = logspace(0, 12, 1001)
    freqs = 1j * w
    hf = sy.lambdify(s, H, 'numpy')
    h = hf(freqs)

    ## Plotting in loglog scale (Bode plot)
    figure(0)
    loglog(w, abs(h), lw=2)
    title("Bode plot of the " + heading)
    xlabel("Frequency (in rad/s)")
    ylabel("Magnitude (in log scale)")
    grid(True)
    show()

# Function to extract coefficients from symbolic representation of a transfer function and returns LTI class.
# Here, using sympy we extract numerator and denominator, convert to polynomials and store coeffs. Then we apply LTI on it.
def LTI_find(sym):
    num, den = sy.fraction(sym)
    num, den = sy.Poly(num, s), sy.Poly(den, s)
    num, den = num.all_coeffs(), den.all_coeffs()
    H_lti = sp.lti(array(num, dtype = float), array(den, dtype = float))
    return H_lti

# Common function to analyse given input signal and transfer function
def plot_analysis(H, Vi, t_range, heading):
    H_lti = LTI_find(H)
    bode_plotting(H, heading)

    ## Computing and plotting step response
    step_response = H * 1/s
    step_response_lti = LTI_find(step_response)
    x = sp.impulse(step_response_lti, None, t_range)[1] ## Warning??
    figure(1)
    plot(t_range, x)
    title("Step Response of the " + heading)
    xlabel("time (in seconds)")
    ylabel("Response (in V)")
    grid(True)
    show()

    ## Response for a damped low frequency signal
    V_low_freq = exp(-300 * t_range) * cos(2 * 10**3 * pi * t_range)
    input_sim(H_lti, V_low_freq, t_range, "Low frequency (1KHz)", heading)

    ## Response to a damped high frequency signal
    V_high_freq = exp(-300 * t_range) * cos(2 * 10**6 * pi * t_range)
    input_sim(H_lti, V_high_freq, t_range, "High frequency (1MHz)", heading)

    ## Response to the given input signal
    input_sim(H_lti, Vi, t_range, "Given", heading)

# Lowpass case
def lowpass_tf(R1, R2, C1, C2, G):
    A = sy.Matrix([[0, 0, 1, -1/G],[-1/(1 + s * R2 * C2), 1, 0, 0],[0, -G, G, 1],[-1/R1 - 1/R2 - s * C1, 1/R2, 0, s * C1]])
    b =  sy.Matrix([0, 0, 0, -1/R1])
    V = A.inv() * b
    return A, b, V

t = linspace(0, 1e-2, 1000000)
## Passing values to compute the transfer function
A, b, V = lowpass_tf(10000, 10000, 1e-9, 1e-9, 1.586)
Vinput = cos(2 * 10**6 * pi * t)+sin(2000 * pi * t)  #input to filter

# Approximate behaviour at low frequencies
Vo_exp = 1/(1 + s * 1e4 * 1e-9) # Expected behaviour at high frequencies
bode_plotting(Vo_exp, "approx Lowpass Filter")
print(V[3])
## Passing for plot analysis
plot_analysis(V[3], Vinput, t, "Lowpass Filter")

# Highpass case
def highpass_tf(R1, R3, C1, C2, G):
    A = sy.Matrix([[0, -1, 0, 1/G],[s * C2 * R3/(s * C2 * R3 + 1), 0, -1, 0],[0, G, -G, 1],[-s * C2 -1/R1 - s * C1, 0, s * C2, 1/R1]])
    b = sy.Matrix([0, 0, 0, -s * C1])
    V = A.inv() * b
    return A, b, V

t = linspace(0, 1e-2, 1000000)
## Passing values to compute the transfer function
A, b, V = highpass_tf(1e4, 1e4, 1e-9, 1e-9, 1.586)
Vi = exp(-300 * t) * (sin(2e6 * pi * t) + cos(2e3 * pi * t))

# Approximate behaviour at high frequencies
Vo_exp = (s * 1e4 * 1e-9)/(1 + s * 1e4 * 1e-9)
bode_plotting(Vo_exp, "approx Highpass Filter")

## Passing for plot analysis
plot_analysis(V[3], Vi, t, "Highpass Filter")