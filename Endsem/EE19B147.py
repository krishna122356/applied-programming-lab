"""
Name : Krishna Somasundaram
Roll No: EE19B147
Code Execution: python EE19B147.py
Pdf Report: EE19B147.pdf
"""


# Question 1
"""
Pseudocode: ( S.No is not indicative of the question number here. )

1. Define the axes and the meshgrid.
2. Define phi as angle from 0 to 2*pi. Using it, find x and y coordinates of elements on the loop
3. Find r' and dl for each element. We find the x and y coordinates and concatenate them. r' is in direction of phi cap,
   while dl is perpendicular to it.
4. Find the current elements, which is a function of Cos(phi). Then we plot location of the current elements and the current
   element (physics) vectors. We plot the (physics) vectors using quiver.
5. Function Calc(paramater l: index): l represents the index of the current element. 
    a. Find x, y and z coordinates of R. We do this, for example with Rx as rx-Loop_x[l]
    b. Now, find magnitude of R as sqrt(Rx^2+Ry^2+Rz^2). 
    c. Now, evaluate Ax and Ay as described by the formula in the question. Then return these params.
6. For every element in loop, We call Calc(l) and sum its contribution towards Ax,Ay
7. Now we evaluate B along z axis by using values of Ax and Ay from surrounding cell.

        ________________
        |    |    |    |
        |    |-Ax |    |
        ________________
        |    |    |    |
        |-Ay | O  | Ay |
        ________________
        |    |    |    |
        |    | Ax |    |
        ________________
        
    As delta x and delta y are equal to 1, we sum these 4 values for each point on the z axis and divide by 2. 
    (For further justification, see pdf)
8. Plot the loglog plot of B as a function of z.
9. Use lstsq to try and approximate B(z) to an equivalent of c*(z^b). We do this by taking log of both sides
    a. Concatenate log(z) and ones to get m.
    b. set n to log(B).
    c. Then we obtain b, log(c) as output of lstsq(m,n)[0]. Obtain c by taking exponential of log(c).
10. Lastly, plot the fit of B along the z axis with the actual output.
11. Change the value of current, and repeat steps 4 - 10 twice. (Once for each type of current) 
"""

## Question 2
from pylab import *

# x,y and z axes.
x=linspace(-1,1,3)
y=linspace(-1,1,3)
z=linspace(1,1000,1000)
# We create a mesh grid, as asked in the question. We set the indexing to type ij, as the default goes as y,z,x.
rx,ry,rz=meshgrid(x,y,z,indexing='ij')

## Question 3

phi=linspace(0,2*pi,101)[:-1] # The angle phi as it goes from 0 to 2pi over the circumference of the loop.
x_loop=10*cos(phi) # x coordinates of the loop
y_loop=10*sin(phi) # y coordinates of the loop
# I have done the plots after Question 4.


# Question 4
rdash=c_[x_loop,y_loop]
dl_x=-(pi/5)*(y_loop/10)
dl_y=(pi/5)*(x_loop/10)
dl=c_[dl_x,dl_y]
# We use numpy.c_ to concatenate the x_loop and y_loop vectors for r' and dl_x and dl_y for dl vector.
# As length of each dl element should by 2*pi*r/100,
# it has been simplified as such. Note that x_loop and y_loop contain a 10 factor within them, and r = 10.

# Variation of current as a function of the angle, as described in the question 3 and plot it.
I=(10**7)*cos(phi)

plt.figure(0)
plt.title("Current elements of the loop, which vary as cos(\u03C6)*e^(-jkR)")
plt.scatter(x_loop,y_loop,label="location of elements")
plt.quiver(x_loop,y_loop,dl_x*I,dl_y*I,label = "Current elements")
plt.grid()
plt.legend(loc="upper right")
plt.ylabel("y-axis $\\rightarrow$")
plt.xlabel("x-axis $\\rightarrow$")
plt.show()

# Question 5 and 6
# The function Calc(l) takes the index of the element as a parameter and evaluates Ax and Ay based on it.
def Calc(l):
    l_x,l_y=rdash[l] # x any y coordinates of a particular element in r'

    # Now, let us calculcate |x-l_x| and |y-l_y|, which can be used to finally find R using R = |r-r'|.
    Rx=abs(rx-l_x)
    Ry=abs(ry-l_y)
    Rz=rz
    Rijkl = sqrt(Rx**2+Ry**2+Rz**2)

    # Question 6, compute A vector. k = 0.1
    Axl=cos(2*pi*l/100)*np.exp(complex(0,1)*0.1*Rijkl)*dl[l][0]/Rijkl
    Ayl=cos(2*pi*l/100)*np.exp(complex(0,1)*0.1*Rijkl)*dl[l][1]/Rijkl

    return (Axl,Ayl)

# Question 7
# Ax and Ay are intialized to dAx(zero) and dAy(zero), and using a for loop we compute their value over the 100 elements in the loop.
Ax,Ay=Calc(0)

for l in range(1,100):
    dAx,dAy=Calc(l)
    Ax=Ax+dAx
    Ay=Ay+dAy

# Question 8
# Finding B using simplified expression of curl of A. This is not the formula given in the Problem statement, however I have
# derived this in my report. The formula suggested in the question is dimensionally incorrect. To change this to the formula in
# the problem statement, just divide it by 2.
B = (Ax[1,0,:]-Ax[1,2,:])/2+(Ay[2,1,:]-Ay[0,1,:])/2
# Also see point number 7 in the pseudocode above.

# Question 9
# Plot B as a function of z
plt.figure(1)
plt.grid()
plt.title("B as a function of z")
plt.loglog(z,abs(B),label="B(z)")
plt.ylabel("B(z) $\\rightarrow$")
plt.xlabel("z $\\rightarrow$")
plt.legend()
plt.show()

# Question 10
# Using lstsq method, Approximate B to c*(z^b)
m = c_[log(z),ones(len(z))] # Set m to [log(z),1]
n = log(abs(B)) # Set n = log(B)
p = lstsq(m,n,rcond=None)[0]
b = p[0]
c = exp(p[1])
bzfit = c*(z**b)
print("For a non static current that varies as a function of cos(phi), the output is")
print("Value of b: ",b)
print("Value of c: ",c)

# Plot B as a function of z along with the approximation from lstsq.
plt.figure(2)
plt.title("B as a function of z, with the best lstsq fit")
plt.grid()
plt.ylabel("B(z) $\\rightarrow$")
plt.xlabel("z $\\rightarrow$")
plt.loglog(z,abs(B),label="B(z)")
plt.loglog(z,bzfit,label="lstsq fit")
plt.legend()
plt.show()

# Question 11
"""
As expected for a symmetric field, we have got an almost 0 magnetic field along the z axis for a current that varies as 
a function of cos(phi). However, there is not much of a point in finding b and c using lstsq if this is the case.
Note that b in the lstsq gives us the decay rate. I am treating it as such as it is the slope of our approximation in
the loglog scale and it is negative.
Now, I will re solve the question for the static case, i.e k=0. Not many comments have been included hear after,
as it is exactly the same code as above with only current being changed. 
"""

I=(10**7)*cos(phi)

plt.figure(3)
plt.title("Current elements of the loop, which vary as cos(\u03C6)")
plt.scatter(x_loop,y_loop,label="location of elements")
plt.quiver(x_loop,y_loop,dl_x*I,dl_y*I,label = "Current elements")
plt.grid()
plt.legend(loc="upper right")
plt.ylabel("y-axis $\\rightarrow$")
plt.xlabel("x-axis $\\rightarrow$")
plt.show()

# Calc has also been changed to account for current.
def Calc2(l):
    l_x,l_y=rdash[l] # x any y coordinates of a particular element in r'

    # Now, let us calculcate |x-l_x| and |y-l_y|, which can be used to finally find R using R = |r-r'|.
    Rx=abs(rx-l_x)
    Ry=abs(ry-l_y)
    Rz=rz
    Rijkl = sqrt(Rx**2+Ry**2+Rz**2)

    Axl=cos(2*pi*l/100)*dl[l][0]/Rijkl
    Ayl=cos(2*pi*l/100)*dl[l][1]/Rijkl

    return (Axl,Ayl)

Ax,Ay=Calc2(0)

for l in range(1,100):
    dAx,dAy=Calc2(l)
    Ax=Ax+dAx
    Ay=Ay+dAy

B = (Ax[1,0,:]-Ax[1,2,:])/2+(Ay[2,1,:]-Ay[0,1,:])/2

m = c_[log(z),ones(len(z))] # Set m to [log(z),1]
n = log(abs(B)) # Set n = log(B)
p = lstsq(m,n,rcond=None)[0]
b = p[0]
c = exp(p[1])
bzfit = c*(z**b)
print("For a static current that varies as a function of cos(phi), the output is")
print("Value of b: ",b)
print("Value of c: ",c)

plt.figure(4)
plt.title("B as a function of z, with the best lstsq fit")
plt.grid()
plt.ylabel("B(z) $\\rightarrow$")
plt.xlabel("z $\\rightarrow$")
plt.loglog(z,abs(B),label="B(z)")
plt.loglog(z,bzfit,label="lstsq fit")
plt.legend()
plt.show()

"""
Let us repeat this process one last time, but with a constant current.
"""

I=(10**7)

plt.figure(5)
plt.title("Current elements of the loop, which is constant.")
plt.scatter(x_loop,y_loop,label="location of elements")
plt.quiver(x_loop,y_loop,dl_x*I,dl_y*I,label = "Current elements")
plt.grid()
plt.legend(loc="upper right")
plt.ylabel("y-axis $\\rightarrow$")
plt.xlabel("x-axis $\\rightarrow$")
plt.show()

# Calc has also been changed to account for current.
def Calc3(l):
    l_x,l_y=rdash[l] # x any y coordinates of a particular element in r'

    # Now, let us calculcate |x-l_x| and |y-l_y|, which can be used to finally find R using R = |r-r'|.
    Rx=abs(rx-l_x)
    Ry=abs(ry-l_y)
    Rz=rz
    Rijkl = sqrt(Rx**2+Ry**2+Rz**2)

    Axl=dl[l][0]/Rijkl
    Ayl=dl[l][1]/Rijkl

    return (Axl,Ayl)

Ax,Ay=Calc3(0)

for l in range(1,100):
    dAx,dAy=Calc3(l)
    Ax=Ax+dAx
    Ay=Ay+dAy

B = (Ax[1,0,:]-Ax[1,2,:])/2+(Ay[2,1,:]-Ay[0,1,:])/2

m = c_[log(z),ones(len(z))] # Set m to [log(z),1]
n = log(abs(B)) # Set n = log(B)
p = lstsq(m,n,rcond=None)[0]
b = p[0]
c = exp(p[1])
bzfit = c*(z**b)
print("For a static current that is constant, the output is")
print("Value of b: ",b)
print("Value of c: ",c)

plt.figure(6)
plt.title("B as a function of z, with the best lstsq fit")
plt.grid()
plt.ylabel("B(z) $\\rightarrow$")
plt.xlabel("z $\\rightarrow$")
plt.loglog(z,abs(B),label="B(z)")
plt.loglog(z,bzfit,label="lstsq fit")
plt.legend()
plt.show()

"""
Finally, we have an appreciable output. The field along z-axis is non zero and the decay rate is -2.82.
"""








