import numpy as np
def solver(v0,g,y1):
    #'''the number of points to draw
    # y = k(1-cos(theta))
    # x = k(theta- sin(theta))
    # Point 1:(-b,-v0**2/(2*g)) theta=0
    # Point 2:(0,0) theta0
    # Point 3:(1,1) theta1'''
    a = v0**2.0/(2*g)
    def f(para):
        [k,b,theta0,theta1] = para
        out = np.zeros(4)
        out[0] = k*(1-np.cos(theta0))-a
        out[1] = k*(theta0 - np.sin(theta0))-b
        out[2] = k*(1-np.cos(theta1))-y1-a
        out[3] = k*(theta1 - np.sin(theta1))-1-b
        return np.array(out)
    from scipy.optimize import fsolve
    return fsolve(f,[2.0, 0.5 , 0.5 , 2.0]) #start point

if __name__ == "__main__":
    y=1.0
    v0=0
    g=10
    para = solver(v0,g,y)
    para = np.append(para,v0**2.0/(2*g))
    theta_list = np.linspace(para[2],para[3],3)
    x_exact = para[0]*(theta_list-np.sin(theta_list))-para[1]
    y_exact = para[0]*(1-np.cos(theta_list))-para[4]
    print(para)
    print(theta_list)
    print(x_exact)
    print(y_exact)
