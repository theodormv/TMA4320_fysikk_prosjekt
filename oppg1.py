import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import time

#Runge-Kutta metode
def RK(x_init,x_end,y_init,h0,tol,f,alpha):
    h = [h0]
    h_with_discarded = [h0]
    x = [x_init]
    y_1 = [y_init[0]]
    y_2 = [y_init[1]]
    n = 0
    
    while x_end - x[n] > 0:
        y_cur = np.array([y_1[n], y_2[n]])
        h[n] = min(h[n],x_end - x[n])
    
        k_1= f(x[n], y_cur)
        k_2 = f(x[n] + 1/2*h[n], y_cur + 1/2*h[n]*k_1)
        k_3 = f(x[n] + 3/4*h[n], y_cur + 3/4*h[n]*k_2)
        y_new = y_cur + 1/9*h[n]*(2*k_1 + 3*k_2 + 4*k_3)

        k_4 = f(x[n]+h[n],y_new)
        z_new = y_cur + 1/24*h[n]*(7*k_1 + 6*k_2 + 8*k_3 + 3*k_4)
        
        est = np.sqrt(np.sum((y_new - z_new)**2))
        x.append(x[n] + h[n])
        y_1.append(y_new[0])
        y_2.append(y_new[1])
        h_with_discarded.append(alpha*h[n-1]*(tol/est)**(1/3))
        if est < tol:
            n = n + 1
        h.append(alpha*h[n-1]*(tol/est)**(1/3))

    return x, y_1,y_2, h, len(h), h_with_discarded

#funsjon som brukes i oppgave 1c -1f
def f(x,y):
    return np.array([y[1],-4*np.sin(2*x)])

        
def oppg_c():
    x_init = 0
    x_end = 2*np.pi
    alpha = 0.8
    tol = 10**-7
    y_init = np.array([0.,2.])
    h0 = 0.001
    x, y_1, y_2, h, h_tot, h, h_w_disc = RK(x_init, x_end, y_init, h0, tol, f, alpha)

    
    plt.plot(x, y_1, label = 'y(x)')
    plt.plot(x, y_2, label = r"$y'(x)$")
    plt.xlabel("x")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(x, h, label = 'steglengde')
    plt.xlabel("x")
    plt.ylabel("h")
    plt.grid()
    plt.legend()
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(x, y_1, label = 'y(x)')
    plt.grid()
    ax2 = ax1.twinx()
    ax2.plot(x, h, color = 'g', label = 'steglengde')
    plt.legend()
    plt.show()




def y_anal(x):
    return np.sin(2*np.array(x))

def oppg_d():

    x_init = 0
    x_end = 2*np.pi
    y_init = np.array([0.,2.])
    h0 = 0.001
    alpha = 0.8

    tol_list_len = 1000
    tol_list = np.linspace(10**(-9),10**(-4),tol_list_len)
    error_list =[]

    for tol in tol_list:
        x, y_1_built, y_2, h, h_tot, h_w_disc = RK(x_init, x_end, y_init, h0, tol, f, alpha)

        error_list.append(np.sum(np.absolute(y_anal(x) - y_1_built)))
        
    plt.plot(tol_list, error_list)
    plt.xlabel('tol')
    plt.ylabel('error from analytical')
    plt.grid()
    plt.legend()
    plt.show()

    ############################################################################################
    tol = 10**-7
    alpha_list_len = 10
    list_alpha = np.linspace(0.1,0.9,alpha_list_len)
    h_tot_w_disc_list = []

    for alpha in list_alpha:
        x, y_1, y_2, h, h_tot, h_w_disc = RK(x_init, x_end, y_init, h0, tol, f, alpha)
        h_tot_w_disc_list.append(len(h_w_disc))

    plt.plot(list_alpha,h_tot_w_disc_list)
    plt.ylabel(r'$h_{total}$ [total number of steps]')
    plt.xlabel(r'$\alpha$')
    plt.grid()
    plt.show()

oppg_d()
    

def g(z):
    return z + np.sin(z) + np.cos(z)

def secant(g, z_0, z_1, tol):
    z = [z_0,z_1]
    n = 1
    while np.absolute(z[n-1] - z[n]) >= tol:
        z.append((z[n-1]*g(z[n]) - z[n]*g(z[n-1]))/(g(z[n]) - g(z[n-1])))
        n = n + 1
    return z

def oppg_e():
    z = secant(g,12,10,10**(-7))
    print(f'The root of g is {z[-1]}')
    
    if g(z[-1]) == 0:
        print('z found gives 0')
    elif np.abs(g(z[-1])) < 1e-15:
        print('g(z) aint 0 exactly, but it sure is darn close')

#Boundry value problem solver
def BVP(x_init, x_end, f):
    b_0 = 10
    b_1 = 5
    tol = 10**(-7)
    alpha = 0.8
    tol = 10**-7
    h0 = 0.001
    
    def y_at_end(b):
        x, y_1, y_2, h, h_tot, h_w_disc = RK(x_init, x_end, np.array([0,b]), h0, tol, f, alpha)
        return y_1[-1]
    b_cor = secant(y_at_end, b_0, b_1, tol)
    x_list = []
    y_list = []

    for z in b_cor:
        y_init = np.array([0.,z])
        x, y_1, y_2, h, h_tot, h_w_disc = RK(x_init, x_end, y_init, h0, tol, f, alpha)
        x_list.append(x)
        y_list.append(y_1)

    return  x_list, y_list, b_cor, x_list[-1], x_list[-1]

def oppg_f():
    x_list, y_list, z = BVP(0,2*np.pi,f)
    
    for x in range(len(z)):
        plt.plot(x_list[x],y_list[x], linewidth = 2 - x/10, linestyle = '--', label = f'z[{x}]')
    plt.grid()
    plt.legend()
    plt.show()



#Funksjon som brukes i 1g og 1h
def F(x,y):
    return np.array([y[1],y[0] + np.sin(x)])


def oppg_g():
    x_list, y_list, z = BVP(0,12, F)
    plt.plot(x_list[-1],y_list[-1], label = 'BVP')
    plt.grid()
    plt.show()

    


def bc(ya, yb):
    return np.array([ya[0], yb[0]]) 

def oppg_h():
    t_own_start = time.perf_counter()
    x_list, y_list, z, x_sol, y_sol = BVP(0,12, F)
    t_own_end = time.perf_counter()
    t_own = t_own_end - t_own_start
    
    x = np.linspace(0,12,len(y_sol))
    y_a = np.zeros((2, x.size))

    t_sci_start = time.perf_counter()
    res = solve_bvp(F, bc, x, y_a)
    t_sci_end = time.perf_counter()
    t_sci = t_sci_end - t_sci_start

    y = res['y']

    print(f'built BVP solver: {t_own} \n scipy BVP solver: {t_sci}')

    plt.plot(x, y[0])
    plt.show()

    plt.plot(x_sol, y_list[-1] - y[0])
    plt.grid()
    plt.show()




