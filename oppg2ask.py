import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

def C2x2MatrixToR8Vector(matrix):
    real = np.real(matrix)
    imag = np.imag(matrix)

    return np.concat([real, imag]).flatten()

def R8VectorToC2x2Matrix(vec):
    matrix = np.reshape(vec, (2,2,2))
    matrix = matrix[0] + 1j*matrix[1]

    return matrix

def MergeVectors(m1, m2, m3, m4):
    tmp_arr = np.concat([m1,m2,m3,m4])
    return tmp_arr.flatten()

def SplitVectors(m1):
    return m1.reshape(4, 8)

def flatten_matrices(gm, gm_h, omg, omg_h):
    matrix_list = [gm, gm_h, omg, omg_h]
    real_vectors = [C2x2MatrixToR8Vector(m) for m in matrix_list]
    v = MergeVectors(real_vectors[0], real_vectors[1], real_vectors[2], real_vectors[3])
    return v

def pack_matrices(v):
    real_vectors = SplitVectors(v)
    gm, gm_h, omg, omg_h = [R8VectorToC2x2Matrix(x) for x in real_vectors]
    return gm, gm_h, omg, omg_h



def calc_N(gm, gm_t):
    N = np.linalg.inv(np.identity(2) - np.einsum('ij, jk', gm, gm_t))
    return N

def calc_N_t(gm, gm_t):
    N_t = np.linalg.inv(np.identity(2) - np.einsum('ij, jk', gm_t, gm))
    return N_t

#2d
def calc_dv(v):
    gm, gm_t, omg, omg_t = pack_matrices(v)

    dgm = omg
    dgm_t = omg_t

    N = calc_N(gm, gm_t)
    N_t = calc_N_t(gm, gm_t)

    #TODO: Siden de er komplekskonjugat, mer effektivt å bare bruke np.conjugate ?
    domg = -2j * (eps + 1j*delta) * gm - 2* np.einsum('ij,jk,kl,lm', omg, N_t, gm_t, omg)
    domg_t = -2j * (eps + 1j*delta) * gm_t - 2* np.einsum('ij,jk,kl,lm', omg_t, N, gm, omg_t)

    dv = flatten_matrices(dgm, dgm_t, domg, domg_t)
    return dv

#2e
def calc_dvec(x: np.array, vec : np.array):
    dvec = np.empty_like(vec)
    m = np.size(x)
    for i in range(m):
        vi = vec[:, i]
        dvi = calc_dv(vi)
        dvec[:, i] = dvi
    return dvec

#2f
def calc_boundary_normmetals(v_left,v_right):
    'Forenklet LHS ettersom alle riccati-matrisene er null'
    gm_left, gm_left_t, omg_left, omg_left_t = pack_matrices(v_left)
    gm_right, gm_right_t, omg_right, omg_right_t = pack_matrices(v_right)

    N_L = calc_N(gm_left, gm_left_t)
    N_L_t = calc_N_t(gm_left, gm_left_t)

    N_R = calc_N(gm_right, gm_right_t)
    N_R_t = calc_N_t(gm_right, gm_right_t)

    eq13 = omg_right + np.einsum('ij,jk,kl', 1/(zeta*l) * np.identity(2), N_R, -gm_right)
    eq14 = omg_right_t + np.einsum('ij,jk,kl', 1/(zeta*l) * np.identity(2), N_R_t, -gm_right_t)

    eq15 = omg_left - np.einsum('ij,jk,kl', 1/(zeta*l) * np.identity(2), N_L, -gm_left)
    eq16 = omg_left_t - np.einsum('ij,jk,kl', 1/(zeta*l) * np.identity(2), N_L_t, -gm_left_t)

    final_v = flatten_matrices(eq13, eq14, eq15, eq16)
    return final_v

#2i

def nuPluss():
    return np.arctanh(1/(eps + delta*1j))

def nuMinus():
    return np.arctanh(-1/(eps + delta*1j))

def fetch_nonzero_riccati(phaseL = 0, phaseR = 0):
    plussElement = np.sinh(nuPluss())/(1+np.cosh(nuPluss()))
    minusElement = np.sinh(nuMinus())/(1+np.cosh(nuMinus()))

    MaterialGammaLeft = np.array(((0, plussElement),(minusElement, 0)))*np.exp(phaseL*1j)
    MaterialGammaTildeLeft = np.array(((0, minusElement),(plussElement, 0)))*np.exp(-phaseL*1j)

    MaterialGammaRight = np.array(((0, plussElement),(minusElement, 0)))*np.exp(phaseR*1j)
    MaterialGammaTildeRight = np.array(((0, minusElement),(plussElement, 0)))*np.exp(-phaseR*1j)
    
    return MaterialGammaLeft, MaterialGammaTildeLeft, MaterialGammaRight, MaterialGammaTildeRight

def calc_boundary_superconduct_metals(v_left,v_right):
    gm_left, gm_left_t, omg_left, omg_left_t = pack_matrices(v_left)
    gm_right, gm_right_t, omg_right, omg_right_t = pack_matrices(v_right)

    mat_gamma_left, mat_gamma_t_left, mat_gamma_right, mat_gamma_t_right = fetch_nonzero_riccati(eps, delta)

    N_L = calc_N(gm_left, gm_left_t)
    N_L_t = calc_N_t(gm_left, gm_left_t)

    N_R = calc_N(gm_right, gm_right_t)
    N_R_t = calc_N_t(gm_right, gm_right_t)

    eq13 = omg_left + np.einsum('ij,jk,kl', 1/(zeta*l) * (np.identity(2) - gm_left @ mat_gamma_left), N_L, mat_gamma_left -gm_left)
    eq14 = omg_left_t + np.einsum('ij,jk,kl', 1/(zeta*l) * (np.identity(2) - gm_left_t @ mat_gamma_left), N_L_t, mat_gamma_t_left -gm_left_t)

    eq15 = omg_right - np.einsum('ij,jk,kl', 1/(zeta*l) * (np.identity(2) - gm_right @ mat_gamma_right), N_R, mat_gamma_right -gm_right)
    eq16 = omg_right_t - np.einsum('ij,jk,kl', 1/(zeta*l) * (np.identity(2) - gm_right_t @ mat_gamma_right), N_R_t, mat_gamma_t_right -gm_right_t)

    final_v = flatten_matrices(eq13, eq14, eq15, eq16)
    return final_v


def CalculateGreensFunction(sol):

    gamma, gammaTilde, omega, omegaTilde = pack_matrices(sol)

    N = calc_N(gamma, gammaTilde)
    NTilde = calc_N_t(gamma, gammaTilde)

    topLeft = 2*N - np.identity(2)
    topRight = 2*N @ gamma
    bottomLeft = -2*NTilde @ gammaTilde
    bottomRight = -2*NTilde + np.identity(2)

    greens = np.concatenate ((np.concatenate((topLeft ,topRight), axis=1), np.concatenate((bottomLeft, bottomRight), axis = 1)), axis = 0)

    return greens

def CalculateDensityOfStates(greensFunctions): #greensfunctions is assumed to be (m x 4 x 4) tensor
    roHat3 = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    D = np.zeros(greensFunctions.shape[0])
    for i in range(greensFunctions.shape[0]):
        D[i] = np.real(np.einsum("ii",roHat3 @ greensFunctions[i])) / 4
    return D



def oppgave_2k():

    epsilon = np.linspace(2,0,101)
    lengths = np.array((0.5,1,2))
    
    epsN = 101

    DOS = np.zeros(epsN)
    greensFunctions = np.zeros((epsN, 4, 4), dtype = np.complex128)
    xm = 101

    fig = plt.figure()


    y = np.zeros((32,xm))
    j = 1
    global l
    for l in lengths:
        x = np.linspace(0,l,xm)
        i = 0
        global eps
        global m
        for eps in tqdm(epsilon):
            global delta
            global zeta
            
            delta = 0.01
            zeta = 3

            delta = 0.01
            l = 1
            m = 101


            sol = sp.integrate.solve_bvp(calc_dvec, calc_boundary_superconduct_metals, x, y, max_nodes = xm)
            y = sol["y"]
            solAtX_2 = sol["y"][:, ((xm+1)//2)]
            greensFunctions[i] = CalculateGreensFunction(solAtX_2)
            i += 1

        
        ax = fig.add_subplot(1,3,j)
        ax.grid()
        ax.set_title(r"$l =$" + str(float(l)))
        ax.set_xlabel("Energy")
        ax.set_ylabel("DOS")

        DOS = CalculateDensityOfStates(greensFunctions)

        ax.plot(epsilon, DOS)
        j += 1
    
    fig.suptitle(r"DOS at $x = \frac{l}{2}$")
    

    fig.tight_layout()
    fig.savefig("./output/default_oppg_k.png")
    fig.show()

def calc_dgreen(sol):
    greens = np.zeros(( 4, 4), dtype = np.complex128)

    gamma, gammaTilde, omega, omegaTilde = pack_matrices(sol)

    N = calc_N(gamma, gammaTilde)
    NTilde = calc_N_t(gamma, gammaTilde)

    dN = np.einsum('ij,jk,kl', N, omega @ gammaTilde + gammaTilde @ omega, N)
    dNTilde = np.einsum('ij,jk,kl', NTilde, omegaTilde @ gamma + gamma @ omegaTilde, NTilde)

    topLeft = dN
    topRight = N @ omega + dN @ gamma
    bottomLeft = 2*NTilde @ omegaTilde - dNTilde @ gammaTilde
    bottomRight = -dNTilde

    dgreens = np.concatenate ((np.concatenate((topLeft ,topRight), axis=1), np.concatenate((bottomLeft, bottomRight), axis = 1)), axis = 0)
    return dgreens


def oppgave_2l():
    
    epsilon = [2, 1.5, 1, 0.5, 0]
    lengths = np.array([1])
    
    epsN = 101

    DOS = np.zeros(epsN)
    greensFunctions = np.zeros((epsN, 4, 4), dtype = np.complex128)
    dgreensFunctions = np.zeros((epsN, 4, 4), dtype= np.complex128)
    current = np.zeros(epsN, dtype=np.float64)
    current_plot_vals = []
    roHat3 = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    xm = 101


    y = np.zeros((32,xm))
    global l
    for l in lengths:
        x = np.linspace(0,l,xm)
        i = 0
        global eps
        global m
        for eps in tqdm(epsilon):
            global delta
            global zeta
            
            zeta = 3
            delta = 0.01
            l = 1
            m = 101


            sol = sp.integrate.solve_bvp(calc_dvec, calc_boundary_superconduct_metals, x, y, max_nodes = xm)
            y = sol["y"]
            solAtX_2 = sol["y"][:, ((xm+1)//2)]
            for k in range(np.shape(y)[1]):
                greensFunctions[k] = CalculateGreensFunction(solAtX_2)
                dgreensFunctions[k] = calc_dgreen(solAtX_2)
                current[k] = np.real(np.einsum('ij, ji ->',  roHat3, greensFunctions[k] @ dgreensFunctions[i] - dgreensFunctions[k] @ greensFunctions[k]))
            current_plot_vals.append(current)
            
            i += 1

    for (idx, current) in enumerate(current_plot_vals):
        plt.plot(x, current, label=f'$\\varepsilon = {epsilon[idx]}$')
    
    plt.legend()
    plt.savefig("./output/oppg2l.png")


oppgave_2l()