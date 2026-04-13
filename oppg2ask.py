import numpy as np
import scipy as sp

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
def calc_dv(v, eps):
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
        dvi = calc_dv(vi, eps)
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

    eq13 = omg_right + np.einsum('ij,jk,kl', 1/(zeta*eps) * np.identity(2), N_R, -gm_right)
    eq14 = omg_right_t + np.einsum('ij,jk,kl', 1/(zeta*eps) * np.identity(2), N_R_t, -gm_right_t)

    eq15 = omg_left + np.einsum('ij,jk,kl', 1/(zeta*eps) * np.identity(2), N_L, -gm_left)
    eq16 = omg_left_t + np.einsum('ij,jk,kl', 1/(zeta*eps) * np.identity(2), N_L_t, -gm_left_t)

    final_v = flatten_matrices(eq13, eq14, eq15, eq16)
    return final_v


delta = 0.01
zeta = 3

l = 1
eps = 1
m = 101

x = np.linspace(0,l,m)
y = np.zeros((32,m))


print(sp.integrate.solve_bvp(calc_dvec, calc_boundary_normmetals, x, y))