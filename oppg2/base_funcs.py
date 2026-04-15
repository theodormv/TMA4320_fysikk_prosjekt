import scipy as sp
from functools import partial
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

#2a
@jax.jit
def C2x2MatrixToR8Vector(matrix):
    real = jnp.real(matrix)
    imag = jnp.imag(matrix)

    return jnp.concat([real, imag]).flatten()

@jax.jit
def R8VectorToC2x2Matrix(vec):
    matrix = jnp.reshape(vec, (2,2,2))
    matrix = matrix[0] + 1j*matrix[1]

    return matrix

#2b
@jax.jit
def MergeVectors(m1, m2, m3, m4):
    tmp_arr = jnp.concat([m1,m2,m3,m4])
    return tmp_arr.flatten()

@jax.jit
def SplitVectors(m1):
    return m1.reshape(4, 8)


@jax.jit
def flatten_matrices(gm, gm_h, omg, omg_h):
    matrix_list = [gm, gm_h, omg, omg_h]
    real_vectors = [C2x2MatrixToR8Vector(m) for m in matrix_list]
    v = MergeVectors(real_vectors[0], real_vectors[1], real_vectors[2], real_vectors[3])
    return v

#2c
@jax.jit
def pack_matrices(v):
    real_vectors = SplitVectors(v)
    gm, gm_h, omg, omg_h = [R8VectorToC2x2Matrix(x) for x in real_vectors]
    return gm, gm_h, omg, omg_h

#2d
@jax.jit
def calc_N(gm, gm_t):
    N = jnp.linalg.inv(jnp.identity(2) - jnp.einsum('ij, jk', gm, gm_t))
    return N

@jax.jit
def calc_N_t(gm, gm_t):
    N_t = jnp.linalg.inv(jnp.identity(2) - jnp.einsum('ij, jk', gm_t, gm))
    return N_t

@jax.jit
def calc_dv(v, eps, delta):
    gm, gm_t, omg, omg_t = pack_matrices(v)

    dgm = omg
    dgm_t = omg_t

    N = calc_N(gm, gm_t)
    N_t = calc_N_t(gm, gm_t)

    domg = -2j * (eps + 1j*delta) * gm - 2* jnp.einsum('ij,jk,kl,lm', omg, N_t, gm_t, omg)
    domg_t = -2j * (eps + 1j*delta) * gm_t - 2* jnp.einsum('ij,jk,kl,lm', omg_t, N, gm, omg_t)

    dv = flatten_matrices(dgm, dgm_t, domg, domg_t)
    return dv

#2e
@jax.jit
def calc_dvec(x: jnp.array, vec : jnp.array, eps, delta):
    m = jnp.size(x)
    for i in range(m):
        vi = vec[:, i]
        dvi = calc_dv(vi, eps, delta).reshape(-1,1)
        if i == 0:
            dvec = dvi
        else:
            dvec = jnp.concat([dvec, dvi], axis=1)
    return dvec

#2f
@jax.jit
def calc_boundary_normmetals(v_left,v_right, zeta, l):
    'Forenklet LHS ettersom alle riccati-matrisene er null'
    gm_left, gm_left_t, omg_left, omg_left_t = pack_matrices(v_left)
    gm_right, gm_right_t, omg_right, omg_right_t = pack_matrices(v_right)

    N_L = calc_N(gm_left, gm_left_t)
    N_L_t = calc_N_t(gm_left, gm_left_t)

    N_R = calc_N(gm_right, gm_right_t)
    N_R_t = calc_N_t(gm_right, gm_right_t)

    eq13 = omg_right + jnp.einsum('ij,jk,kl', 1/(zeta*l) * jnp.identity(2), N_R, -gm_right)
    eq14 = omg_right_t + jnp.einsum('ij,jk,kl', 1/(zeta*l) * jnp.identity(2), N_R_t, -gm_right_t)

    eq15 = omg_left - jnp.einsum('ij,jk,kl', 1/(zeta*l) * jnp.identity(2), N_L, -gm_left)
    eq16 = omg_left_t - jnp.einsum('ij,jk,kl', 1/(zeta*l) * jnp.identity(2), N_L_t, -gm_left_t)

    final_v = flatten_matrices(eq13, eq14, eq15, eq16)
    return final_v

@jax.jit
def nuPluss(eps, delta):
    return jnp.arctanh(1/(eps + delta*1j))

@jax.jit
def nuMinus(eps, delta):
    return jnp.arctanh(-1/(eps + delta*1j))

@jax.jit
def fetch_nonzero_riccati(eps, delta, phaseL = 0, phaseR = 0):
    plussElement = jnp.sinh(nuPluss(eps, delta))/(1+jnp.cosh(nuPluss(eps, delta)))
    minusElement = jnp.sinh(nuMinus(eps, delta))/(1+jnp.cosh(nuMinus(eps, delta)))

    MaterialGammaLeft = jnp.array(((0, plussElement),(minusElement, 0)))*jnp.exp(phaseL*1j)
    MaterialGammaTildeLeft = jnp.array(((0, minusElement),(plussElement, 0)))*jnp.exp(-phaseL*1j)

    MaterialGammaRight = jnp.array(((0, plussElement),(minusElement, 0)))*jnp.exp(phaseR*1j)
    MaterialGammaTildeRight = jnp.array(((0, minusElement),(plussElement, 0)))*jnp.exp(-phaseR*1j)
    
    return MaterialGammaLeft, MaterialGammaTildeLeft, MaterialGammaRight, MaterialGammaTildeRight

@jax.jit
def calc_boundary_superconduct_metals(v_left,v_right, eps, delta, zeta, l, phiL, phiR):
    gm_left, gm_left_t, omg_left, omg_left_t = pack_matrices(v_left)
    gm_right, gm_right_t, omg_right, omg_right_t = pack_matrices(v_right)

    mat_gamma_left, mat_gamma_t_left, mat_gamma_right, mat_gamma_t_right = fetch_nonzero_riccati(eps, delta, phiL, phiR)

    N_L = calc_N(mat_gamma_left, mat_gamma_t_left)
    N_L_t = calc_N_t(mat_gamma_t_left, mat_gamma_left)

    N_R = calc_N(mat_gamma_right, mat_gamma_t_right)
    N_R_t = calc_N_t(mat_gamma_t_right, mat_gamma_right)

    eq13 = omg_left + jnp.einsum('ij,jk,kl', 1/(zeta*l) * (jnp.identity(2) - gm_left @ mat_gamma_left), N_L, mat_gamma_left -gm_left)
    eq14 = omg_left_t + jnp.einsum('ij,jk,kl', 1/(zeta*l) * (jnp.identity(2) - gm_left_t @ mat_gamma_left), N_L_t, mat_gamma_t_left -gm_left_t)

    eq15 = omg_right - jnp.einsum('ij,jk,kl', 1/(zeta*l) * (jnp.identity(2) - gm_right @ mat_gamma_right), N_R, mat_gamma_right -gm_right)
    eq16 = omg_right_t - jnp.einsum('ij,jk,kl', 1/(zeta*l) * (jnp.identity(2) - gm_right_t @ mat_gamma_right), N_R_t, mat_gamma_t_right -gm_right_t)

    final_v = flatten_matrices(eq13, eq14, eq15, eq16)
    return final_v

@jax.jit
def CalculateGreensFunction(sol):

    gamma, gammaTilde, omega, omegaTilde = pack_matrices(sol)

    N = calc_N(gamma, gammaTilde)
    NTilde = calc_N_t(gamma, gammaTilde)

    topLeft = 2*N - jnp.identity(2)
    topRight = 2*N @ gamma
    bottomLeft = -2*NTilde @ gammaTilde
    bottomRight = -2*NTilde + jnp.identity(2)

    greens = jnp.concatenate ((jnp.concatenate((topLeft ,topRight), axis=1), jnp.concatenate((bottomLeft, bottomRight), axis = 1)), axis = 0)

    return greens


'''def CalculateGreensFunctions(sol):
    greens = jnp.zeros((sol.shape[1], 4, 4), dtype = jnp.complex128)

    for i in range(sol.shape[1]):
        gamma, gammaThilde, omega, omegaThilde = pack_matrices(sol[:,i])

        N = calc_N(gamma, gammaThilde)
        NThilde = calc_N_t(gamma, gammaThilde)

        topLeft = 2*N - jnp.identity(2)
        topRight = 2*N @ gamma
        bottomLeft = -2*NThilde @ gammaThilde
        bottomRight = -2*NThilde + jnp.identity(2)

        greens[i] = jnp.concatenate ((jnp.concatenate((topLeft ,topRight), axis=1), jnp.concatenate((bottomLeft, bottomRight), axis = 1)), axis = 0)

    return greens'''
@jax.jit
def calc_dgreen(sol):
    greens = jnp.zeros(( 4, 4), dtype = jnp.complex128)

    gamma, gammaTilde, omega, omegaTilde = pack_matrices(sol)

    N = calc_N(gamma, gammaTilde)
    NTilde = calc_N_t(gamma, gammaTilde)

    dN = jnp.einsum('ij,jk,kl', N, omega @ gammaTilde + gammaTilde @ omega, N)
    dNTilde = jnp.einsum('ij,jk,kl', NTilde, omegaTilde @ gamma + gamma @ omegaTilde, NTilde)

    topLeft = dN
    topRight = N @ omega + dN @ gamma
    bottomLeft = 2*NTilde @ omegaTilde - dNTilde @ gammaTilde
    bottomRight = -dNTilde

    dgreens = jnp.concatenate ((jnp.concatenate((topLeft ,topRight), axis=1), jnp.concatenate((bottomLeft, bottomRight), axis = 1)), axis = 0)
    return dgreens

@jax.jit
def CalculateDensityOfStates(greensFunctions): #greensfunctions is assumed to be (m x 4 x 4) tensor
    roHat3 = jnp.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    D = jnp.zeros(greensFunctions.shape[0])
    for i in range(greensFunctions.shape[0]):
        temporary_density = jnp.real(jnp.einsum("ii",roHat3 @ greensFunctions[i])) / 4
        temporary_density = temporary_density.reshape(-1, 1)
        if i == 0:
            D = temporary_density
        else:
            D = jnp.concat([D, temporary_density], axis=1)
    return D


@jax.jit
def CalculateDensityOfState(greensFunction): 
    roHat3 = jnp.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    density = jnp.real(jnp.einsum("ii",roHat3 @ greensFunction)) / 4
    density = density.reshape(-1, 1)
    return density

def calculate_currents(lengths, epsilon, phiLeft, phiRight, all_positions=False):
    epsN = len(epsilon)
    zeta = 3
    delta = 0.01
    l = 1

    currents = []
    roHat3 = jnp.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    xm = epsN
    x = jnp.linspace(0,l,xm)

    phiR = phiRight
    for phiL in phiLeft:
        for l in lengths:
            for eps in epsilon:
                partial_dvec = partial(calc_dvec, eps=eps, delta=delta)
                partial_boundary = partial(calc_boundary_superconduct_metals, 
                                       eps=eps, delta=delta, zeta=zeta, l=l,
                                       phiL=phiL, phiR=phiR)
                
                
                y = jnp.zeros((32,xm))
                sol = sp.integrate.solve_bvp(partial_dvec, partial_boundary, x, y, max_nodes = xm)
                y = sol["y"]

                @jax.jit
                def find_current_single_point(greensFunction, dgreensFunction):
                    current = jnp.real(jnp.einsum('ij, ji ->',  roHat3, greensFunction @ dgreensFunction - dgreensFunction @ greensFunction))
                    return current
                if all_positions:
                    greensFunctions = jax.vmap(CalculateGreensFunction, in_axes=1)(y)
                    dgreensFunctions = jax.vmap(calc_dgreen, in_axes=1)(y)
                    currents.append(jax.vmap(find_current_single_point, in_axes=(0,0))(greensFunctions, dgreensFunctions))
                    print(f'All currents has been calculated for eps = {eps :.2f} | Length = {l :.2f} | PhiL = {phiL :.2f}.')
                else:
                    y_singlePoint = sol["y"][:, ((xm+1)//2)]  # X = X_max/2
                    greensFunction = CalculateGreensFunction(y_singlePoint)
                    dgreensFunction = calc_dgreen(y_singlePoint)
                    currents.append(find_current_single_point(greensFunction, dgreensFunction))
                    print(f"Current at x/2 has been calculated for eps = {eps :.2f} | Length = {l :.2f} | PhiL = {phiL :.2f}")           
                
    return x, jnp.array(currents)


def integerate_currents(lengths, epsilon, phiLeft, phiRight):
    Integrals = []
    x, currents = calculate_currents(lengths, epsilon, phiLeft, phiRight)
    current_per_phi = currents.reshape(-1, len(epsilon))
    for current in current_per_phi:
        this_integral = sp.integrate.simpson(current, epsilon)
        Integrals.append(this_integral)
    
    return jnp.array(Integrals)

