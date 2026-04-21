import scipy as sp
from functools import partial
from tqdm import tqdm
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


#2c
@jax.jit
def convert_matrices_to_vector(m1, m2, m3, m4):
    matrix_list = [m1, m2, m3, m4]
    real_vectors = [C2x2MatrixToR8Vector(m) for m in matrix_list]
    v = MergeVectors(real_vectors[0], real_vectors[1], real_vectors[2], real_vectors[3])
    return v

@jax.jit
def convert_vector_to_matrices(v):
    real_vectors = SplitVectors(v)
    m1, m2, m3, m4 = [R8VectorToC2x2Matrix(x) for x in real_vectors]
    return m1, m2, m3, m4

#2d
@jax.jit
def calc_N(gamma, gamma_tilde):
    N = jnp.linalg.inv(jnp.identity(2) - jnp.einsum('ij, jk', gamma, gamma_tilde))
    return N

@jax.jit
def calc_N_tilde(gamma, gamma_tilde):
    N_tilde = jnp.linalg.inv(jnp.identity(2) - jnp.einsum('ij, jk', gamma_tilde, gamma))
    return N_tilde

@jax.jit
def calc_dv(v, eps, delta):
    'Derivert av v-vektoren (32x1)'
    gamma, gamma_tilde, omega, omega_tilde = convert_vector_to_matrices(v)

    gamma_derivative = omega
    gamma_tilde_derivative = omega_tilde

    N = calc_N(gamma, gamma_tilde)
    N_tilde = calc_N_tilde(gamma, gamma_tilde)

    omega_derivative = -2j * (eps + 1j*delta) * gamma - 2* jnp.einsum('ij,jk,kl,lm', omega, N_tilde, gamma_tilde, omega)
    omega_tilde_derivative = -2j * (eps + 1j*delta) * gamma_tilde - 2* jnp.einsum('ij,jk,kl,lm', omega_tilde, N, gamma, omega_tilde)

    dv = convert_matrices_to_vector(gamma_derivative, gamma_tilde_derivative, omega_derivative, omega_tilde_derivative)
    return dv

#2e
@jax.jit
def calc_dvec(x: jnp.array, vec : jnp.array, eps, delta):
    'Derivert av matrise av v-vektorer'
    dvec = jax.vmap(calc_dv, in_axes=(1, None, None), out_axes=1)(vec, eps, delta)
    return dvec

#2f
@jax.jit
def calc_boundary_normmetals(v_left,v_right, zeta, l):
    'Beregning av grensebetingelser for system av kun normale metaler, dvs alle ricatti-matriser er null'
    gamma_left, gamma_left_tilde, omega_left, omega_left_tilde = convert_vector_to_matrices(v_left)
    gamma_right, gamma_right_tilde, omega_right, omega_right_tilde = convert_vector_to_matrices(v_right)

    N_left = calc_N(gamma_left, gamma_left_tilde)
    N_left_tilde = calc_N_tilde(gamma_left, gamma_left_tilde)

    N_right = calc_N(gamma_right, gamma_right_tilde)
    N_right_tilde = calc_N_tilde(gamma_right, gamma_right_tilde)

    eq13 = omega_right + jnp.einsum('ij,jk,kl', 1/(zeta*l) * jnp.identity(2), N_right, -gamma_right)
    eq14 = omega_right_tilde + jnp.einsum('ij,jk,kl', 1/(zeta*l) * jnp.identity(2), N_right_tilde, -gamma_right_tilde)

    eq15 = omega_left - jnp.einsum('ij,jk,kl', 1/(zeta*l) * jnp.identity(2), N_left, -gamma_left)
    eq16 = omega_left_tilde - jnp.einsum('ij,jk,kl', 1/(zeta*l) * jnp.identity(2), N_left_tilde, -gamma_left_tilde)

    final_v = convert_matrices_to_vector(eq13, eq14, eq15, eq16)
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

    material_gamma_left = jnp.array(((0, plussElement),(minusElement, 0)))*jnp.exp(phaseL*1j)
    material_gamma_tilde_left = jnp.array(((0, minusElement),(plussElement, 0)))*jnp.exp(-phaseL*1j)

    material_gamma_right = jnp.array(((0, plussElement),(minusElement, 0)))*jnp.exp(phaseR*1j)
    material_gamma_tilde_right = jnp.array(((0, minusElement),(plussElement, 0)))*jnp.exp(-phaseR*1j)
    
    return material_gamma_left, material_gamma_tilde_left, material_gamma_right, material_gamma_tilde_right

@jax.jit
def calc_boundary_superconduct_metals(v_left,v_right, eps, delta, zeta, l, phiL, phiR):
    'Beregning av grensebetingelser for ikke-null riccati-matriser'
    gamma_left, gamma_left_tilde, omega_left, omega_left_tilde = convert_vector_to_matrices(v_left)
    gamma_right, gamma_right_tilde, omega_right, omega_right_tilde = convert_vector_to_matrices(v_right)

    material_gamma_left, material_gamma_left_tilde, material_gamma_right, material_gamma_right_tilde = fetch_nonzero_riccati(eps, delta, phiL, phiR)

    N_left = calc_N(material_gamma_left, material_gamma_left_tilde)
    N_left_tilde = calc_N_tilde(material_gamma_left, material_gamma_left_tilde)

    N_right = calc_N(material_gamma_right, material_gamma_right_tilde)
    N_right_tilde = calc_N_tilde(material_gamma_right, material_gamma_right_tilde)

    eq13 = omega_left + jnp.einsum('ij,jk,kl', 1/(zeta*l) * (jnp.identity(2) - gamma_left @ material_gamma_left_tilde), N_left, material_gamma_left -gamma_left)
    eq14 = omega_left_tilde + jnp.einsum('ij,jk,kl', 1/(zeta*l) * (jnp.identity(2) - gamma_left_tilde @ material_gamma_left), N_left_tilde, material_gamma_left_tilde -gamma_left_tilde)

    eq15 = omega_right - jnp.einsum('ij,jk,kl', 1/(zeta*l) * (jnp.identity(2) - gamma_right @ material_gamma_right_tilde), N_right, material_gamma_right -gamma_right)
    eq16 = omega_right_tilde - jnp.einsum('ij,jk,kl', 1/(zeta*l) * (jnp.identity(2) - gamma_right_tilde @ material_gamma_right), N_right_tilde, material_gamma_right_tilde -gamma_right_tilde)

    final_v = convert_matrices_to_vector(eq13, eq14, eq15, eq16)
    return final_v

@jax.jit
def CalculateGreensFunction(solution):

    gamma, gamma_tilde, omega, omega_tilde = convert_vector_to_matrices(solution)

    N = calc_N(gamma, gamma_tilde)
    N_tilde = calc_N_tilde(gamma, gamma_tilde)

    topLeft = 2*N - jnp.identity(2)
    topRight = 2*N @ gamma
    bottomLeft = -2*N_tilde @ gamma_tilde
    bottomRight = -2*N_tilde + jnp.identity(2)

    greens = jnp.block([[topLeft, topRight], [bottomLeft, bottomRight]])
    return greens


@jax.jit
def calculate_greens_function_derivative(solution):
    gamma, gamma_tilde, omega, omega_tilde = convert_vector_to_matrices(solution)

    N = calc_N(gamma, gamma_tilde)
    N_tilde = calc_N_tilde(gamma, gamma_tilde)

    N_derivative = jnp.einsum('ij,jk,kl', N, omega @ gamma_tilde + gamma @ omega_tilde, N)
    N_tilde_derivative = jnp.einsum('ij,jk,kl', N_tilde, omega_tilde @ gamma + gamma_tilde @ omega, N_tilde)

    topLeft = N_derivative
    topRight = N @ omega + N_derivative @ gamma
    bottomLeft = -N_tilde @ omega_tilde - N_tilde_derivative @ gamma_tilde
    bottomRight = -N_tilde_derivative

    dgreens = 2 * jnp.block([[topLeft, topRight], [bottomLeft, bottomRight]])
    return dgreens

@jax.jit
def CalculateDensityOfState(greensFunction): 
    roHat3 = jnp.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    density = jnp.real(jnp.einsum("ii",roHat3 @ greensFunction)) / 4
    density = density.reshape(-1, 1)
    return density

@jax.jit
def find_current_at_single_point(greensFunction, dgreensFunction):
    roHat3 = jnp.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    current = jnp.real(jnp.einsum('ij, ji ->',  roHat3, greensFunction @ dgreensFunction - dgreensFunction @ greensFunction))
    return current

def calculate_currents(lengths, epsilon, phiLeft, phiRight, all_positions=False):
    epsN = len(epsilon)
    zeta = 3
    delta = 0.01
    l = 1

    currents = []
    xm = epsN
    x = jnp.linspace(0,l,xm)
    y = jnp.zeros((32,xm))

    phiR = phiRight
    for phiL in phiLeft:
        for l in lengths:
            print(f"Length: {l} | phi: {phiL}")
            for eps in tqdm(epsilon):
                partial_dvec = partial(calc_dvec, eps=eps, delta=delta)
                partial_boundary = partial(calc_boundary_superconduct_metals, 
                                       eps=eps, delta=delta, zeta=zeta, l=l,
                                       phiL=phiL, phiR=phiR)
                
                
                solution = sp.integrate.solve_bvp(partial_dvec, partial_boundary, x, y, max_nodes = xm, tol=1E-6)
                y = solution["y"]
                if all_positions:
                    greensFunctions = jax.vmap(CalculateGreensFunction, in_axes=1)(y)
                    dgreensFunctions = jax.vmap(calculate_greens_function_derivative, in_axes=1)(y)
                    currents.append(jax.vmap(find_current_at_single_point, in_axes=(0,0))(greensFunctions, dgreensFunctions))
                    #print(f'All currents has been calculated for eps = {eps :.2f} | Length = {l :.2f} | PhiL = {phiL :.2f}.')
                else:
                    y_singlePoint = solution["y"][:, (xm//2)]  # X = l/2
                    greensFunction = CalculateGreensFunction(y_singlePoint)
                    dgreensFunction = calculate_greens_function_derivative(y_singlePoint)
                    currents.append(find_current_at_single_point(greensFunction, dgreensFunction))
                    #print(f"Current at x=l/2 has been calculated for eps = {eps :.2f} | Length = {l :.2f} | PhiL = {phiL :.2f}")           
                
    return x, jnp.array(currents)


def integerate_currents(lengths, epsilon, phiLeft, phiRight):
    Integrals = []
    x, currents = calculate_currents(lengths, epsilon, phiLeft, phiRight)
    current_per_phi = currents.reshape(-1, len(epsilon))
    for current in current_per_phi:
        this_integral = sp.integrate.simpson(current, epsilon)
        Integrals.append(this_integral)
    
    return jnp.array(Integrals)

