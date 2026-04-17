from base_funcs import *
import matplotlib.pyplot as plt

#2g
def oppgave_2g():
    epsilon = jnp.array([2,1,0])
    lengths = l = 1
    delta = 0.01
    zeta = 3
    xm = 101

    densities = list()

    y = jnp.zeros((32,xm))
    x = jnp.linspace(0,l,xm)
    for eps in tqdm(epsilon):
        
        partial_dvec = partial(calc_dvec, eps=eps, delta=delta)
        partial_boundary = partial(calc_boundary_normmetals, zeta=zeta, l=l)

        sol = sp.integrate.solve_bvp(partial_dvec, partial_boundary, x, y, max_nodes = xm)
        y = sol["y"]
        greensFunctions = jax.vmap(CalculateGreensFunction, in_axes=1)(y)
        densities.append(jax.vmap(CalculateDensityOfState)(greensFunctions).flatten())

    return x, densities

#2h
def oppgave_2h():
    x, densities = oppgave_2g()
    for idx, density in enumerate(densities):
        plt.plot(x, density, label=f"$\\varepsilon = {idx}$")
    
    plt.grid()
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("DOS")
    plt.title("Density of states for interfacing normal metals")
    plt.savefig("./output/2h.png")

def oppgave_2j():
    eps = 2
    l = 1
    delta = 0.01
    zeta = 3
    m = 101
    x = jnp.linspace(0, l, m)
    y = jnp.zeros((32,m))
    phiL = phiR = 0

    partial_dvec     =  partial(calc_dvec, eps=eps, delta=delta)
    partial_boundary =  partial(calc_boundary_superconduct_metals, 
                                eps=eps, delta=delta, zeta=zeta, l=l,
                                phiL=phiL, phiR=phiR)


    sol = sp.integrate.solve_bvp(partial_dvec, partial_boundary, x, y, verbose=1)
    greensFunctions = jax.vmap(CalculateGreensFunction, in_axes=1)(sol["y"])
    density = jax.vmap(CalculateDensityOfState)(greensFunctions).squeeze(axis=-1)
    plt.plot(x, density)
    plt.grid()
    plt.xlabel("")
    plt.title("Density of states for superconductors interfacing normal metal")
    plt.savefig("./output/2j.png")


def oppgave_2k():

    epsilon = jnp.linspace(2,0,101)
    lengths = jnp.array((0.5,1,2))
    delta = 0.01
    zeta = 3
    phiL = phiR = 0
    xm = 101

    y = jnp.zeros((32,xm))
    for l in lengths:
        greensFunctions = list()
        x = jnp.linspace(0,l,xm)
        for eps in tqdm(epsilon):
            partial_dvec = partial(calc_dvec, eps=eps, delta=delta)
            partial_boundary = partial(calc_boundary_superconduct_metals, 
                                       eps=eps, delta=delta, zeta=zeta, l=l,
                                       phiL=phiL, phiR=phiR)

            solution = sp.integrate.solve_bvp(partial_dvec, partial_boundary, x, y, max_nodes = xm)
            y = solution["y"]
            solutionAtX_2 = solution["y"][:, ((xm)//2)]
            greensFunctions.append(CalculateGreensFunction(solutionAtX_2))


        greensFunctions = jnp.array(greensFunctions, dtype=jnp.complex128)
        DOS = jax.vmap(CalculateDensityOfState)(greensFunctions).flatten()
        plt.plot(epsilon, DOS, label = r"$l =$" + str(float(l)))
    
    plt.grid()
    plt.xlabel("Energy")
    plt.ylabel("DOS")
    plt.title(r"DOS at $x = \frac{l}{2}$")
    plt.legend()
    plt.savefig("./output/2k.png")
    

def oppgave_2l():
    epsilon_for_calculation = jnp.linspace(2,0,101)
    epsilon_for_plotting = jnp.array([2, 1.5, 1, 0.5, 0])
    plot_indices = jnp.isin(epsilon_for_calculation, epsilon_for_plotting)
    lengths = [1]
    phiLeft = [0] # Må være liste fordi programmet støtter iterering gjennom flere verdier
    phiRight = 0
    x, current = calculate_currents(lengths, epsilon_for_calculation, phiLeft, phiRight, all_positions=True)
    current_plot_vals = current[plot_indices]
    for (idx, current) in enumerate(current_plot_vals):
        plt.plot(x, current, label=f'$\\varepsilon = {epsilon_for_plotting[idx]}$')
    
    plt.grid()
    plt.title("Current integrand $j(x,\\varepsilon)$")
    plt.xlabel("Position")
    plt.ylabel("$j$")
    plt.ylim(-1E-6, 1E-6)
    plt.legend()
    plt.savefig("./output/2l.png")

def oppgave_2m():
    phiLeft = [1]
    phiRight = 0
    epsilon = jnp.linspace(2,0,101)
    lengths = jnp.array([1])

    x, current_plot_vals = calculate_currents(lengths, epsilon, phiLeft, phiRight)
    plt.plot(epsilon, current_plot_vals)
    
    plt.grid()
    plt.title("Current integrand $j(\\frac{l}{2}, \\varepsilon)$")
    plt.xlabel("Dimensionless energy $\\varepsilon$")
    plt.ylabel("$j$")
    plt.savefig("./output/2m.png")

def oppgave_2n():
    phiLeft = jnp.linspace(0, jnp.pi, 33)
    phiRight = 0
    epsilon = jnp.linspace(2,0,101)
    lengths = jnp.array([1])

    Integrals = integerate_currents(lengths, epsilon, phiLeft, phiRight)
    plt.plot(phiLeft, Integrals)
    plt.xlabel("$\\Delta\\phi$")
    plt.ylabel("$I$")
    plt.title("Current $I$ with varying phase difference")
    plt.grid()
    plt.savefig("./output/2n.png")