from base_funcs import *
import matplotlib.pyplot as plt
from tqdm import tqdm

#2g
def oppg_2g():
    epsilon = jnp.linspace(2,0,101)
    lengths = [1]
    delta = 0.01
    zeta = 3
    epsN = 101

    greensFunctions = jnp.zeros((epsN, 4, 4), dtype = jnp.complex64)
    xm = 101

    y = jnp.zeros((32,xm))
    for l in lengths:
        x = jnp.linspace(0,l,xm)
        i = 0
        for eps in tqdm(epsilon):
            
            partial_dvec = partial(calc_dvec, eps=eps, delta=delta)
            partial_boundary = partial(calc_boundary_normmetals, zeta=zeta, l=l)

            sol = sp.integrate.solve_bvp(partial_dvec, partial_boundary, x, y, max_nodes = xm)
            y = sol["y"]
            solAtX_2 = sol["y"][:, ((xm+1)//2)]
            greensFunctions[i] = CalculateGreensFunction(solAtX_2)
            i += 1

def oppgave_2j():
    eps = 2
    l = 1
    delta = 0.01
    zeta = 3
    m = 101
    x = jnp.linspace(0, l, m)
    y = jnp.zeros((32,m))
    phiL = phiR = 0

    partial_dvec = partial(calc_dvec, eps=eps, delta=delta)
    partial_boundary = partial(calc_boundary_superconduct_metals, 
                                eps=eps, delta=delta, zeta=zeta, l=l,
                                phiL=phiL, phiR=phiR)


    sol = sp.integrate.solve_bvp(partial_dvec, partial_boundary, x, y, verbose=1)
    greensFunctions = jax.vmap(CalculateGreensFunction, in_axes=1)(sol["y"])
    density = jax.vmap(CalculateDensityOfState)(greensFunctions).squeeze(axis=-1)
    plt.plot(x, density)
    plt.grid()
    plt.show()


def oppgave_2k():

    epsilon = jnp.linspace(2,0,101)
    lengths = jnp.array((0.5,1,2))
    delta = 0.01
    zeta = 3
    epsN = 101
    phiL = phiR = 0

    DOS = jnp.zeros(epsN)
    greensFunctions = jnp.zeros((epsN, 4, 4), dtype = jnp.complex64)
    xm = 101

    fig = plt.figure()


    y = jnp.zeros((32,xm))
    j = 1
    for l in lengths:
        x = jnp.linspace(0,l,xm)
        i = 0
        for eps in tqdm(epsilon):
            partial_dvec = partial(calc_dvec, eps=eps, delta=delta)
            partial_boundary = partial(calc_boundary_superconduct_metals, 
                                       eps=eps, delta=delta, zeta=zeta, l=l,
                                       phiL=phiL, phiR=phiR)

            sol = sp.integrate.solve_bvp(partial_dvec, partial_boundary, x, y, max_nodes = xm, verbose=1)
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


def oppgave_2l():
    epsilon = [2, 1.5, 1, 0.5, 0]
    lengths = [1]
    x, current_plot_vals = calculate_currents(lengths, epsilon)
    for (idx, current) in enumerate(current_plot_vals):
        plt.plot(x, current, label=f'$\\varepsilon = {epsilon[idx]}$')
    
    plt.legend()
    plt.savefig("./output/oppg2l.png")

def oppgave_2m():
    phiLeft = [1]
    phiRight = 0
    epsilon = jnp.linspace(2,0,101)
    lengths = jnp.array([1])

    x, current_plot_vals = calculate_currents(lengths, epsilon, phiLeft, phiRight)
    plt.plot(x, current_plot_vals)
    
    plt.legend()
    plt.savefig("./output/oppg2m.png")

def oppgave_2n():
    phiLeft = jnp.arange(0, jnp.pi, step=jnp.pi/8)
    phiRight = 0
    epsilon = jnp.linspace(0,2,101)
    lengths = jnp.array([1])

    Integrals = integerate_currents(lengths, epsilon, phiLeft, phiRight)
    plt.plot(phiLeft, Integrals, label=f"$\\phi_{{L}} = {phiRight}$")
    plt.legend()
    plt.show()

oppgave_2n()
