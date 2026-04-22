# src/postprocess.py
import numpy as np
from src.elements import compute_B


def compute_stresses(nodes, elements, u, D):
    """
    Recover element stresses from the displacement solution.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    elements : ndarray, shape (n_elems, 3)
    u : ndarray, shape (n_dof,)
    D : ndarray, shape (3, 3)

    Returns
    -------
    stresses : ndarray, shape (n_elems, 3)
        Stress components [sigma_xx, sigma_yy, tau_xy] at each element centroid.
    """

    n_elems = elements.shape[0]

    #Initialize stresses matrix n_elements X 3 // 3 stresses for each element.
    stresses = np.zeros((n_elems, 3), dtype=float)

    for e, conn in enumerate(elements): #for each element (e: element id/number // conn (nodes id/number))
        coords = nodes[conn]  # shape (3, 2)

        #Get the nodes numbers
        n1, n2, n3 = conn

        #Determine the DOFs
        dofs = np.array([
            2*n1, 2*n1 + 1,
            2*n2, 2*n2 + 1,
            2*n3, 2*n3 + 1
        ], dtype=int)

        #Get the displacements of the element's nodes
        u_e = u[dofs]

        B = compute_B(coords)         # shape (3, 6)
        
        #Compute the strains (B*u) = [epsilon_xx, epsilon_yy, gamma_xy]
        strain = B @ u_e              
        
        #Compute the stresses: (D*B*u) = [sigma_xx, sigma_yy, tau_xy]
        stress = D @ strain           

        #Store stresses of the element
        stresses[e, :] = stress

    return stresses
    


def compute_von_mises(stresses):
    """Von Mises stress from [sigma_xx, sigma_yy, tau_xy] per element."""
    sigma_xx = stresses[:, 0] #Gets the sigma_xx for each element
    sigma_yy = stresses[:, 1] #Gets the sigma_yy for each element
    tau_xy = stresses[:, 2]   #Gets the tau_xy for each element

    #Compute von_misses stress = sqrt(sigma_xx^2 + sigma_yy^2 - sigma_xx*sigma_yy + 3tau_xy)
    sigma_vm = np.sqrt(
        sigma_xx**2
        - sigma_xx * sigma_yy
        + sigma_yy**2
        + 3.0 * tau_xy**2
    )

    return sigma_vm


def strain_energy(K, u):
    """Return 0.5 * u^T K u."""
    return 0.5*float(u.T @ (K @u))
