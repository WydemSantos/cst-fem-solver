# src/assembly.py
import numpy as np
from scipy.sparse import lil_matrix
from src.elements import compute_B, compute_D, compute_k


def assemble_K(nodes, elements, D, thickness):
    """
    Assemble the global stiffness matrix from element contributions.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    elements : ndarray, shape (n_elems, 3)
    D : ndarray, shape (3, 3)
    thickness : float

    Returns
    -------
    K : scipy.sparse.csr_matrix, shape (n_dof, n_dof)
        where n_dof = 2 * n_nodes.

    Notes
    -----
    Use scipy.sparse.lil_matrix for assembly (efficient for incremental insertion),
    then convert to CSR before returning (efficient for solving).
    The DOF ordering convention is: [u0, v0, u1, v1, ...] -- interleaved.
    For each element, extract the 6 global DOF indices from the 3 node indices,
    then scatter the 6x6 element stiffness into the global matrix.
    """
    n_nodes = nodes.shape[0]
    n_dof = 2 * n_nodes

    #Creates the matrix
    K = lil_matrix((n_dof, n_dof), dtype=float)

    for conn in elements: #For each element 
        coords = nodes[conn]              # Gets the number of the element's nodes 
        ke = compute_k(coords, D, thickness)   # Compute the stiffness matrix of the element

        n1, n2, n3 = conn    #The numbers of each node
        
        #DOF numbering
        dofs = np.array([
            2*n1, 2*n1 + 1,
            2*n2, 2*n2 + 1,
            2*n3, 2*n3 + 1
        ], dtype=int)


        # (i,j) Local coordinates / (I,J) Global coordinates
        for i in range(6):
            I = dofs[i]
            for j in range(6):
                J = dofs[j]
                K[I, J] += ke[i, j]

    return K.tocsr()
    

def assemble_R_parabolic_shear(nodes, loaded_nodes, P, h):
    """
    Assemble the global load vector for a parabolic shear traction at the cantilever tip.

    The traction distribution along the tip edge (x = L) is:
        t_y(y) = (3P / 2h) * (1 - 4y^2/h^2)

    This must be integrated consistently using shape functions along each edge
    segment between adjacent loaded nodes (not applied as point loads).

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    loaded_nodes : list of int
        Node indices along the loaded edge (x = L), to be sorted by y-coordinate.
    P : float
        Total applied tip shear force.
    h : float
        Plate height.

    Returns
    -------
    R : ndarray, shape (n_dof,)

    Notes
    -----
    For each edge segment between two adjacent loaded nodes, use at least 2-point
    Gauss quadrature to integrate t_y(y) * N_a(y) dy and t_y(y) * N_b(y) dy,
    where N_a and N_b are the linear (1D) shape functions along the edge.
    Verify: R.sum() should equal P (global force equilibrium).
    """

    n_nodes = nodes.shape[0]
    n_dof = 2 * n_nodes
    R = np.zeros(n_dof, dtype=float)  #Load vector

    # Sort the nodes accordingly to their y coordinate
    loaded_nodes = sorted(loaded_nodes, key=lambda n: nodes[n, 1])

    # Gauss 2 points
    gauss_xi = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    gauss_w = [1.0, 1.0]

    def traction_y(y): #Traction expression
        return (3.0 * P / (2.0 * h)) * (1.0 - 4.0 * y**2 / h**2)

    for i in range(len(loaded_nodes) - 1): #For each segment 
        na = loaded_nodes[i]  #First node of the segment
        nb = loaded_nodes[i + 1] #Second node of the segment
        
        #Coordinates
        xa, ya = nodes[na]  
        xb, yb = nodes[nb]

        #Considering the possibility of a non-vertical segment
        edge_length = np.sqrt((xb - xa)**2 + (yb - ya)**2)

        #Jacobian
        J = edge_length / 2.0

        re = np.zeros(4, dtype=float) #Local force vector (zeros)

        for xi, w in zip(gauss_xi, gauss_w): #For each Gauss point
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)

            y = N1 * ya + N2 * yb
            ty = traction_y(y)

            #N vector
            N = np.array([0.0, N1, 0.0, N2], dtype=float)
            re += N * ty * J * w  #Local force vector

        dofs = np.array([2*na, 2*na + 1, 2*nb, 2*nb + 1], dtype=int) #Segment's DOFs
        R[dofs] += re #Adding the contribution of each segment

    return R

def assemble_R_uniform_tension(nodes, loaded_nodes, sigma_inf, thickness):
    """
    Assemble the global load vector for uniform tension applied
    to a set of boundary nodes (used for the plate-with-hole problem).

    The traction is: t_x = sigma_inf applied along the loaded edge.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    loaded_nodes : list of int
    sigma_inf : float
    thickness : float

    Returns
    -------
    R : ndarray, shape (n_dof,)
    """
    n_nodes = nodes.shape[0]
    n_dof = 2 * n_nodes
    R = np.zeros(n_dof, dtype=float) #Global force vector

    # Sort the nodes accordingly to their y coordinate
    loaded_nodes = sorted(loaded_nodes, key=lambda n: nodes[n, 1])

    # 2 Gauss points
    gauss_xi = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    gauss_w = [1.0, 1.0]

    for i in range(len(loaded_nodes) - 1): #For each segment
        na = loaded_nodes[i]  #Start node
        nb = loaded_nodes[i + 1] #End node

        #Coordinates
        xa, ya = nodes[na]
        xb, yb = nodes[nb]

        #Segment length considering non-vertical segment
        edge_length = np.sqrt((xb - xa)**2 + (yb - ya)**2)
        #Jacobian
        J = edge_length / 2.0
        
        #Initializing local force vector
        re = np.zeros(4, dtype=float)

        for xi, w in zip(gauss_xi, gauss_w):
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)

            #N vector
            N = np.array([N1, 0.0, N2, 0.0], dtype=float)
            re += N * sigma_inf * thickness * J * w
        
        #Segment's DOFs
        dofs = np.array([2*na, 2*na + 1, 2*nb, 2*nb + 1], dtype=int)
        R[dofs] += re #Adding each segment contribuition

    return R
