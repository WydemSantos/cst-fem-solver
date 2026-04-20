# src/elements.py
import numpy as np


def compute_area(coords):
    """
    Compute the signed area of a triangle.

    Parameters
    ----------
    coords : ndarray, shape (3, 2)
        Node coordinates [[x0,y0], [x1,y1], [x2,y2]].

    Returns
    -------
    float
        Signed area of the triangle (positive if nodes are counter-clockwise).
    """
    # Get the coordinates
    x0,y0 = coords[0]
    x1,y1 = coords[1]
    x2,y2 = coords[2]

    #Compute the area
    area = 1/2*(x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1))
    return area


def compute_B(coords):
    """
    Compute the strain-displacement matrix B for a CST element.

    The CST element has constant strain throughout, so B is constant.
    B maps the 6x1 element displacement vector [u0,v0,u1,v1,u2,v2]
    to the 3x1 strain vector [eps_xx, eps_yy, gamma_xy].

    Parameters
    ----------
    coords : ndarray, shape (3, 2)
        Node coordinates.

    Returns
    -------
    ndarray, shape (3, 6)
        Strain-displacement matrix.

    Notes
    -----
    Derive B from the linear shape functions N_i(x,y) = (a_i + b_i*x + c_i*y) / (2A).
    The coefficients b_i and c_i come from cyclic permutations of the node coordinates.
    Refer to CIVL 537 Lecture Notes, Section 4.
    """
    # Get the coordinates
    x0,y0 = coords[0]
    x1,y1 = coords[1]
    x2,y2 = coords[2]

    b0 = y1 - y2
    b1 = y2 - y0
    b2 = y0 - y1
    
    c0 =  x2 - x1
    c1 =  x0 - x2
    c2 =  x1 - x0

    area = compute_area(coords)

    if area <= 0:
        raise ValueError("Triangle area must be positive. Check node ordering.")
    
    
    B = (1/(2*area))*np.array([[b0, 0, b1, 0, b2, 0],
                              [0, c0, 0, c1, 0, c2],
                              [c0, b0, c1, b1, c2, b2]])

    return B

def compute_D(E, nu, mode="plane_stress"):
    """
    Compute the 3x3 constitutive (material stiffness) matrix D.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    mode : str
        Either "plane_stress" or "plane_strain".

    Returns
    -------
    ndarray, shape (3, 3)
        Constitutive matrix relating stress to strain.

    Notes
    -----
    Plane stress: sigma_zz = 0, eliminate eps_zz from 3D Hooke's law.
    Plane strain: eps_zz = 0, eliminate sigma_zz from 3D Hooke's law.
    The resulting D matrices are different. Implement them based on the course notes.
    """
    if mode == "plane_stress":
        D = (E/(1-nu**2))*np.array([[1.0, nu, 0.0],
                                    [nu, 1.0, 0.0],
                                    [0.0, 0.0, (1-nu)/2]])
    elif mode == "plane_strain":
        D = (E*(1-nu)/((1+nu)*(1-2*nu)))*np.array([[1.0, nu/(1-nu), 0.0],
                                                   [nu/(1-nu), 1.0, 0.0],
                                                   [0.0, 0.0, (1-2*nu)/(2*(1-nu))]])
    else:
        raise ValueError('Selected mode not available. Must be "plane_strain" or "plane_stress"')

    return D


def compute_k(coords, D, thickness):
    """
    Compute the 6x6 element stiffness matrix for a CST element.

    Parameters
    ----------
    coords : ndarray, shape (3, 2)
        Node coordinates.
    D : ndarray, shape (3, 3)
        Constitutive matrix.
    thickness : float
        Element thickness (relevant for plane stress; set to 1 for plane strain).

    Returns
    -------
    ndarray, shape (6, 6)
        Element stiffness matrix: k = t * A * B^T D B.

    Notes
    -----
    Since B is constant over the element, the integral simplifies to
    a single multiplication (no numerical quadrature needed).


    """
    B = compute_B(coords)
    area = compute_area(coords)

    k_e = thickness*area*(B.T @ D @ B)

    return k_e
