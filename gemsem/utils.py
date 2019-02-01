import numpy as np
from math import atan2, sin, cos, sqrt, pi
from datetime import datetime


def L_curve(rho, eta, alpha, truncate=False):
    """
    The functions is related to the regularised least squares problem, and the regularisation parameter.
    This function returns the optimum regularisation parameter value, ie the optimum alpha,
    by a maximum curvature (kappa) estimation.

    Args:
        rho (float ndarray): list of misfit values for models evaluated at different alphas
        eta (float ndarray): list of regularisation model norm values for models evaluated at different alphas
        alpha (int): list of alphas corresponding to the list of misfit and model norms.
        truncate (int): option to not include the first elements of misfit_list and model_norm_list

    Returns:
        corner_alpha (float):
        corner_index (int):
        kappa (float ndarray):

    Converted from:
    Parameter Estimation and Inverse Problems, 2nd edition, 2011
    originally for Matlab by R. Aster, B. Borchers, C. Thurber
    @author: Eigil Y. H. Lippert, Student DTU Space, <s132561@student.dtu.dk>
    """


    # L-curve is defined in log-log space
    x = np.log(rho)
    y = np.log(eta)

    # if a input is list, it still works
    alpha = np.array(alpha)

    # Circumscribed circle simple approximation to curvature (after Roger Stafford)

    # vectors containing three sets of points from the input
    x1 = x[0:-2]
    x2 = x[1:-1]
    x3 = x[2::]
    y1 = y[0:-2]
    y2 = y[1:-1]
    y3 = y[2::]

    # the length of the sides of the triangles in the circumscribed circle
    a = np.sqrt((x3 - x2 )**2 + (y3 - y2 )**2)
    b = np.sqrt((x1 - x3 )**2 + (y1 - y3 )**2)
    c = np.sqrt((x2 - x1 )**2 + (y2 - y1 )**2)

    s = np.copy((a + b + c) / 2)  # semi - perimeter

    # the radius of the circle
    R = (a * b * c) / (4 * np.sqrt(s * (s - a) * (s - b) * (s - c)))

    # the reciprocal of the radius yields the curvature for each estimate for each value
    kappa = np.pad(1 / R, (1, 1), 'constant')  # zero-padded: end-points has no curvature

    if truncate:
        kappa = np.copy(kappa[truncate::])

    corner_index = np.argmax(np.abs(kappa[1:-1]))
    corner_alpha = alpha[corner_index]  # the optimum alpha as found in the L curve corner
    return corner_alpha, corner_index, kappa


def design_SHA(r, theta, phi, nmax):
    """
    Created on Fri Feb  2 09:18:42 2018
    A_r, A_theta, A_phi = design_SHA(r, theta, phi, N)

     Calculates design matrices A_i that connects the vector
     of (Schmidt-normalized) spherical harmonic expansion coefficients,
     x = (g_1^0; g_1^1; h_1^1; g_2^0; g_2^1; h_2^1; ... g_N^N; h_N^N)
     and the magnetic component B_i, where "i" is "r", "theta" or "phi":
         B_i = A_i*x
         Input: r[:]      radius vector (in units of the reference radius a)
         theta[:]  colatitude    (in radians)
         phi[:]    longitude     (in radians)
         N         maximum degree/order

     A_r, A_theta, A_phi = design_SHA(r, theta, phi, N, i_e_flag)
     with i_e_flag = 'int' for internal sources (g_n^m and h_n^m)
                     'ext' for external sources (q_n^m and s_n^m)

    @author: Nils Olsen, DTU Space - <nilos@space.dtu.dk>
     """
    cml = np.zeros((nmax + 1, len(theta))) # cos(m*phi)
    sml = np.zeros((nmax + 1, len(theta))) # sin(m*phi)
    a_r = np.zeros((nmax + 1, len(theta)))
    cml[0 ]= 1

    for m in np.arange(1, nmax + 1):
        cml[m ] =np.cos( m *phi)
        sml[m ] =np.sin( m *phi)
    for n in np.arange(1, nmax + 1):
        a_r[n ] = r**(-( n +2))

    Pnm = _get_Pnm(nmax, theta)
    sinth = Pnm[1][1]

    # construct A_r, A_theta, A_phi
    A_r =     np.zeros((nmax *(nmax +2), len(theta)))
    A_theta = np.zeros((nmax *(nmax +2), len(theta)))
    A_phi =   np.zeros((nmax *(nmax +2), len(theta)))

    l = 0
    for n in np.arange(1, nmax +1):
        for m in np.arange(0, n+ 1):
            A_r[l] = (n + 1.) * Pnm[n][m] * cml[m] * a_r[n]
            A_theta[l] = -Pnm[m][n + 1] * cml[m] * a_r[n]
            A_phi[l] = m * Pnm[n][m] * sml[m] * a_r[n] / sinth
            l = l + 1
            if m > 0:
                A_r[l] = (n + 1.) * Pnm[n][m] * sml[m] * a_r[n]
                A_theta[l] = -Pnm[m][n + 1] * sml[m] * a_r[n]
                A_phi[l] = -m * Pnm[n][m] * cml[m] * a_r[n] / sinth
                l = l + 1

    return A_r.transpose(), A_theta.transpose(), A_phi.transpose()


def spherical_grid(refinement_degree, grid_type="icosahedral", delete_pole_points=True):
    """
    Computes a spherical grid. Different methods for
    computing the grid can be used, here called grid types.
    Returns a grid in co-latitude [0;180] (theta_grid) and longitude (phi_grid) [-180;180] in degrees.
    Note: healpix works but in the geomagnetic field modelling context, it seems to enables models
    to place all the power in the grid points, this will be investigated in later versions.

    The relation between number of grid points and refinement degree is:
         uniform: Npoints = (180/refinement_degree) * (360/refinement_degree - 1)
         icosahedral: Npoints = 2 + 10 * 4 ** refinement_degree
         healpix: Npoints = 12 * 2 ** (2 * refinement_degree)

    Args:
        degree (int): harmonic degree
        grid_type (str): choose between "uniform", "icosahedral"(default) and "healpix" grids.
        delete_pole_points (boolean): Choose to remove grid points at north and south pole, default True

    Returns:
        theta_grid (float ndarray):
        phi_grid (float ndarray):

    @author: Eigil Y. H. Lippert, Student DTU Space, <s132561@student.dtu.dk>
    """

    if grid_type == "icosahedral":
        # compute icosahedral grid
        icos_points, icos_triangs = _define_icosahedral_grid()

        # create base triangle and refine to chosen degree
        fixed_triangle, fixed_hexagon, n_points = _refine_triangle(refinement_degree)

        # get indices of all faces
        indices = _get_indices(fixed_triangle, fixed_hexagon)

        # project onto sphere
        icosahedral_sphere, icos_faces = _project_grid_to_sphere(fixed_triangle, icos_points, icos_triangs, indices)

        theta, phi = icosahedral_sphere[:, 1], icosahedral_sphere[:, 2]

        # convert to degrees
        theta_grid = theta * 180 / np.pi
        phi_grid = phi * 180 / np.pi

        if delete_pole_points:
            pole_idx_n = np.where(np.round(theta_grid, 6) == 0)
            theta_grid = np.delete(theta_grid, pole_idx_n)
            phi_grid = np.delete(phi_grid, pole_idx_n)

            pole_idx_s = np.where(np.round(theta_grid, 6) == 180)
            theta_grid = np.delete(theta_grid, pole_idx_s)
            phi_grid = np.delete(phi_grid, pole_idx_s)

    elif grid_type == "healpix":
        try:
            import healpy as hp
        except ImportError:
            print('WARNING: You do not have the "healpy" installed, see: https://healpy.readthedocs.io/ on how '
                  'to install guide. Creating grid with icosahedral instead')
            return spherical_grid(refinement_degree, grid_type="icosahedral")

        # the healpy nside parameter must be a power of 2
        nside = 2 ** refinement_degree
        m = np.arange(hp.nside2npix(nside))
        theta, phi = hp.pixelfunc.pix2ang(nside, m, nest=False, lonlat=False)

        # convert to degrees
        theta_grid = theta * 180 / np.pi
        phi_grid = phi * 180 / np.pi

        if delete_pole_points:
            # remove pole points
            theta_grid = np.delete(theta_grid, np.where(theta_grid == 0))
            theta_grid = np.delete(theta_grid, np.where(theta_grid == 180.))

    elif grid_type == "uniform":
        # # computes a uniform grid
        theta = np.arange(180 - refinement_degree / 2, 0, - refinement_degree)
        phi = np.arange(-180 + refinement_degree / 2, 180 - refinement_degree / 2, refinement_degree)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        theta_grid = np.reshape(theta_grid, np.size(theta_grid), 1)
        phi_grid = np.reshape(phi_grid, np.size(phi_grid), 1)

        if delete_pole_points:
            # remove pole points
            theta_grid = np.delete(theta_grid, np.where(theta_grid == 0))
            theta_grid = np.delete(theta_grid, np.where(theta_grid == 180.))

    else:
        print('This grid type is not available, please choose either: "icosahedral" or "uniform" or "healpix":')

    return theta_grid, phi_grid


def _get_Pnm(nmax, theta):
    """
    Calculation of associated Legendre functions P(n,m) (Schmidt normalized)
    and its derivative dP(n,m) vrt. theta.

    Input: theta[:] co-latitude (in rad)
           nmax  maximum spherical harmonic degree
    Output: Pnm    ndarray PD with Legendre functions

    P(n,m) ==> Pnm(n,m) and dP(n,m) ==> Pnm(m,n+1)

    @author: Nils Olsen, DTU Space - <nilos@space.dtu.dk>
    """

    costh = np.cos(theta)
    sinth = np.sqrt( 1 - costh**2)
    Pnm = np.zeros((nmax +1, nmax +2, len(theta)))
    Pnm[0][0] = 1
    Pnm[1][1] = sinth

    rootn = np.sqrt(np.arange(0, 2* nmax ** 2 + 1))

    # Recursion relations after Langel "The Main Field" (1987),
    # eq. (27) and Table 2 (p. 256)
    for m in np.arange(0, nmax):
        Pnm_tmp = rootn[m + m + 1] * Pnm[m][m]
        Pnm[m + 1][m] = costh * Pnm_tmp
        if m > 0:
            Pnm[m + 1][m + 1] = sinth * Pnm_tmp / rootn[m + m + 2]
        for n in np.arange(m + 2, nmax + 1):
            d = n * n - m * m
            e = n + n - 1
            Pnm[n][m] = (e * costh * Pnm[n - 1][m] - rootn[d - e] * Pnm[n - 2][m]) / rootn[d]

    Pnm[0][2] = -Pnm[1][1]
    Pnm[1][2] = Pnm[1][0]
    for n in np.arange(2, nmax + 1):
        l = n + 1
        Pnm[0][l] = -np.sqrt(.5 * (n * n + n)) * Pnm[n][1]
        Pnm[1][l] = .5 * (np.sqrt(2. * (n * n + n)) * Pnm[n][0] - np.sqrt((n * n + n - 2.)) * Pnm[n][2])

        for m in np.arange(2, n):
            Pnm[m][l] = .5 * (
                        np.sqrt((n + m) * (n - m + 1.)) * Pnm[n][m - 1] - np.sqrt((n + m + 1.) * (n - m)) * Pnm[n][
                    m + 1])

        Pnm[n][l] = .5 * np.sqrt(2. * n) * Pnm[n][n - 1]

    return Pnm


def _define_icosahedral_grid():
    """
    Creates a 3d icosahedral grid. It has 20 faces and 12 vertices.

    Args:

    Returns:
        icosahedral_points (float list): list of tuples with 3d-coordinates for the vertices
        icosahedral_triangles (int list): list of tuples with indices referring to which points that makes up each triangle

    Originally adopted from a comment from
    https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid by user 'coproc'
    (3/10/2018)
    @author: Stefan Krebs Lange-Willman <s140447@student.dtu.dk>
    @edit: Eigil Y. H. Lippert <s132561@student.dtu.dk>
    """
    # generate base icosahedral grid
    s, c = 2 / sqrt(5), 1 / sqrt(5)

    # get all top points for triangles in icosahedral grid
    top_points = [(0, 0, 1)] + [(s * cos(i * 2 * pi / 5.), s * sin(i * 2 * pi / 5.), c) for i in range(5)]

    # get all bottom points for triangles in icosahedral grid
    bottom_points = [(-x, y, -z) for (x, y, z) in top_points]

    icosahedral_points = top_points + bottom_points

    # get indices pointing out each triangle made up by the icosahedral points
    icosahedral_triangles = [(0, i + 1, (i + 1) % 5 + 1) for i in range(5)] + \
                            [(6, i + 7, (i + 1) % 5 + 7) for i in range(5)] + \
                            [(i + 1, (i + 1) % 5 + 1, (7 - i) % 5 + 7) for i in range(5)] + \
                            [(i + 1, (7 - i) % 5 + 7, (8 - i) % 5 + 7) for i in range(5)]

    return icosahedral_points, icosahedral_triangles


def _refine_triangle(refinement_degree):
    """
    This funcion refines a equilateral unit triangle based on input refinement degree. Based on the idea
    of finding midpoints between neighboring points within each sub triangle
           2
          /\
         /  \
    mp02/____\mp12       Construct new triangles
       /\    /\
      /  \  /  \
     /____\/____\
    0    mp01    1

    Args:
        refinement_degree (int): the degree of triangle refinement

    Returns:
        fixed_triangle (float ndarray): the (x, y)-vertices for refined fixed 2d triangle.
        fixed_hexagon (float ndarray): the (x, y)-vertices for refined fixed 2d hexagon.
        n_points (int): total number of points in grid


    Originally adopted from a comment from
    https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid by user 'coproc'
    (3/10/2018)
    @author: Stefan Krebs Lange-Willman <s140447@student.dtu.dk>
    @edit: Eigil Y. H. Lippert <s132561@student.dtu.dk>
    """

    # define fixed unit triangle
    fixed_triangle = np.array([[[-0.5, 0], [0.5, 0], [0, sqrt(3) / 2]]])

    # duplicate fixed triangle for refinement
    refined_triangle = fixed_triangle

    # Loop refines triangle iteratively into 4 sub triangles per loop
    for i in range(refinement_degree):
        if refinement_degree == 0:  # no refinement
            break

        mp01 = (refined_triangle[:, 0, :] + refined_triangle[:, 1, :]) / 2
        mp02 = (refined_triangle[:, 0, :] + refined_triangle[:, 2, :]) / 2
        mp12 = (refined_triangle[:, 1, :] + refined_triangle[:, 2, :]) / 2

        # collect the points of each of the 4 subtriangles
        t1 = np.array(list(zip(refined_triangle[:, 0, :], mp01, mp02)))
        t2 = np.array(list(zip(refined_triangle[:, 1, :], mp12, mp01)))
        t3 = np.array(list(zip(refined_triangle[:, 2, :], mp02, mp12)))
        t4 = np.array(list(zip(mp01, mp12, mp02)))

        refined_triangle = np.concatenate((t1, t2, t3, t4), axis=0)

    # midpoints of triangles are hexagon vertices, found by:
    fixed_hexagon = np.sum(refined_triangle, axis=1) / 3

    n_points = 3 * 4 ** refinement_degree  # number of points (no. of triangles in refinement * 3)

    fixed_triangle = np.resize(refined_triangle, (n_points, 2))  # resize into a single list

    fixed_triangle = np.unique(fixed_triangle, axis=0)  # removes duplicate coordinates (local xy)

    return fixed_triangle, fixed_hexagon, n_points


def _project_grid_to_sphere(refined_grid, icosahedral_vertices, icosahedral_triangles, indices,
                            spherical_coordinates=True):
    """
    Projects a icosahedral grid (2d) onto a sphere. It works through each triangle (face) in the icosahedral grid and
    creates a grid inside that triangle based on the refined grid.
    If spherical_coords are set to True, the icosahedral vertices coordinates are returned in spherical coordinates
    (radians)

    Args:
        refined_vertices (float ndarray): the (x, y)-vertices for refined fixed 2d triangle.
        refined_indices (int ndarray): array of indices corrsponding to the vertices in the refined triangles
        icosahedral_vertices (float ndarray): the 2d icosahedral grid (x, y)-vertices.
        icosahedral_indices (float ndarray): indices of the vertices of the icoshedral grid triangles
        spherical_coordinates (boolean): return vertices coordinates in spherical (True, default) or cartisian coordinates (False)

    Returns:
        icosahedral_sphere: 3d spherical grid (x, y, z) as (n, 3)-array where n is number of points
        icosahedral_indices: indices pointing to coordinates, one per face.

    Originally adopted from a comment from
    https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid by user 'coproc'
    (3/10/2018)
    @author: Stefan Krebs Lange-Willman <s140447@student.dtu.dk>
    @edit: Eigil Y. H. Lippert <s132561@student.dtu.dk>
    """

    # project triangular grid in 2d onto sphere
    n_vertices = len(refined_grid)  # number of vertices
    n_faces = len(indices)  # number of faces
    icosahedral_sphere = np.zeros((20 * n_vertices, 3))
    icosahedral_faces = np.zeros((20 * n_faces, 3))

    for j in range(20):
        s1 = icosahedral_vertices[icosahedral_triangles[j][0]]
        s2 = icosahedral_vertices[icosahedral_triangles[j][1]]
        s3 = icosahedral_vertices[icosahedral_triangles[j][2]]
        for i in range(n_vertices):
            p = [refined_grid[i, 0], refined_grid[i, 1]]  # 2D point to be mapped to sphere
            icosahedral_sphere[n_vertices * j + i, :] = _map_gridpoint_to_sphere(p, s1, s2, s3)

        icosahedral_faces[(n_faces * j):(n_faces * (j + 1)), :] = indices + j * n_vertices

    # print(icosahedral_sphere)
    icosahedral_faces = icosahedral_faces.astype(int)

    # convert xyz to spherical coordinates
    if spherical_coordinates:
        xy_tmp = icosahedral_sphere[:, 0] ** 2 + icosahedral_sphere[:, 1] ** 2  # enabling swapping arccos with arctan2
        r = np.sqrt(xy_tmp + icosahedral_sphere[:, 1] ** 2)
        phi = np.arctan2(icosahedral_sphere[:, 1], icosahedral_sphere[:, 0])
        theta = np.arctan2(np.sqrt(xy_tmp), icosahedral_sphere[:, 2])
        icosahedral_sphere = np.array([r, theta, phi]).T

    # in order to not count points on the border between triangles several times, we have to round to get all duplicates
    icosahedral_sphere = np.unique(np.round(icosahedral_sphere, decimals=10), axis=0)

    return icosahedral_sphere, icosahedral_faces


def _get_indices(fixed_triangle, fixed_hexagon):
    """
    Takes the points in a icoshedral grid and find the indices corresponding to the vertices.

    Args:
        fixed_triangle (float ndarray): the (x, y)-vertices for refined fixed 2d triangle.
        fixed_hexagon (float ndarray): the (x, y)-vertices for refined fixed 2d hexagon.

    Returns:
        indices (int ndarray): array of indices corrsponding to the vertices in the refined triangles

    Originally adopted from a comment from
    https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid by user 'coproc'
    (3/10/2018)
    @author: Stefan Krebs Lange-Willman <s140447@student.dtu.dk>
    @edit: Eigil Y. H. Lippert <s132561@student.dtu.dk>
    """

    dist_xy = fixed_triangle[:, :, None] - np.transpose(fixed_hexagon)
    dist = (np.sum(dist_xy ** 2, axis=1)).round(10)  # no need to take sqrt
    indices = np.where(dist.transpose() == dist.min())  # indices of minimum values
    indices = np.reshape(indices[1], (-1, 3))

    return indices


def _barycentric_coords(p):
    """
    Computes barycentric coordinates for triangle, given a (x, y) point.
    Triangle given by: (-0.5,0),(0.5,0),(0,sqrt(3)/2)
    Barycentric coordinates describes every point inside a triangle r by the convex combination of its three vertices r1, r2 and r3:
    r = lambda1*r1 + lambda2*r2 + lambda3*r3
    Where lambda1 + lambda2 + lambda3 = 1 and lambda1, lambda2, lambda3 >= 0

    Args:
        p (float, tuple): (x,y)-coordinate of the point to be mapped into barycentric coordinates

    Returns:
        lambda1 (float): barycentric coordinate
        lambda2 (float): barycentric coordinate
        lambda3 (float): barycentric coordinate

    Originally adopted from a comment from
    https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid by user 'coproc'
    (3/10/2018)
    @author: Stefan Krebs Lange-Willman <s140447@student.dtu.dk>
    @edit: Eigil Y. H. Lippert <s132561@student.dtu.dk>
    """
    x, y = p
    lambda3 = y * 2. / sqrt(3.)  # lambda3*sqrt(3)/2 = y
    lambda2 = x + 0.5 * (1 - lambda3)  # 0.5*(lambda2 - lambda1) = x
    lambda1 = 1 - lambda2 - lambda3  # lambda1 + lambda2 + lambda3 = 1
    return lambda1, lambda2, lambda3


def _scalar_product(p1, p2):
    """
    Helping function for computing the arc used in Spherical Linear Interpolation.
    Computes the sum of coordinate points

    Args:
        p1 (float, list of tuples): (x,y)-coordinates in list
        p2 (float, list of tuples): (x,y)-coordinates in list

    Returns:
        sum([p1[i] * p2[i] for i in range(len(p1))])

    Originally adopted from a comment from
    https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid by user 'coproc'
    (3/10/2018)
    @author: Stefan Krebs Lange-Willman <s140447@student.dtu.dk>
    @edit: Eigil Y. H. Lippert <s132561@student.dtu.dk>
    """
    return sum([p1[i] * p2[i] for i in range(len(p1))])


def _slerp(p0, p1, t):
    """
    Computes Spherical Linear Interpolation based on the arc defined by the
    coordinates given by p0 and p1 (around origin)

    Args:
        p0 (float, list of tuples): (x,y)-coordinates in list
        p1 (float, list of tuples): (x,y)-coordinates in list
        t (float): projection weighting based on coordinate point position: t: t=0 -> p0, t=1 -> p1

    Returns:
        tuple([(l0 * p0[i] + l1 * p1[i]) / ang0_sin for i in range(len(p0))]): the projected points

    Originally adopted from a comment from
    https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid by user 'coproc'
    (3/10/2018)
    @author: Stefan Krebs Lange-Willman <s140447@student.dtu.dk>
    @edit: Eigil Y. H. Lippert <s132561@student.dtu.dk>
    """
    # uniform interpolation of arc defined by p0, p1 (around origin)
    # t=0 -> p0, t=1 -> p1
    assert abs(_scalar_product(p0, p0) - _scalar_product(p1, p1)) < 1e-7
    ang0_cos = _scalar_product(p0, p1) / _scalar_product(p0, p0)
    ang0_sin = sqrt(1 - ang0_cos * ang0_cos)
    ang0 = atan2(ang0_sin, ang0_cos)

    # weight project given t: t=0 -> p0, t=1 -> p1
    l0 = sin((1 - t) * ang0)
    l1 = sin(t * ang0)
    return tuple([(l0 * p0[i] + l1 * p1[i]) / ang0_sin for i in range(len(p0))])


def _map_gridpoint_to_sphere(p, s1, s2, s3):
    """
    Computes Spherical Linear Interpolation based on the arc defined by the
    coordinates given by p0 and p1 (around origin)

    Args:
        p (float, list of tuples): (x,y)-coordinates in list
        s1 (float, list): vertices length N
        s2 (float, list): vertices length N
        s3 (float, list): vertices length N

    Returns:
        _slerp(p12, s3, lambda3): The double slerp'ed points

    Originally adopted from a comment from
    https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid by user 'coproc'
    (3/10/2018)
    @author: Stefan Krebs Lange-Willman <s140447@student.dtu.dk>
    @edit: Eigil Y. H. Lippert <s132561@student.dtu.dk>
    """
    # map 2D point p to spherical triangle s1, s2, s3 (3D vectors of equal length)
    lambda1, lambda2, lambda3 = _barycentric_coords(p)
    if abs(lambda3 - 1) < 1e-10:
        return s3
    lambda2s = lambda2 / (lambda1 + lambda2)
    p12 = _slerp(s1, s2, lambda2s)
    return _slerp(p12, s3, lambda3)


def _to_mjd2000(year, month, day, hour=0, minute=0, second=0):
    '''
    converts a date in the input date to Modified Julian Dates(MJD2000), which starts at Jan 1 2000 0h00 and returns the result in decimal days
    '''
    difference_in_days = (datetime(year, month, day, hour, minute, second)
                          - datetime(2000, 1, 1))

    return difference_in_days.days + difference_in_days.seconds / 86400


def _revolutions_to_radians(revolutions):
    '''
    Parameters: revolutions in range; 0 <= revolutions <= 1
    Output: Corresponding revolutions in radians; 0 <= radians <= 2*pi
    '''
    return 2 * np.pi * np.mod(revolutions, 1)


def _sun_mjd2000(mjd2000_time):
    '''
    Solar emphemeris
    input: modified julian day (MJD2000) time
    output:
        right_ascension: right ascension of the sun [radians] in range; 0 <= right_ascension <= 2 pi
        declination: declination of the sun [radians] in range; -pi/2 <= declination <= pi/2

    Notes:
    coordinates are inertial, geocentric, equatorial and true-of-date

    Ported from MATLAB by Eigil Lippert
    Modified by Nils Olsen, DSRI
    '''
    atr = np.pi / 648000

    # time arguments
    djd = mjd2000_time - 0.5;

    t = (djd / 36525) + 1;

    # fundamental arguments (radians)
    gs = _revolutions_to_radians(0.993126 + 0.0027377785 * djd);
    lm = _revolutions_to_radians(0.606434 + 0.03660110129 * djd);
    ls = _revolutions_to_radians(0.779072 + 0.00273790931 * djd);
    g2 = _revolutions_to_radians(0.140023 + 0.00445036173 * djd);
    g4 = _revolutions_to_radians(0.053856 + 0.00145561327 * djd);
    g5 = _revolutions_to_radians(0.056531 + 0.00023080893 * djd);
    rm = _revolutions_to_radians(0.347343 - 0.00014709391 * djd);

    # geocentric, ecliptic longitude of the sun (radians)
    plon = 6910 * np.sin(gs) + 72 * np.sin(2 * gs) - 17 * t * np.sin(gs)
    plon = plon - 7 * np.cos(gs - g5) + 6 * np.sin(lm - ls) + 5 * np.sin(4 * gs - 8 * g4 + 3 * g5)
    plon = plon - 5 * np.cos(2 * (gs - g2)) - 4 * (np.sin(gs - g2) - np.cos(4 * gs - 8 * g4 + 3 * g5));
    plon = plon + 3 * (np.sin(2 * (gs - g2)) - np.sin(g5) - np.sin(2 * (gs - g5)));
    plon = ls + atr * (plon - 17 * np.sin(rm));

    # geocentric distance of the sun (kilometers)
    rsm = 149597870.691 * (1.00014 - 0.01675 * np.cos(gs) - 0.00014 * np.cos(2 * gs));

    # obliquity of the ecliptic (radians)
    obliq = atr * (84428 - 47 * t + 9 * np.cos(rm));

    # geocentric, equatorial right ascension and declination (radians)
    a = np.sin(plon) * np.cos(obliq);
    b = np.cos(plon);

    right_ascension = np.arctan2(a, b);
    declination = np.arcsin(np.sin(obliq) * np.sin(plon));

    return right_ascension, declination