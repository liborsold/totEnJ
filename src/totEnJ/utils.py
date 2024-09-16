import numpy as np
import sympy as sp

rounding_precision = 15

def linear_chain_DMI_coeffs_vector(Ns, max_d_order=7):
    """_summary_

    Args:
        Ns (_type_): _description_
        max_d_order (int, optional): _description_. Defaults to 8.
    """
    if type(Ns) is not int:
        print(f"Ns must be an integer, converting {Ns} to {int(Ns)}")
        Ns = int(Ns)
    coeffs = [abs(Ns)] + [round(abs(Ns)*np.sin(m*2*np.pi/Ns), rounding_precision) for m in range(1, max_d_order+1)]
    return coeffs


def coeff_matrix(Ns_array, max_d_order=8):
    """__summary__
    
    Args:
        coeff_matrix (_type_): _description_"""
    coeff_matrix = np.array([linear_chain_DMI_coeffs_vector(Ns, max_d_order=max_d_order) for Ns in Ns_array])
    return coeff_matrix


def diagonalize_coefficient_matrix(M):
    """Having a general real (m-by-n) matrix M, where m<=n, returns matrix D with the first n-by-n block diagonalized, and also the transformation matrix T, such that M = T*D*T^-1.

    Args:
        M (m-by-n matrix of real numbers): _description_
    """
    # ensure that m > n
    assert M.shape[0] <= M.shape[1], "The number of rows of the matrix must be larger than the number of columns!"
    m, n = M.shape
    M_sub = M[:m, :m]

    # ensure that M_sub is invertible
    assert np.linalg.det(M_sub) != 0, "The m-by-m submatrix of the m-by-n matrix (m <= n) is not invertible! Use different set of Ns orders to get linearly independent solutions."

    # compute inversion of M_sub with sympy
    M_sub_inv = sp.Matrix(M_sub).inv()

    M_partially_diagonalized = M_sub_inv @ M

    # M_partially_diagonalized = np.round(M_partially_diagonalized, rounding_precision)
    return M_partially_diagonalized, M_sub_inv


def explain_results_verbose(M_sub_inv, M_partially_diagonalized, Ns_array):
    """_summary_

    Args:
        M_sub_inv (_type_): _description_
        M_partially_diagonalized (_type_): _description_
    """
    calculation_strings = [f"E({Ns:d})" for Ns in Ns_array]
    print(f"Having calculated the DFT total energies for supercells (" + ', '.join([f'{Ns:d}' for Ns in Ns_array]) +
                f"), corresponding to angles (" +  ', '.join([f'{360/Ns:.1f}' for Ns in Ns_array]) + f") degrees, \n\
                we obtain the ground-state energy E0 and the DMI coefficients d_n up to order n_max = {M_partially_diagonalized.shape[1]-1}\n\
                  with the following formulas:\n" + \
                    f"E0 = " + ' + '.join([f'{M_sub_inv[0, n]:.3f} {calculation_strings[n]}' for n in range(len(Ns_array)) if np.round(M_sub_inv[0, n], 8) != 0]) + '\n' + \
                    '\n'.join([' + '.join([f'{M_partially_diagonalized[n, m]:.3f} d_{m:d}' for m in range(1, M_partially_diagonalized.shape[1]) if M_partially_diagonalized[n, m] != 0]) + ' = ' + ' + '.join([f'{M_sub_inv[n, m]:.3f} {calculation_strings[m]}' for m in range(len(Ns_array)) if M_sub_inv[n, m] != 0]) for n in range(1, len(Ns_array))]))


def construct_list_of_atoms(supercell):
    """Construct a tuple contatining atoms positions in a supercell of dimensions supercell.
        Each position is a 3x1 tuple of integers.

    Args:
        supercell (3x1 tuple of integers): e.g. (2, 2, 1) or (4,1,1)
    """
    list_of_atoms = []
    for i in range(supercell[0]):
        for j in range(supercell[1]):
            for k in range(supercell[2]):
                list_of_atoms.append((i, j, k))
    return tuple(list_of_atoms)


def cyclic_perm(a):
    """Return a matrix of cyclic permutations of 'a'. 
        For instance, for a = [1, 2, 3]
            returns
                [[1, 2, 3], 
                 [2, 3, 1], 
                 [3, 1, 2]]

    Args:
        a (_type_): vector

    Returns:
        nxn matrix: matrix of all cyclic permutations
    """
    n = len(a)
    b = [[a[i - j] for i in range(n)] for j in range(n)]
    return np.array(b)


def permute_rows(v, supercell, atoms_positions):
    """_summary_

    Args:
        v (interactions (from manual analysis)): e.g. [2, 0, 4, 0] meaning that (for this chosen order of nearest-neighbors) there is interaction with 2 atoms of the same type, 0 atoms of the next type, 4 atoms of the next type, and 0 atoms of the last type
        supercell (tuple of integers): e.g. (2, 2, 1)
        atoms_positions (list of tuples of integers): e.g. [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]

    Returns:
        permuted matrix M: dimensions len(v) x len(v), each row is a permutation of v  by subtracting one-by-one atom position from all the others
    """

    def get_position_in_list_of_tuples(list_of_tuples, tuple_to_check):
        for i, tup in enumerate(list_of_tuples):
            if tup == tuple_to_check:
                return i
        return None

    v = np.array(v, dtype=type(v[0]))
    M = [v]
    for i in range(1, len(v)):
        new_home_position = np.array(atoms_positions[i])
        # subtract from all
        permutation = []
        for pos in atoms_positions:
            new_pos = np.mod(np.array(pos) - new_home_position, supercell)
            # index of new_pos in atoms_positions
            index = get_position_in_list_of_tuples(atoms_positions, tuple(new_pos))
            permutation.append( index )
        v_permuted = v[permutation]
        M.append(v_permuted)

    return np.array(M)


def drop_zero_rows(M):
    M = np.array(M)
    # remove rows having all zeroes
    M = M[~np.all(M == 0, axis=1)]
    return M


def total_energy_equations(supercell, atom_positions, order_NN, nNN_interactions, states):
    """For a given order 

    Args:
        supercell (_type_): _description_
        order_NN (_type_): _description_
        nNN_interactions (_type_): _description_
        states (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    M = []
    n_atoms = len(atom_positions)
    for state in states:
        cz, *m_vec = state
        m_vec = np.array(m_vec).T

         # ----- definition of Hamiltonian -----
        # construct equation
            # E0 always present
            # Ku only if cz = 1 and its value is -nx (number of atoms present)
        eq = [1, -n_atoms*cz]
            # add all the nNN interactions up to order `order_NN`
        for n in range(1, order_NN+1):
            M_n = permute_rows(nNN_interactions[n], supercell, atom_positions)

            coeff_n = -m_vec.T @ M_n @ m_vec

            # add nth NN coefficient to equation either as 
            # - for spin in-plane add: [coeff_n, 0]
            # - for spin out-of-plane add: [0, coeff_n]
            if cz == 0:
                eq += [coeff_n, 0]
            elif cz == 1:
                eq += [0, coeff_n]
            else:
                raise Exception('cz can only have values 0 (in-plane spins) or 1 (out-of-plane spins)')
            
        M.append(eq)

    # print(M)
    return np.array(M, dtype=np.int64)


def unique_rows_in_matrix(M):
    M = np.array(M)
    n_rows = M.shape[0]
    unique_rows = np.full((n_rows, ), fill_value=True)

    for i in range(n_rows):
        for j in range(i+1, n_rows):
            if np.all(M[i, :] == M[j, :]):
                unique_rows[j] = False

    # drop all rows that are not unique
    M_unique = M[unique_rows, :]

    return M_unique, unique_rows


def equivalent_rows_in_matrix(M):
    """Return list of lists of indices of equivalent rows in matrix M.

    Args:
        M (_type_): numerical matrix

    Returns:
        equivalent rows (list of lists): 
    """
    M = np.array(M)
    n_rows = M.shape[0]
    rows_left = list(range(n_rows))
    equivalent_rows = []
    while rows_left:
        row = rows_left.pop(0)
        equivalent_rows.append([row])
        for row_left in rows_left:
            if np.all(M[row, :] == M[row_left, :]):
                equivalent_rows[-1].append(row_left)
                rows_left.remove(row_left)
    return equivalent_rows


def test_unique_rows_in_matrix():
    M = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    M_unique = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    M_un, unique_rows = unique_rows_in_matrix(M)
    assert np.all(M_un == M_unique)
    assert np.all(unique_rows == np.array([True, False, True]))


def coeff_matrix_Jxy_Jz_K(lattice='hexagonal_2D', supercell=(4,1,1), order_NN=5):
    from itertools import product
    # E_DFT = M * J
    # E_DFT .... column vector of DFT energies calculated for different spin states with a supercell
    # M .... matrix of equations
    # J .... column vector of unknowns
    #    defined as 
    # J = (E0, Ku, J1NN_xy, J1NN_z, J2NN_xy, J2NN_z, ...)

    # for hexagonal 2D lattice
    #    and 3x1x1 supercell
    nNN_interaction_database = {
        'hexagonal_2D': {
            (2,1,1): {1:[2, 4], 2: [2, 4], 3: [6, 0], 4: [4, 8], 5: [2, 4]},
            (3,1,1): {1: [2, 2, 2], 2: [0, 3, 3], 3: [2, 2, 2], 4: [4, 4, 4], 5: [6, 0, 0]},
            (4,1,1): {1:[2, 2, 0, 2], 2: [0, 2, 2, 2], 3: [2, 0, 4, 0], 4: [0, 4, 4, 4], 5: [2, 2, 0, 2]},
            (2,2,1): {1:[0, 2, 2, 2], 2: [0, 2, 2, 2], 3: [6, 0, 0, 0], 4: [0, 4, 4, 4], 5: [0, 2, 2, 2]},
            (3,2,1): {1:[0, 2, 1, 1, 1, 1], 2: [0, 0, 1, 2, 1, 2], 3: [2, 0, 2, 0, 2, 0], 4: [2, 2, 1, 3, 1, 3], 5: [2, 4, 0, 0, 0, 0]},
            }
    }

    nNN_interactions = nNN_interaction_database[lattice][supercell]
    atom_positions = construct_list_of_atoms(supercell)

        #{1: [2, 2, 2], 2: [0, 3, 3], 3: [2, 2, 2], 4: [4, 4, 4], 5: [6, 0, 0]}   # this is a function of lattice type and nx

    #  # factor 2 because considering 2 sets: FM and AFM states
    #   # then one atom's spin is fixed and the rest of (nx-1) atoms can take parallel or antiparallel state to the first atom
    # n_equations = 2 * 2**(nx-1)    # = 8   for  nx = 3

    # do a cartesian product of several vectors
    c_z_vect = [0, 1] # <==> [in_plane, out-of-plane]
    m = [1, -1]  # <==> [parallel, anti-parallel]

    # each state will be (c_z, m1, m2, ...) where c_z in {0, 1} and m in {-1, 1}
        # cz = 0 / 1 ... in-plane / out-of-plane spins
        # m = 1 / -1 ... parallel / anti-parallel
    states = np.array(list(product(c_z_vect, [1], *[m for i in range(len(atom_positions)-1)])))

    # calculate total-energy in terms of unknows for all the states and return as a matrix
    M = total_energy_equations(supercell, atom_positions, order_NN=order_NN, nNN_interactions=nNN_interactions, states=states)
    return M, states


def get_RREF_with_its_transformation_matrix(M):
    # important: must be an array of integers
    # M = np.array([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]], dtype=np.int64)

    n_cols_M = M.shape[1]

    # add identity matrix to the right of M
    I = np.eye(M.shape[0], dtype=np.int64)
    M_I = np.hstack((M, I))

    M_I_rref, ind = sp.Matrix(M_I).rref()

    # rescale all rows so that fractions become integers again
      # NOTE: rescaling is done also for the T matrix (right part of M_I) so it will be accounted for in the final T matrix
    for i in range(M_I_rref.shape[0]):
        denominators = [sp.denom(M_I_rref[i, j]) for j in range((M_I_rref.shape[1]))]
        lcm = np.lcm.reduce(denominators)
        M_I_rref[i, :] *= lcm
        numerators = [sp.numer(M_I_rref[i, j]) for j in range(M_I_rref.shape[1])]
        gcd = np.gcd.reduce(numerators)
        M_I_rref[i, :] /= gcd


    M_rref = M_I_rref[:, :M.shape[1]]
    T = M_I_rref[:, M.shape[1]:]

    # sp.pprint(T)

    # CHECK:
    assert np.all( M_rref == T @ M )
    # convert back to numpy
    M_rref = np.array(M_rref, dtype=np.int64)
    return M_rref, T, ind


def concatenate_all_matrices(M_all):
    """E0 must be treated as a separate unknown for each of the supercells!
    So, concatenate all M in M_all vertically (J1x under J1x, J1z under J1z etc.), but the first column of each M must be in its own final column with zeros for the other matrices, 
    so that E0 is treated as a separate unknown.

    Args:
        M_all (list of numpy matrices of the same dimensions mxn): _description_
    """
    # ensure that all matrices have the same number of columns
       # get the summed number of rows of all matrices
    n_rows = M_all[0].shape[0]
    for i in range(1, len(M_all)):
        n_rows += M_all[i].shape[0]
        assert M_all[i].shape[1] == M_all[0].shape[1], "All matrices must have the same number of columns!"

    n_cols = M_all[0].shape[1]
    n_matrices = len(M_all)
    M = np.zeros((n_rows, n_cols+n_matrices-1), dtype=np.int64)
    i_next = 0
    for i in range(n_matrices):
        n_rows_curr = M_all[i].shape[0]
        M[i_next:i_next+n_rows_curr, i] = M_all[i][:,0]
        M[i_next:i_next+n_rows_curr, n_matrices:] = M_all[i][:,1:]
        i_next += n_rows_curr
    return M


def largest_submatrix_with_nonzero_last_row(M):
    """Returns the largest square matrix from the top left corner of the matrix M
        so that the bottom row is not all zeros.

    Args:
        M (m-by-n matrix): _description_

    Returns:
        m-by-m matrix: largest invertible square matrix from the top left corner of the matrix M
    """
    M = np.array(M)
    m, n = M.shape

    # decrease the square matrix size until the last row is not all zeros
    for i in range(min(n,m), 0, -1):
        # print(M[:i, :])
        if not np.all( M[i-1,:i] == 0):
            return M[:i, :]
    raise Exception('No invertible submatrix found!')


def clean_allzero_columns(T_final, states):
    """CLEAN the all-zero columns of T_final and the corresponding states in 'states'

    Args:
        T_final (sympy.matrices.dense.MutableDenseMatrix): matrix of rational numbers returned by scipy (but uses sympy matrices of rational numbers - to keep the fractions in nice form, not convert to floats)
        states (list of tuples): states used for calculation of total energies
    """
    # indices of all-zero columns by scipy
    all_zero_columns = np.sum(np.array(T_final) == 0, axis=0) == np.array(T_final).shape[0]

    states = [state for i, state in enumerate(states) if not all_zero_columns[i]]

    # drop the nth column of T_final which is sympy.matrices.dense.MutableDenseMatrix
    n_deleted_columns = 0
    for i in range(len(all_zero_columns)):
        if all_zero_columns[i]:
            T_final.col_del(i-n_deleted_columns)
            n_deleted_columns += 1

    return T_final, states


# def neighbor_types_list()


# ----------- TESTS -------------

def test_largest_submatrix_with_nonzero_last_row():
    M = np.array([[3, 2, 1, 0],
                 [0, 0, 5, 6]])
    M_result = np.array([[3, 2, 1, 0]])
    assert np.all( largest_submatrix_with_nonzero_last_row(M) == M_result )

    M = np.array([[3, 2, 1, 0],
                 [1, 0, 5, 6], 
                 [0, 0, 0, 0]])
    M_result = np.array([[3, 2, 1, 0], [1, 0, 5, 6]])
    assert np.all( largest_submatrix_with_nonzero_last_row(M) == M_result )

    M = np.array([[3, 2, 1, 0],
                 [1, 0, 5, 6], 
                 [0, 0, 1, 0]])
    M_result = np.array([[3, 2, 1, 0], [1, 0, 5, 6], [0, 0, 1, 0]])
    assert np.all( largest_submatrix_with_nonzero_last_row(M) == M_result )


def test_permute_rows():
    test_vectors = [(0,1,2,3), \
                    (0,1,2,3)]
    
    test_supercells = [(4, 1, 1), 
                       (2, 2, 1)]
    
    test_atom_positions = [((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)), 
                           ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0))]
    
    test_Matrices = [np.array([[0, 1, 2, 3],[3, 0, 1, 2],[2, 3, 0, 1],[1, 2, 3, 0]]), 
                     np.array([[0, 1, 2, 3],[1, 0, 3, 2],[2, 3, 0, 1],[3, 2, 1, 0]])]

    for v, supercell, atom_positions, M in zip(test_vectors, test_supercells, test_atom_positions, test_Matrices):
        assert np.all( permute_rows(v, supercell, atom_positions) == M )

# test_unique_rows_in_matrix()
# test_largest_submatrix_with_nonzero_last_row()
# test_permute_rows()
