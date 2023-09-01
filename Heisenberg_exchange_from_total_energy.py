from utils import get_RREF_with_its_transformation_matrix, largest_submatrix_with_nonzero_last_row, \
        diagonalize_coefficient_matrix, clean_allzero_columns, unique_rows_in_matrix, concatenate_all_matrices, \
        equivalent_rows_in_matrix, coeff_matrix_Jxy_Jz_K
import numpy as np
import sympy as sp



def main():
    # supercell = (2, 2, 1)
    supercells = [(4,1,1),]


    M_all = []
    states_all = []
    equivalent_states_all = []
    for supercell in supercells:
        M, states = coeff_matrix_Jxy_Jz_K(lattice='hexagonal_2D', supercell=supercell, order_NN=5)
        print('Original Equations:')
        print(M)
        print('\n')

        M, unique_rows = unique_rows_in_matrix(M)
        equivalent_rows = equivalent_rows_in_matrix(M)

        # get equivalent states
        equivalent_states = [states[row] for row in equivalent_rows]

        # get only unique states
        states = states[unique_rows]

        M_all.append(M)
        states_all.append(states)
        equivalent_states_all.append(equivalent_states)

    # concatenate all the matrices and all the states
    M = concatenate_all_matrices(M_all)
    states = [list(state) for state_group in states_all for state in state_group]
    equivalent_states = [state for equivalent_states in equivalent_states_all for state in equivalent_states]

    # get rid of linearly dependent equations

    # cast M to row echelon form and return a matrix of such transformation T, such that M_rref = T * M
    # T tells you how to mix the rows (i.e., the DFT energies) to get the linearly independent solutions
    M_rref, T, ind = get_RREF_with_its_transformation_matrix(M)

    # drop the last row of M_rref that would give all zeros in a square matrix
    M_rref = largest_submatrix_with_nonzero_last_row(M_rref)

    # diagonalize the square submatrix of M_rref
    M_partially_diagonalized, M_sub_inv = diagonalize_coefficient_matrix(M_rref)

    # total T matrix from both Gaussian elimination and diagonalization
    n_final_expressions = M_sub_inv.shape[0]
    T_final = M_sub_inv @ T[:n_final_expressions, :]

    T_final, states = clean_allzero_columns(T_final, states)


    # ======= RESULTS =======
    print('Final equations:')
    print('')

    print('the calculated terms')
    sp.pprint(M_partially_diagonalized)
    print('\n')

    print('summing rules')


    sp.pprint(T_final)
    print(T_final)
    print('\n')

    print('states whose energies are to be summed by rules above to get the calculated terms')
    print(states)

    print('\nequivalent states')
    for entry in equivalent_states:
        print(entry)


if __name__ == '__main__':
    main()