import numpy as np

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
    assert np.linalg.det(M_sub) != 0, "The n-by-n submatrix of the m-by-n matrix (m >= n) is not invertible! Use different set of Ns orders to get linearly independent solutions."
    
    M_sub_inv = np.linalg.inv(M_sub)
    M_partially_diagonalized = M_sub_inv @ M

    M_partially_diagonalized = np.round(M_partially_diagonalized, rounding_precision)
    return M_partially_diagonalized, M_sub_inv


def explain_results(M_sub_inv, M_partially_diagonalized, Ns_array):
    """_summary_

    Args:
        M_sub_inv (_type_): _description_
        M_partially_diagonalized (_type_): _description_
    """
    print("ahoj")
