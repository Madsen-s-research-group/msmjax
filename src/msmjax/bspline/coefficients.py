import itertools

import numpy as np
import scipy.linalg as la
import sympy
from scipy.interpolate import BSpline
from scipy.special import comb

P_INDEX = 6
MU = 3


def old_bspline_basis_equivalent(x, n):
    """Function with signature of scipy.signal.bspline that works with scipy versions >= 1.13

    For backwards compatibility only - you would not want to use this in any production setting!
    """
    # TODO: remove this function (but currently used in build_col_of_B
    knots = np.arange(-(n + 1) / 2, (n + 3) / 2)
    out = BSpline.basis_element(knots)(x)
    out[(x < knots[0]) | (x > knots[-1])] = 0.0
    return out


def calc_Phi(p, u):
    """Evaluate the B-spline denoted by Phi in the article.

    Args:
        p: The degree of the spline in the convention of the article.
        u: The value or array or values where the function should be
            evaluated.

    Returns:
        The values(s) of Phi(u) at u.

    Raises:
        ValueError: If p is not a positive, even integer.
    """
    # TODO: remove this function (but currently used in build_col_of_B
    if p < 2 or p % 2 != 0:
        raise ValueError("p must be a positive even integer")
    return old_bspline_basis_equivalent(u, p - 1)


def build_col_of_B(p):
    """Create a column of B according to the equation before (10).

    This function creates a vector with Phi(u) evaluated at all the integers
    inside its domain. Only the part of the column starting at and including
    the main diagonal is returned.

    Args:
        p: The degree of the spline in the convention of the article.

    Returns:
        A vector of p/2 elements with all the non-zero entries in any column
        of the banded matrix B, starting at and including the main diagonal.
    """
    # TODO: Change this to the JAX implementation of the B-spline used by the
    #  the rest of the code?
    return calc_Phi(p, np.arange(0, p // 2, dtype=np.float64))


def build_col_of_B2(p):
    """Create a column of B**2 based on the column of B.

    Args:
        p: The degree of the spline in the convention of the article.

    Returns:
        A vector with all the non-zero entries in any column of the banded
        matrix B**2, starting at and including the main diagonal.
    """
    full_col_of_B = build_col_of_B(p)
    full_col_of_B = np.concatenate([full_col_of_B[::-1], full_col_of_B[1::]])
    # Obtain the generic row of the square of the infinite Toeplitz matrix through
    # a simple discrete convolution.
    full_col_of_B2 = np.convolve(full_col_of_B, full_col_of_B, mode="full")
    # Return only the non-redundant part.
    return full_col_of_B2[len(full_col_of_B2) // 2 :]


def build_Toeplitz_matrix(p, n_rows):
    """Build an approximation to B**2 as a Toeplitz matrix.

    Args:
        p: The degree of the spline in the convention of the article.
        n_rows: The number of rows of the matrix.

    Returns:
        The matrix as a vector in a format that scipy.linalg.toeplitz can
        understand.

    Raises:
        ValueError: If n_rows is too small for this matrix.
    """
    first_col = build_col_of_B2(p)
    n_first_col = len(first_col)
    if n_rows < n_first_col:
        raise ValueError("not enough rows were requested")
    first_col = np.array(first_col.tolist() + [0.0] * (n_rows - n_first_col))
    return first_col


def build_Toeplitz_system(p, rhs, padding):
    """Build a finite approximation to the linear system defining the cm.

    Args:
        p: The degree of the spline in the convention of the article.
        rhs: The non-zero part of the right hand side, which must have an
            odd length.
        padding: The amount of padding around the minimal Toeplitz
            approximation, as a positive integer.

    Returns:
        A tuple with two vectors. The first one represents the Toeplitz
        coefficient matrix, and the second one is the symmetric Kronecker
        delta independent right-hand side.

    Raises:
        ValueError: If "padding" is negative or zero, or if rhs has an even
            length.
    """
    if len(rhs) % 2 == 0:
        raise ValueError("rhs must have an odd length")
    if padding < 1:
        raise ValueError("padding must be a positive integer")
    minimal = p - 1
    minimal = minimal + (minimal + 1) % 2
    minimal = max(minimal, len(rhs))
    n_rows = minimal + 2 * padding

    center = n_rows // 2
    beginning = center - len(rhs) // 2
    full_rhs = np.zeros(n_rows)
    full_rhs[beginning : beginning + len(rhs)] = rhs

    coefficients = build_Toeplitz_matrix(p, n_rows)
    return (coefficients, full_rhs)


def compute_c_m(p, C_coeffs, mu, rtol=1e-5, atol=1e-8):
    """Compute c_m the coefficients of the approximation to B^(-2).

    The solution is based on the construction of progressively larger
    approximations to an infinite Toeplitz matrix, until the solution
    converges.

    Args:
        p: The degree of the spline in the convention of the article.
        mu: A non-negative integer determining how many terms are calculated.
            The output will contain (2 * mu + 1) coefficients.
        C_coefficients: The right-hand side of the unnumbered equation
            in appendix C that defined the c_m, in the basis of powers
            of E (not delta**2!), as a NumPy array.
        rtol: The relative tolerance to be passes to np.allclose() when
            checking for convergence.
        atol: The absolute tolerance to be passes to np.allclose() when
            checking for convergence.
        abs_tol: The absolute tolerance for the change in norm between
            iterations.

    Returns:
        The coefficients c_m in a Numpy Array.

    Raises:
        ValueError: If mu is negative.
    """
    if mu < 0:
        raise ValueError("mu cannot be negative")
    n_elements = 2 * mu + 1
    last_solution = np.ones(n_elements) * np.inf
    # This is just a safe initial estimate for the amount of padding needed.
    # There is a hard minimum imposed by the number of elements requested, and
    # we add len(C_coeffs) to be on the safe side.
    initial_padding = max(1, n_elements - len(C_coeffs)) + len(C_coeffs)
    for padding in itertools.count(initial_padding):
        first_col, rhs = build_Toeplitz_system(p, C_coeffs, padding)
        solution = la.solve_toeplitz(first_col, rhs)
        center = len(solution) // 2
        solution = solution[center - mu : center + mu + 1]
        if np.allclose(solution, last_solution, rtol, atol):
            return solution
        last_solution = solution


def generating_function(z, s):
    """Compute the function defined in equation (B1) of the paper.

    Args:
        z: The argument on which the function will be Mclaurin-expanded.
        s: The parameter that determines the Mclaurin coefficients.

    Returns:
        The value of the function.
    """
    sz = s * z
    return s * sympy.sinh(sz) / (s * s + 2 * (1 - sympy.cosh(sz)))


def calc_Mclaurin_coefficient(function, variable, order):
    """Calculate a coefficient in truncated McLaurin series of a function.

    Args:
        function: A SymPy function.
        variable: The SymPy variable of the power series.
        order: The order of the coefficient.

    Returns:
        The coefficient as a SymPy expression.
    """
    return sympy.expand(
        function.diff(variable, order).subs(variable, 0)
        / sympy.factorial(order)
    )


def calc_B_p_over_2(p):
    """Build the polynomial B_(p/2) defined in equation B1.

    Args:
        p: The index of the polynomial. It must be a positive even integer.

    Returns:
        The polynomial as a SymPy object. The variable is s=delta**2.

    Raises:
        ValueError: If p does not make sense.
    """
    if p < 2 or p % 2 != 0:
        raise ValueError("p must be a positive even integer")
    symbolic_z = sympy.Symbol("z")
    symbolic_s = sympy.Symbol("s")
    mclaurin = calc_Mclaurin_coefficient(
        generating_function(symbolic_z, symbolic_s), symbolic_z, p - 1
    )
    sympy_polynomial = sympy.poly(
        mclaurin.subs(symbolic_s**2, symbolic_s), symbolic_s
    )
    return sympy_polynomial


def calc_polynomials(p):
    """Obtain the polynomials b and C defined by equation (C1).

    Args:
        p: The index of the polynomial. It must be an even integer greater
            than 2.

    Returns:
        A tuple with the polynomials (b, C) as SymPy objects.

    Raises:
        ValueError: If p violates the constraints.
    """
    if p < 4 or p % 2 != 0:
        raise ValueError("p must be an even integer greater than 2")
    # Build equation (C1) symbolically, but putting everything in the left-hand
    # side.
    polynomial_B = calc_B_p_over_2(p)
    symbolic_s = polynomial_B.gen
    polynomial_B2 = polynomial_B**2
    lhs_coefficient_generator = sympy.numbered_symbols("b")
    b_symbolic_coefficients = [
        next(lhs_coefficient_generator) for i in range(p // 2)
    ][::-1]
    b_polynomial = sympy.Poly(b_symbolic_coefficients, symbolic_s)
    lhs = polynomial_B2 * b_polynomial
    C_coefficient_generator = sympy.numbered_symbols("C")
    C_symbolic_coefficients = [
        next(C_coefficient_generator) for i in range(p - 2)
    ][::-1]
    C_polynomial = sympy.Poly(C_symbolic_coefficients, symbolic_s)
    lhs += symbolic_s ** (p // 2) * C_polynomial - 1
    # Now demand that all coefficients be equal to zero and solve the system
    # of linear equations resulting from this condition.
    lhs_symbolic_coefficients = lhs.all_coeffs()
    # Solve it to get the coefficients of both polynomials.
    solution = sympy.solve(lhs_symbolic_coefficients)
    # Replace the symbolic coefficients by their values and return.
    b_polynomial = b_polynomial.subs(solution)
    C_polynomial = C_polynomial.subs(solution)
    return (sympy.Poly(b_polynomial), sympy.Poly(C_polynomial))


def explode_Add(add_object):
    """Extract all the coefficients from a sum of powers of a single variable.

    This function is intended for a narrow class of sympy.Add objects
    containing only positive and negative values of a single variable, and
    its behavior for any other inputs is undefined.

    Args:
        add_object: The sympy.Add object to be analyzed.

    Returns:
        A diccionary of coefficients of the powers contained in add_object. The
        keys are the exponents.

    Raises:
        ValueError: If the argument is not a sympy.Add object.
    """
    if not isinstance(add_object, sympy.Add):
        raise ValueError("the argument should be a sympy.Add object")
    nruter = dict()
    terms = add_object.args
    for term in terms:
        if isinstance(term, sympy.Mul):
            # All actual powers go here.
            subterms = term.args
            power = subterms[1]
            if isinstance(power, sympy.Symbol):
                # This is the linear term.
                nruter[1] = subterms[0]
            else:
                # This is a power with an exponent different from 1.
                subsubterms = power.args
                nruter[subsubterms[1]] = subterms[0]
        else:
            # Otherwise, we are dealing with the independent term.
            nruter[0] = term
    return nruter


def coeff_dict_to_array(coefficients):
    """Translate the output of explode_Add() into a numerical array.

    Args:
        coefficients: The return code of explode_Add(), a dictionary of
            SymPy objects indexed by their exponent.

    Returns:
        A NumPy array of floating point numbers with the coefficients, that
        can be passed to an equation solver.

    Raises:
        ValueError: If the list of coefficients does not correspond to an
            even function.
    """
    exponents = list(coefficients.keys())
    min_exponent = min(exponents)
    max_exponent = max(exponents)
    if min_exponent != -max_exponent:
        raise ValueError("the coefficients do not come from an even function")
    nruter = np.array(
        [
            float(coefficients[i]) if i in exponents else 0
            for i in range(min_exponent, max_exponent + 1)
        ]
    )
    return nruter


def compute_coeffs_with_truncation(p, mu):
    b_polynomial, C_polynomial = calc_polynomials(p)
    symbolic_s = b_polynomial.gen

    # Insert the actual expression of the centered difference operator,
    # and gather together all the terms corresponding to each power of E.
    # After this, the objects are no longer SymPy polynomials, but
    # "Add" objects that can be decomposed into a sum of "Mul" objects.
    symbolic_E = sympy.Symbol("E")
    delta2 = symbolic_E + 1 / symbolic_E - 2
    b_expr = b_polynomial.subs(symbolic_s, delta2).expr.expand()
    C_expr = C_polynomial.subs(symbolic_s, delta2).expr.expand()

    # Build the non-zero "core" of the right-hand side of the unnumbered equation
    # in appendix C that defined the c_m.
    C_coefficients = explode_Add(C_expr)
    C_coefficients = coeff_dict_to_array(C_coefficients)

    # Solve for c_m
    c_m = compute_c_m(p, C_coefficients, mu)

    # Use SymPy one last time to multipy the expression defined by the c_m
    # by a factor of delta**p and collect the coefficients.
    c_m_contr = sympy.Poly(c_m, symbolic_E).expr / symbolic_E ** (
        len(c_m) // 2
    )
    c_m_contr *= delta2 ** (p // 2)
    c_m_contr = c_m_contr.expand()
    multiplied_c_m = coeff_dict_to_array(explode_Add(c_m_contr))

    # The quasi-interpolating omega_prime consist of two contributions: one coming
    # from the "exact" part (the b polynomial) and the other from the truncated
    # c_m. Therefore, we also need to translate the coefficients of b into a NumPy
    # array.
    b_coefficients = coeff_dict_to_array(explode_Add(b_expr))

    # Finally, add both paths of the solution to get the quasi_omega_prime.
    # To do that, we need to pad the shortest of those two arrays.
    common_length = max(len(b_coefficients), len(multiplied_c_m))
    b_coefficients = np.pad(
        b_coefficients,
        max(0, (common_length - len(b_coefficients)) // 2),
        "constant",
        constant_values=0.0,
    )
    multiplied_c_m = np.pad(
        multiplied_c_m,
        max(0, (common_length - len(multiplied_c_m)) // 2),
        "constant",
        constant_values=0.0,
    )

    omega_prime = b_coefficients + multiplied_c_m

    return omega_prime, c_m


def compute_j_zeroplus(p: int) -> np.ndarray:
    """Compute the sequence J of spline-nesting coefficients given by eq. (22).

    Args:
        p: The degree of the spline in the convention of the article.

    Returns:
        Array of coefficients for expressing B-splines at one grid level in
        terms of the B-splines at the next finer level.
        The result contains only the elements with non-negative index,
        those with negative index follow by symmetry.
    """
    if p < 2 or p % 2 != 0:
        raise ValueError("p must be a positive even integer")
    n = np.arange(0, p // 2 + 1)
    return 2.0 ** (1 - p) * comb(p, p // 2 + n)


if __name__ == "__main__":
    omega_prime, c_m = compute_coeffs_with_truncation(p=P_INDEX, mu=MU)

    print("-" * 72)
    print("p =", P_INDEX, ", mu =", MU)
    print("-" * 72)
    print("c_m:")
    print(c_m[len(c_m) // 2 :])
    print("omega_prime:")
    print(omega_prime[len(omega_prime) // 2 :])
    print("-" * 72)
