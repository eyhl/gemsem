import numpy as np

def l2_model(alpha, B, G, Ar, ce_inv, convergence_limit=5e-5, c=1.345, print_iteration=None):
    # Huber weights, initialised as identity
    wh = np.ones(len(B))

    # array of ones:
    array_ones = np.copy(wh)

    # set arbitrarily higher than convergence limit
    convergence = convergence_limit + 1

    # set up weighting matrix
    W = np.diag(ce_inv * wh)

    # intialise a simple model
    lhs = G.T.dot(W.dot(G))
    rhs = G.T.dot(W.dot(B))

    # solve for the model coefficients to obtain initial model
    model_previous = np.linalg.solve(lhs, rhs)

    i = 0  # counter for convergence print
    for i in range(int(1e6)):
        if convergence < convergence_limit:
            break

        # setup and solve inverse problem:
        rhs = G.T.dot(W.dot(B))
        lhs = G.T.dot(W.dot(G)) + alpha ** 2 * Ar.T.dot(Ar)

        model_current = np.linalg.solve(lhs, rhs)

        # calculate residual, normalise by standard deviation
        normalised_residual = (B - G.dot(model_current)) * np.sqrt(ce_inv)

        # update Huber weights
        wh = np.fmin(array_ones, np.abs(c / normalised_residual))

        # update weights
        W = np.diag(ce_inv * wh)

        # compute convergence parameter
        convergence = np.linalg.norm(model_current - model_previous) / np.linalg.norm(model_current)

        model_previous = np.copy(model_current)

        if print_iteration is not None:
            i += 1
            if i % print_iteration == 1:
                print('i no.', i, 'convergence', convergence)

    residuals = B - G.dot(model_current)

    # vector norm of misfit
    misfit_norm = residuals.T.dot(W.dot(residuals))

    # vector norm of measured model complexity mT.LT.L.m
    model_norm = model_current.T.dot(Ar.T).dot(Ar.dot(model_current))
    return model_current, model_norm, misfit_norm, residuals, W


#%%

def l1_model(alpha, B, G, Ar, ce_inv, convergence_limit=5e-5, c=1.345, kappa=1e-8, print_iteration=None):
    # Huber weights, initialised as identity
    wh = np.ones(len(B))

    # array of ones:
    array_ones = np.copy(wh)

    # Setup weight matrix:
    W = np.diag(ce_inv * wh)

    # for every new alpha, start up with initial guess:
    gamma = np.ones(Ar.shape[0])

    # initialise L1-regularised model in order to compute convergence condition in 1st while-loop:
    Ws = np.diag(1 / np.sqrt(gamma ** 2 + kappa ** 2))

    rhs = G.T.dot(W.dot(B))
    lhs = G.T.dot(W.dot(G)) + alpha ** 2 * Ar.T.dot(Ws).dot(Ar)

    model_previous = np.linalg.solve(lhs, rhs)

    # initial while loop condition: a number higher than convergence limit
    convergence = convergence_limit + 1

    # counter for print option
    i = 0
    for i in range(int(1e6)):
        if convergence < convergence_limit:
            break

        # compute gamma
        gamma = Ar.dot(model_previous)

        # defining the weight vector, the diagonal elements, for L1
        Ws = np.diag(1 / np.sqrt(gamma ** 2 + kappa ** 2))

        rhs = G.T.dot(W.dot(B))
        lhs = G.T.dot(W.dot(G)) + alpha ** 2 * Ar.T.dot(Ws).dot(Ar)

        # solve for the model coefficients
        model_current = np.linalg.solve(lhs, rhs)

        # calculate residual
        normalised_residual = (B - G.dot(model_current)) * np.sqrt(ce_inv)

        # update Huber weights
        wh = np.fmin(array_ones, np.abs(c / normalised_residual))

        # update weighting matrix
        W = np.diag(ce_inv * wh)

        # compute convergence parameter
        convergence = np.linalg.norm(model_current - model_previous) / np.linalg.norm(model_current)
        model_previous = np.copy(model_current)

        # prints every tenth or every single iteration
        if print_iteration is not None:
            i += 1
            if i % print_iteration == 1:
                print('i no.', i, 'convergence', convergence)

    residuals = B - G.dot(model_current)

    # vector norm of misfit
    misfit_norm = residuals.T.dot(W.dot(residuals))

    # vector norm of measured model complexity mT.LT.Ws.L.m
    model_norm = (Ar.dot(model_current)).T.dot(Ws).dot(Ar.dot(model_current))

    return model_current, model_norm, misfit_norm, residuals, W


#%%


def max_ent_model(alpha, omega, initial_model, B, G, Ar, ce_inv, find_omega=False,
            convergence_limit=5e-5, c=1.345, print_iteration=None):

    # Use L2 regularised model found earlier as the initial model
    model_previous = initial_model

    # Huber weights, initialised as identity
    wh = np.ones(len(B))

    # array of ones:
    array_ones = np.copy(wh)

    # Setup weight matrix:
    W = np.diag(ce_inv * wh)

    # initial while loop condition: a number higher than convergence limit
    convergence = convergence_limit + 1

    # for counting tries of finding omega
    count_tries = 0
    for i in range(int(1e6)):
        if convergence < convergence_limit:
            break

        # compute Br at CMB based on previous model
        Br_cmb = Ar.dot(model_previous)

        # define max entropy variables
        psi = np.sqrt(Br_cmb ** 2 + 4 * omega ** 2)
        alpha_1 = (4 * omega) / psi
        beta = np.log((psi + Br_cmb) / (2 * omega))
        alpha_n = Ar.T.dot(np.diag(alpha_1)).dot(Ar)

        # update model
        lhs = 2 * G.T.dot(W).dot(G) + alpha ** 2 * alpha_n
        rhs = 2 * G.T.dot(W).dot(B) + alpha ** 2 * (alpha_n.dot(model_previous) - 4 * omega * alpha ** 2 * (Ar.T.dot(beta)))
        model_current = np.linalg.solve(lhs, rhs)

        # calculate residual
        normalised_residual = (B - G.dot(model_current)) * np.sqrt(ce_inv)

        # update Huber weights
        wh = np.fmin(array_ones, np.abs(c / normalised_residual))

        # update weights
        W = np.diag(ce_inv * wh)

        # compute convergence parameter
        convergence = np.linalg.norm(model_current - model_previous) / np.linalg.norm(model_current)

        # update model
        model_previous = np.copy(model_current)

        # prints every tenth iteration
        if print_iteration is not None:
            i += 1
            if i % print_iteration == 1:
                print('i no.', i, 'convergence', convergence)

        # try a higher value for omega
        if find_omega:
            if i > 1000:
                if print_iteration:
                    print('Could not converge with omega={0:.2f} in {0:.0f} iterations.'.format(omega, i))
                    print('Will try omega={0:.2f}'.format(omega + omega / 2))
                    count_tries += 1

                omega = omega + omega / 2
                break

        if count_tries > 10:
            print('Tried increasing omega 10 times without convergence, omega={0:.2f}. Breaking loop'.format(omega))
            break
        if np.isnan(convergence):
            print('Convergence is nan')
            break

    # computes model norm, residuals and misfit for evaluation purposes
    residuals = B - G.dot(model_current)

    # vector norm of misfit
    misfit_norm = residuals.T.dot(W.dot(residuals))

    # vector norm of measured model complexity -4 * omega * np.sum(psi - 2 * omega - Br_cmb * beta)
    model_norm = - 4 * omega * np.sum(psi - 2 * omega - Br_cmb * beta)

    return model_current, model_norm, misfit_norm, residuals, W


# # list for finding the best model wrt given alpha
# model_list.append(model_current)
# residuals_list.append(residuals)
# misfit_norm_list.append(misfit_norm)
# misfit_list.append(misfit)
# model_norm_list.append(model_norm)

