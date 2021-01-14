import torch
from plnn.proxlp_solver import utils
import math
import copy
from plnn.explp_solver import anderson_optimization


def layer_primal_linear_minimization(lay_idx, f_k, g_k, cl_k, cu_k):
    """
    Given the post-activation bounds and the (functions of) dual variables of the current layer tensors
    (shape 2 * n_neurons_to_opt x c_layer_size), compute the values of the primal variables (x and z) minimizing the
    inner objective.
    :return: optimal x, optimal z (tensors, shape: 2 * n_neurons_to_opt x c_layer_size)
    """
    opt_x_k = (torch.where(f_k >= 0, cu_k.unsqueeze(1), cl_k.unsqueeze(1)))
    if lay_idx > 0:
        opt_z_k = (torch.where(g_k >= 0, torch.ones_like(g_k), torch.zeros_like(g_k)))
    else:
        # g_k is defined from 1 to n - 1.
        opt_z_k = None

    return opt_x_k, opt_z_k


def compute_bounds(weights, dual_vars, clbs, cubs, l_preacts, u_preacts):
    """
    Given the network layers, post- and pre-activation bounds  as lists of tensors, and dual variables
    (and functions thereof) as DualVars. compute the value of the (batch of) network bounds.
    :return: a tensor of bounds, of size 2 x n_neurons of the layer to optimize. The first half is the negative of the
    upper bound of each neuron, the second the lower bound.
    """

    c_dual_vars = dual_vars

    bounds = 0
    for lin_k, alpha_k_1 in zip(weights, c_dual_vars.alpha[1:]):
        b_k = lin_k.get_bias()
        bounds += utils.bdot(alpha_k_1, b_k)

    for f_k, cl_k, cu_k in zip(c_dual_vars.fs, clbs, cubs):
        bounds -= utils.bdot(torch.clamp(f_k, 0, None), cu_k.unsqueeze(1))
        bounds -= utils.bdot(torch.clamp(f_k, None, 0), cl_k.unsqueeze(1))

    for g_k in c_dual_vars.gs:
        bounds -= torch.clamp(g_k, 0, None).view(*g_k.shape[:2], -1).sum(dim=-1)  # z to 1

    for beta_k_1, l_preact, lin_k in zip(c_dual_vars.beta_1[1:], l_preacts[1:], weights):
        bounds += utils.bdot(beta_k_1, (l_preact.unsqueeze(1) - lin_k.get_bias()))

    return bounds


def compute_dual_subgradient(weights, dual_vars, lbs, ubs, l_preacts, u_preacts):
    """
    Given the network layers, post- and pre-activation bounds as lists of
    tensors, and dual variables (and functions thereof) as DualVars, compute the subgradient of the dual objective.
    :return: DualVars instance representing the subgradient for the dual variables (does not contain fs and gs)
    """

    # The step needs to be taken for all layers at once, as coordinate ascent seems to be problematic,
    # see https://en.wikipedia.org/wiki/Coordinate_descent

    nb_relu_layers = len(dual_vars.beta_0)

    alpha_subg = [torch.zeros_like(dual_vars.alpha[0])]
    beta_0_subg = [torch.zeros_like(dual_vars.beta_0[0])]
    beta_1_subg = [torch.zeros_like(dual_vars.beta_1[0])]
    xkm1, _ = layer_primal_linear_minimization(0, dual_vars.fs[0], None, lbs[0], ubs[0])
    for lay_idx in range(1, nb_relu_layers):
        # For each layer, we will do one step of subgradient descent on all dual variables at once.
        lin_k = weights[lay_idx - 1]
        # solve the inner problems.
        xk, zk = layer_primal_linear_minimization(lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1],
                                                  lbs[lay_idx], ubs[lay_idx])

        # compute and store the subgradients.
        xk_hat = lin_k.forward(xkm1)
        alpha_subg.append(xk_hat - xk)
        beta_0_subg.append(xk - zk * u_preacts[lay_idx].unsqueeze(1))
        beta_1_subg.append(xk + (1 - zk) * l_preacts[lay_idx].unsqueeze(1) - xk_hat)

        xkm1 = xk

    return DualVars(alpha_subg, beta_0_subg, beta_1_subg, None, None, None, None)


class DualVars:
    """
    Class representing the dual variables alpha, beta_0, and beta_1, and their functions f and g.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for beta_0, for indices 0 to n for
    the others.
    """
    def __init__(self, alpha, beta_0, beta_1, fs, gs, alpha_back, beta_1_back):
        """
        Given the dual vars as lists of tensors (of correct length) along with their computed functions, initialize the
        class with these.
        alpha_back and beta_1_back are lists of the backward passes of alpha and beta_1. Useful to avoid
        re-computing them.
        """
        self.alpha = alpha
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.fs = fs
        self.gs = gs
        self.alpha_back = alpha_back
        self.beta_1_back = beta_1_back

    @staticmethod
    def naive_initialization(weights, additional_coeffs, device, input_size):
        """
        Given parameters from the optimize function, initialize the dual vairables and their functions as all 0s except
        some special corner cases. This is equivalent to initialising with naive interval propagation bounds.
        """
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]

        alpha = []  # Indexed from 0 to n, the last is constrained to the cost function, first is zero
        beta_0 = []  # Indexed from 0 to n-1, the first is always zero
        beta_1 = []  # Indexed from 0 to n, the first and last are always zero
        alpha_back = []  # Indexed from 1 to n,
        beta_1_back = []  # Indexed from 1 to n, last always 0

        # Build also the shortcut terms f and g
        fs = []  # Indexed from 0 to n-1
        gs = []  # Indexed from 1 to n-1

        # Fill in the variable holders with variables, all initiated to zero
        zero_tensor = lambda size: torch.zeros((*batch_size, *size), device=device)
        # Insert the dual variables for the box bound
        fs.append(zero_tensor(input_size))
        fixed_0_inpsize = zero_tensor(input_size)
        beta_0.append(fixed_0_inpsize)
        beta_1.append(fixed_0_inpsize)
        alpha.append(fixed_0_inpsize)
        for lay_idx, layer in enumerate(weights[:-1]):
            nb_outputs = layer.get_output_shape(beta_0[-1].shape)[2:]

            # Initialize the dual variables
            alpha.append(zero_tensor(nb_outputs))
            beta_0.append(zero_tensor(nb_outputs))
            beta_1.append(zero_tensor(nb_outputs))

            # Initialize the shortcut terms
            fs.append(zero_tensor(nb_outputs))
            gs.append(zero_tensor(nb_outputs))

        # Add the fixed values that can't be changed that comes from above
        alpha.append(additional_coeffs[len(weights)])
        beta_1.append(torch.zeros_like(alpha[-1]))

        for lay_idx in range(1, len(alpha)):
            alpha_back.append(weights[lay_idx-1].backward(alpha[lay_idx]))
            beta_1_back.append(weights[lay_idx-1].backward(beta_1[lay_idx]))

        # Adjust the fact that the last term for the f shorcut is not zero,
        # because it depends on alpha.
        fs[-1] = -weights[-1].backward(additional_coeffs[len(weights)])

        return DualVars(alpha, beta_0, beta_1, fs, gs, alpha_back, beta_1_back)

    def update_f_g(self, l_preacts, u_preacts, lay_idx="all"):
        """
        Given the network pre-activation bounds as lists of tensors, update f_k and g_k in place.
        lay_idx are the layers (int or list) for which to perform the update. "all" means update all
        """
        if lay_idx == "all":
            lay_to_iter = range(len(self.beta_0))
        else:
            lay_to_iter = [lay_idx] if type(lay_idx) is int else list(lay_idx)

        for lay_idx in lay_to_iter:
            self.fs[lay_idx] = (
                    self.alpha[lay_idx] - self.alpha_back[lay_idx] -
                    (self.beta_0[lay_idx] + self.beta_1[lay_idx]) + self.beta_1_back[lay_idx])
            if lay_idx > 0:
                self.gs[lay_idx - 1] = (self.beta_0[lay_idx] * u_preacts[lay_idx].unsqueeze(1) +
                                        self.beta_1[lay_idx] * l_preacts[lay_idx].unsqueeze(1))

    def projected_linear_combination(self, coeff, o_vars, weights):
        """
        Given a batch of coefficients (a tensor) and another set of dual variables (instance of this calss), perform a
        linear combination according to the coefficient.
        Then project on the feasible domain (non-negativity constraints).
        This is done in place the set of variables of this class.
        """
        for lay_idx in range(1, len(self.beta_0)):
            self.alpha[lay_idx] = torch.clamp(self.alpha[lay_idx] + coeff * o_vars.alpha[lay_idx], 0, None)
            self.beta_0[lay_idx] = torch.clamp(self.beta_0[lay_idx] + coeff * o_vars.beta_0[lay_idx], 0, None)
            self.beta_1[lay_idx] = torch.clamp(self.beta_1[lay_idx] + coeff * o_vars.beta_1[lay_idx], 0, None)
            self.alpha_back[lay_idx - 1] = weights[lay_idx - 1].backward(self.alpha[lay_idx])
            self.beta_1_back[lay_idx - 1] = weights[lay_idx - 1].backward(self.beta_1[lay_idx])
        return

    def get_nonnegative_copy(self, weights, l_preacts, u_preacts):
        """
        Given the network layers and pre-activation bounds as lists of tensors, clamp all dual variables to be
        non-negative. A heuristic to compute some bounds.
        Returns a copy of this instance where all the entries are non-negative and the f and g functions are up-to-date.
        """
        nonneg = self.copy()
        for lay_idx in range(1, len(nonneg.beta_0)):
            nonneg.alpha[lay_idx].clamp_(0, None)
            nonneg.beta_0[lay_idx].clamp_(0, None)
            nonneg.beta_1[lay_idx].clamp_(0, None)
            nonneg.alpha_back[lay_idx-1] = weights[lay_idx-1].backward(nonneg.alpha[lay_idx])
            nonneg.beta_1_back[lay_idx-1] = weights[lay_idx-1].backward(nonneg.beta_1[lay_idx])
        nonneg.update_f_g(l_preacts, u_preacts)
        return nonneg

    def update_from_anchor_points(self, anchor_point, xt, zt, xhatt, y, eta, weights, l_preacts, u_preacts, lay_idx="all"):
        """
        Given the anchor point (DualVars instance), post-activation bounds, primal vars as lists of
        tensors (y is a YVars instance), compute and return the updated the dual variables (anchor points) with their
        closed-form from KKT conditions. The update is performed in place.
        lay_idx are the layers (int or list) for which to perform the update. "all" means update all
        """
        if lay_idx == "all":
            lay_to_iter = range(1, len(self.beta_0))
        else:
            lay_to_iter = [lay_idx] if type(lay_idx) is int else list(lay_idx)

        for lay_idx in lay_to_iter:
            # For each layer, do the dual anchor points' update.
            # compute the quadratic terms (-the subgradients, in l2 norm).
            self.alpha[lay_idx] = anchor_point.alpha[lay_idx] - (1 / (2 * eta)) * (xt[lay_idx] - xhatt[lay_idx-1]
                                                                                       - y.ya[lay_idx])
            self.beta_0[lay_idx] = anchor_point.beta_0[lay_idx] - (1 / (2 * eta)) * (
                    zt[lay_idx-1] * u_preacts[lay_idx].unsqueeze(1) - xt[lay_idx] - y.yb0[lay_idx])
            self.beta_1[lay_idx] = anchor_point.beta_1[lay_idx] - (1 / (2 * eta)) * (
                    xhatt[lay_idx - 1] - xt[lay_idx] - (1 - zt[lay_idx-1]) * l_preacts[lay_idx].unsqueeze(1) -
                    y.yb1[lay_idx])
            self.alpha_back[lay_idx-1] = weights[lay_idx-1].backward(self.alpha[lay_idx])
            self.beta_1_back[lay_idx-1] = weights[lay_idx-1].backward(self.beta_1[lay_idx])

    def copy(self):
        """
        deep-copy the current instance
        :return: the copied class instance
        """
        return DualVars(
            copy.deepcopy(self.alpha),
            copy.deepcopy(self.beta_0),
            copy.deepcopy(self.beta_1),
            copy.deepcopy(self.fs),
            copy.deepcopy(self.gs),
            copy.deepcopy(self.alpha_back),
            copy.deepcopy(self.beta_1_back)
        )

    def as_explp_initialization(self, weights, clbs, cubs, l_preacts, u_preacts):
        """
        Given the network layers and pre-activation bounds as lists of tensors,
        compute and return the corresponding initialization of the explp (Anderson) variables from the instance of this
        class.
        """
        dual_vars = self.copy()
        sum_beta = [None] * len(dual_vars.beta_0)
        sum_Wp1Ibetap1 = [None] * (len(dual_vars.beta_0))
        sum_W1mIubeta = [None] * len(dual_vars.beta_0)
        sum_WIlbeta = [None] * len(dual_vars.beta_0)
        xs = [None] * len(dual_vars.beta_0)
        zs = [None] * (len(dual_vars.beta_0) - 1)
        for lay_idx in range(len(dual_vars.beta_0)+1):
            if lay_idx == 0:
                sum_beta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                sum_W1mIubeta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                sum_WIlbeta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                xs[lay_idx], _ = layer_primal_linear_minimization(lay_idx, dual_vars.fs[lay_idx], None, clbs[lay_idx],
                                                          cubs[lay_idx])
            elif lay_idx > 0:
                if lay_idx < len(dual_vars.beta_0):
                    sum_beta[lay_idx] = dual_vars.beta_0[lay_idx] + dual_vars.beta_1[lay_idx]
                    sum_W1mIubeta[lay_idx] = dual_vars.beta_0[lay_idx] * (u_preacts[lay_idx].unsqueeze(1)
                                                                          - weights[lay_idx-1].get_bias())
                    sum_WIlbeta[lay_idx] = dual_vars.beta_1[lay_idx] * (l_preacts[lay_idx].unsqueeze(1)
                                                                        - weights[lay_idx-1].get_bias())
                    xs[lay_idx], zs[lay_idx-1] = layer_primal_linear_minimization(
                        lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1], clbs[lay_idx], cubs[lay_idx])
                sum_Wp1Ibetap1[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.beta_1[lay_idx])

        return dual_vars.alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, dual_vars.fs, dual_vars.gs, xs, zs

    def as_cut_initialization(self, weights, clbs, cubs, l_preacts, u_preacts):
        """
        Given the network layers and pre-activation bounds as lists of tensors,
        compute and return the corresponding initialization of the explp (Anderson) variables from the instance of this
        class.
        """
        dual_vars = self.copy()
        sum_beta = [None] * len(dual_vars.beta_0)
        sum_Wp1Ibetap1 = [None] * (len(dual_vars.beta_0))
        sum_W1mIubeta = [None] * len(dual_vars.beta_0)
        sum_WIlbeta = [None] * len(dual_vars.beta_0)
        xs = [None] * len(dual_vars.beta_0)
        zs = [None] * (len(dual_vars.beta_0) - 1)
        beta_list = []
        I_list = []
        for lay_idx in range(len(dual_vars.beta_0)+1):
            xkm1 = xs[lay_idx-1]
            if lay_idx == 0:
                sum_beta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                sum_W1mIubeta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                sum_WIlbeta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                xs[lay_idx], _ = layer_primal_linear_minimization(lay_idx, dual_vars.fs[lay_idx], None, clbs[lay_idx],
                                                                  cubs[lay_idx])
            elif lay_idx > 0:
                I_list.append([])
                beta_list.append([])
                if lay_idx < len(dual_vars.beta_0):
                    sum_beta[lay_idx] = dual_vars.beta_0[lay_idx] + dual_vars.beta_1[lay_idx]
                    beta_list[lay_idx-1].append(dual_vars.beta_0[lay_idx])
                    beta_list[lay_idx-1].append(dual_vars.beta_1[lay_idx])
                    lin_k = weights[lay_idx - 1]
                    xs[lay_idx], zs[lay_idx - 1] = layer_primal_linear_minimization(
                        lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1], clbs[lay_idx], cubs[lay_idx])
                    if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]:
                        unfolded_xkm1 = lin_k.unfold_input(xkm1)  # input space unfolding
                        I_shape = lin_k.unfold_output(zs[lay_idx-1]).shape[:3] + unfolded_xkm1.shape[-2:]
                    else:
                        # Fully connected layer.
                        if lin_k.flatten_from_shape is not None:
                            xkm1 = xkm1.view(*xkm1.shape[:2], -1)
                        I_shape = zs[lay_idx - 1].shape[:3] + xkm1.shape[-1:]
                    I_list[lay_idx - 1].append(torch.zeros(I_shape, dtype=torch.bool, device=xkm1.device))
                    I_list[lay_idx - 1].append(torch.ones(I_shape, dtype=torch.bool, device=xkm1.device))
                    sum_W1mIubeta[lay_idx] = dual_vars.beta_0[lay_idx] * (u_preacts[lay_idx].unsqueeze(1) -
                                                                          weights[lay_idx-1].get_bias())
                    sum_WIlbeta[lay_idx] = dual_vars.beta_1[lay_idx] * (l_preacts[lay_idx].unsqueeze(1) -
                                                                        weights[lay_idx-1].get_bias())
                sum_Wp1Ibetap1[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.beta_1[lay_idx])

        return dual_vars.alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, dual_vars.fs, dual_vars.gs, xs, zs, beta_list, I_list



class DualADAMStats:
    """
    class storing (and containing operations for) the ADAM statistics for the dual variables.
    they are stored as lists of tensors, for ReLU indices from 1 to n-1.
    """
    def __init__(self, beta_0, beta1=0.9, beta2=0.999):
        """
        Given beta_0 to copy the dimensionality from, initialize all ADAM stats to 0 tensors.
        """
        # first moments
        self.m1_alpha = []
        self.m1_beta_0 = []
        self.m1_beta_1 = []
        # second moments
        self.m2_alpha = []
        self.m2_beta_0 = []
        self.m2_beta_1 = []
        for lay_idx in range(1, len(beta_0)):
            self.m1_alpha.append(torch.zeros_like(beta_0[lay_idx]))
            self.m1_beta_0.append(torch.zeros_like(beta_0[lay_idx]))
            self.m1_beta_1.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_alpha.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_beta_0.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_beta_1.append(torch.zeros_like(beta_0[lay_idx]))

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def update_moments_take_projected_step(self, weights, step_size, outer_it, dual_vars, dual_vars_subg):
        """
        Update the ADAM moments given the subgradients, and normal gd step size, then take the projected step from
        dual_vars.
        Update performed in place on dual_vars.
        """
        for lay_idx in range(1, len(dual_vars.beta_0)):
            # Update the ADAM moments.
            self.m1_alpha[lay_idx-1].mul_(self.coeff1).add_(dual_vars_subg.alpha[lay_idx], alpha=1-self.coeff1)
            self.m1_beta_0[lay_idx-1].mul_(self.coeff1).add_(dual_vars_subg.beta_0[lay_idx], alpha=1-self.coeff1)
            self.m1_beta_1[lay_idx-1].mul_(self.coeff1).add_(dual_vars_subg.beta_1[lay_idx], alpha=1-self.coeff1)
            self.m2_alpha[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.alpha[lay_idx], dual_vars_subg.alpha[lay_idx], value=1 - self.coeff2)
            self.m2_beta_0[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.beta_0[lay_idx], dual_vars_subg.beta_0[lay_idx], value=1 - self.coeff2)
            self.m2_beta_1[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.beta_1[lay_idx], dual_vars_subg.beta_1[lay_idx], value=1 - self.coeff2)

            bias_correc1 = 1 - self.coeff1 ** (outer_it + 1)
            bias_correc2 = 1 - self.coeff2 ** (outer_it + 1)
            corrected_step_size = step_size * math.sqrt(bias_correc2) / bias_correc1

            # Take the projected (non-negativity constraints) step.
            alpha_step_size = self.m1_alpha[lay_idx-1] / (self.m2_alpha[lay_idx-1].sqrt() + self.epsilon)
            dual_vars.alpha[lay_idx] = torch.clamp(dual_vars.alpha[lay_idx] + corrected_step_size * alpha_step_size, 0, None)

            beta_0_step_size = self.m1_beta_0[lay_idx-1] / (self.m2_beta_0[lay_idx-1].sqrt() + self.epsilon)
            dual_vars.beta_0[lay_idx] = torch.clamp(dual_vars.beta_0[lay_idx] + corrected_step_size * beta_0_step_size, 0, None)

            beta_1_step_size = self.m1_beta_1[lay_idx-1] / (self.m2_beta_1[lay_idx-1].sqrt() + self.epsilon)
            dual_vars.beta_1[lay_idx] = torch.clamp(dual_vars.beta_1[lay_idx] + corrected_step_size * beta_1_step_size, 0, None)

            # update pre-computed backward passes.
            dual_vars.alpha_back[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.alpha[lay_idx])
            dual_vars.beta_1_back[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.beta_1[lay_idx])


class BigMPInit(anderson_optimization.AndersonPInit):
    """
    Parent Init class for Anderson-relaxation-based solvers.
    """

    def as_stack(self, stack_size):
        # Repeat the content of this parent init to form a stack of size "stack_size"
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append(
                [pinits[0].unsqueeze(0).repeat(((stack_size,) + (1,) * (pinits.dim() - 1))) for pinits in varset])
        return BigMPInit(DualVars(*stacked_dual_list))

    def get_stack_entry(self, batch_idx):
        # Return the stack entry at batch_idx as a new ParentInit instance.
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append([csol[batch_idx].unsqueeze(0) for csol in varset])
        return BigMPInit(DualVars(*stacked_dual_list))

    def get_lb_init_only(self):
        # Get instance of this class with only entries relative to LBs.
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append([c_init[:, -1].unsqueeze(1) for c_init in varset])
        return BigMPInit(DualVars(*stacked_dual_list))