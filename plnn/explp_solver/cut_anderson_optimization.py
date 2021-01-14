import torch
from plnn.proxlp_solver import utils
import math
import copy
from plnn.explp_solver import anderson_optimization, bigm_optimization


def compute_dual_subgradient_adam(weights, clbs, cubs, nubs, dual_vars, primal_vars, steps, precision=torch.float,
                                  opt_args=None):
    """
    Given the network layers, post- and pre-activation bounds as lists of
    tensors, and dual variables (and functions thereof) as DualVars, compute the subgradient of the dual objective.
    :return: DualVars instance representing the subgradient for the dual variables (does not contain fs and gs)
    """

    # The step needs to be taken for all layers at once, as coordinate ascent seems to be problematic,
    # see https://en.wikipedia.org/wiki/Coordinate_descent
    # print('computing dual subgradient')
    nb_relu_layers = len(dual_vars.fs)

    alpha_subg = []
    atom_grads = []
    atom_Is = []
    WIs = []
    WIl_lmos = []
    W1mIu_lmos = []
    for lay_idx in range(nb_relu_layers):

        if lay_idx == 0:
            x0, _ = CutPrimalVars.primalsk_min(lay_idx, dual_vars, clbs, cubs, precision)
            primal_vars.update_primals_from_primalsk(lay_idx, x0, None)
            alpha_subg.append(torch.zeros_like(dual_vars.alpha[lay_idx]))
            atom_grads.append(torch.zeros_like(dual_vars.sum_beta[lay_idx]))
            atom_Is.append([])
            W1mIu_lmos.append(torch.zeros_like(dual_vars.sum_W1mIubeta[lay_idx]))
            WIl_lmos.append(torch.zeros_like(dual_vars.sum_WIlbeta[lay_idx]))

        # For each layer, we will do one step of subgradient descent on all dual variables at once.
        if lay_idx != 0:

            new_xk, new_zk = CutPrimalVars.primalsk_min(lay_idx, dual_vars, clbs, cubs, precision)

            primal_vars.update_primals_from_primalsk(lay_idx, new_xk, new_zk)
            # compute and store the subgradients.
            new_alpha_k, grad_alphak, new_alphak_M = dual_vars.alphak_grad(lay_idx, weights, primal_vars, precision)
            alpha_subg.append(grad_alphak)

            atom_grad, WI, WIl_lmo, W1mIu_lmo, atom_I = dual_vars.betak_grad_lmo(
                lay_idx, weights, clbs, cubs, nubs, primal_vars, steps, opt_args)
            atom_grads.append(atom_grad)
            atom_Is.append(atom_I)
            WIs.append(WI)
            WIl_lmos.append(WIl_lmo)
            W1mIu_lmos.append(W1mIu_lmo)

    if not (len(dual_vars.I_list[lay_idx-1]) <= opt_args['max_cuts'] and steps%opt_args['cut_frequency'] < opt_args['cut_add']):
        atom_grads = []
        atom_Is = []
        WIs = []
        WIl_lmos = []
        W1mIu_lmos = []

    return CutDualVars(alpha_subg, atom_grads, WIs, W1mIu_lmos, WIl_lmos, None, None, alpha_M=dual_vars.alpha_M,
                       beta_M=dual_vars.beta_M, beta_list=None, I_list=atom_Is)


class CutDualVars(anderson_optimization.DualVars):
    """
    Class representing the dual variables alpha, beta_0, and beta_1, and their functions f and g.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for beta_0, for indices 0 to n for
    the others.
    """
    def __init__(self, alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs, alpha_M=None, beta_M=None, beta_list=None, I_list=[]):
        """
        Given the dual vars as lists of tensors (of correct length) along with their computed functions, initialize the
        class with these.
        alpha_back and beta_1_back are lists of the backward passes of alpha and beta_1. Useful to avoid
        re-computing them.
        """
        super().__init__(alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs)
        self.beta_list = beta_list
        self.beta_hat_list = copy.deepcopy(beta_list)
        self.alpha_hat = copy.deepcopy(alpha)
        self.I_list = I_list
        self.alpha_M = [1e-2] * len(alpha) if alpha_M is None else alpha_M
        self.beta_M = [1e-2] * len(sum_beta) if beta_M is None else beta_M

    @staticmethod
    def from_super_class(super_instance, alpha_M=None, beta_M=None, beta_list=None, I_list=None):
        """
        Return an instance of this class from an instance of the super class.
        """
        return CutDualVars(super_instance.alpha, super_instance.sum_beta, super_instance.sum_Wp1Ibetap1,
                           super_instance.sum_W1mIubeta, super_instance.sum_WIlbeta, super_instance.fs,
                           super_instance.gs, alpha_M=alpha_M, beta_M=beta_M, beta_list=beta_list, I_list=I_list)

    @staticmethod
    def naive_initialization(weights, additional_coeffs, device, input_size, alpha_M=None, beta_M=None):
        """
        Given parameters from the optimize function, initialize the dual vairables and their functions as all 0s except
        some special corner cases. This is equivalent to initialising with naive interval propagation bounds.
        """
        base_duals = anderson_optimization.DualVars.naive_initialization(weights, additional_coeffs, device, input_size)

        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]

        alpha = []  # Indexed from 0 to n, the last is constrained to the cost function, first is zero
        beta_a = []  # Indexed from 0 to n-1, the first is always zero
        #############################
        beta_list = []
        I_list=[]
        #############################
        # Build also the shortcut terms f and g

        # Fill in the variable holders with variables, all initiated to zero
        zero_tensor = lambda size: torch.zeros((*batch_size, *size), device=device)
        # Insert the dual variables for the box bound
        fixed_0_inpsize = zero_tensor(input_size)
        beta_a.append(fixed_0_inpsize)
        #############################
        for lay_idx, layer in enumerate(weights[:-1]):
            nb_outputs = layer.get_output_shape(beta_a[-1].shape)[2:]

            # Initialize the dual variables
            alpha.append(zero_tensor(nb_outputs))
            beta_a.append(zero_tensor(nb_outputs))
            I_list.append([])
            beta_list.append([])

        # Add the fixed values that can't be changed that comes from above
        alpha.append(additional_coeffs[len(weights)])
        beta_a.append(torch.zeros_like(alpha[-1]))

        return CutDualVars.from_super_class(base_duals, alpha_M=alpha_M, beta_M=beta_M, beta_list=beta_list, I_list=I_list)

    @staticmethod
    def bigm_initialization(bigm_duals, weights, additional_coeffs, device, input_size, clbs, cubs, lower_bounds,
                            upper_bounds, opt_args, alpha_M=None, beta_M=None):
        """
        Given bigm dual variables, network weights, post/pre-activation lower and upper bounds,
        initialize the Anderson dual variables and their functions to the corresponding values of the bigm duals.
        Additionally, it returns the primal variables corresponding to the inner bigm minimization with those dual
        variables.
        """
        alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs, xt, zt, beta_list, I_list = \
            bigm_duals.as_cut_initialization(weights, clbs, cubs, lower_bounds, upper_bounds)

        base_duals, primals = CutDualVars(alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs, beta_list=beta_list, I_list=I_list), \
                              CutPrimalVars(xt, zt, eta=opt_args["eta"], volume=opt_args["volume"])

        return base_duals, primals

    def copy(self):
        """
        deep-copy the current instance
        :return: the copied class instance
        """
        return CutDualVars(
            copy.deepcopy(self.alpha), copy.deepcopy(self.sum_beta), copy.deepcopy(self.sum_Wp1Ibetap1),
            copy.deepcopy(self.sum_W1mIubeta), copy.deepcopy(self.sum_WIlbeta), copy.deepcopy(self.fs),
            copy.deepcopy(self.gs), alpha_M=copy.deepcopy(self.alpha_M), beta_M=copy.deepcopy(self.beta_M),
            beta_list=copy.deepcopy(self.beta_list), I_list=copy.deepcopy(self.I_list))

    def update_duals_from_alphak(self, lay_idx, weights, new_alpha_k, new_alphak_M):
        """
        Given new values for alphas at layer lay_idx, update the dual variables and their functions.
        """
        super().update_duals_from_alphak(lay_idx, weights, new_alpha_k)
        self.alpha_M[lay_idx] = new_alphak_M

    def update_duals_from_betak(self, lay_idx, weights, new_sum_betak, new_sum_WkIbetak, new_sum_Wk1mIubetak, new_sum_WkIlbetak, new_betak_M):
        """
        Given new values for alphas at layer lay_idx, update the dual variables and their functions.
        """
        super().update_duals_from_betak(lay_idx, weights, new_sum_betak, new_sum_WkIbetak, new_sum_Wk1mIubetak, new_sum_WkIlbetak)
        self.beta_M[lay_idx] = new_betak_M

    def alphak_grad(self, lay_idx, weights, primal_vars, precision):
        # Compute the gradient over alphak
        new_alphak_M = self.alpha_M[lay_idx]

        #########
        grad_alphak = weights[lay_idx - 1].forward(primal_vars.xt[lay_idx-1]) - primal_vars.xt[lay_idx]

        # Let's compute the best atom in the dictionary according to the LMO.
        # If grad_alphak > 0, lmo_alphak = M
        delta_alphak = (grad_alphak).type(precision) * self.alpha_M[lay_idx]
        new_alpha_k = self.alpha[lay_idx] + delta_alphak
        ########### CLAMPING ###############################
        new_alpha_k = torch.clamp(new_alpha_k, 0, None) - self.sum_beta[lay_idx]
        ########### CLAMPING ###############################

        return new_alpha_k, grad_alphak, new_alphak_M

    def betak_grad_lmo(self, lay_idx, weights, clbs, cubs, nubs, primal_vars, outer_it, opt_args):
        Ik_list = self.I_list[lay_idx-1]
        exp_k_grad, WI_lmo, WIl_lmo, W1mIu_lmo, Istar_k = [], [], [], [], []
        if (len(Ik_list) <= opt_args['max_cuts'] and outer_it % opt_args['cut_frequency'] < opt_args['cut_add']):
            print('adding cut number', len(Ik_list)+1)
            # Random mask adds a random mask rather than the selection criterion given by the Anderson oracle.
            masked_op, Istar_k, exp_k_grad, WIl, W1mIu, _ = self.anderson_oracle(
                lay_idx, weights, clbs, cubs, nubs, primal_vars, random_mask=opt_args['random_cuts'])

            WI_lmo = masked_op.WI
            WIl_lmo = WIl
            W1mIu_lmo = W1mIu

        return exp_k_grad, WI_lmo, WIl_lmo, W1mIu_lmo, Istar_k

    def update_from_step(self, weights, dual_vars_subg, lay_idx="all"):
        """
        Given the network pre-activation bounds as lists of tensors, all dual variables (and their functions f and g)
        lay_idx are the layers (int or list) for which to perform the update. "all" means update all
        """
        if lay_idx == "all":
            lay_to_iter = range(len(self.fs))
        else:
            lay_to_iter = [lay_idx] if type(lay_idx) is int else list(lay_idx)
        for lay_idx in lay_to_iter:
            if lay_idx > 0:
                self.update_duals_from_alphak(lay_idx, weights, dual_vars_subg.alpha[lay_idx], dual_vars_subg.alpha_M[lay_idx])
                self.update_duals_from_betak(lay_idx, weights, dual_vars_subg.sum_beta[lay_idx], dual_vars_subg.sum_Wp1Ibetap1[lay_idx-1],
                                             dual_vars_subg.sum_W1mIubeta[lay_idx], dual_vars_subg.sum_WIlbeta[lay_idx], dual_vars_subg.beta_M[lay_idx])


class DualADAMStats:
    """
    class storing (and containing operations for) the ADAM statistics for the dual variables.
    they are stored as lists of tensors, for ReLU indices from 1 to n-1.
    """
    def __init__(self, sum_beta, beta1=0.9, beta2=0.999):
        """
        Given beta_0 to copy the dimensionality from, initialize all ADAM stats to 0 tensors.
        """
        # first moments
        self.m1_alpha = []
        self.m1_sum_beta = []
        # second moments
        self.m2_alpha = []
        self.m2_sum_beta = []
        for lay_idx in range(1, len(sum_beta)):
            self.m1_alpha.append(torch.zeros_like(sum_beta[lay_idx]))
            self.m1_sum_beta.append([])
            self.m2_sum_beta.append([])
            self.m2_alpha.append(torch.zeros_like(sum_beta[lay_idx]))

            self.m1_sum_beta[lay_idx-1].append(torch.zeros_like(sum_beta[lay_idx]))
            self.m1_sum_beta[lay_idx-1].append(torch.zeros_like(sum_beta[lay_idx]))
            self.m2_sum_beta[lay_idx-1].append(torch.zeros_like(sum_beta[lay_idx]))
            self.m2_sum_beta[lay_idx-1].append(torch.zeros_like(sum_beta[lay_idx]))

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def bigm_adam_initialization(self, sum_beta, bigm_adam_stats, beta1=0.9, beta2=0.999):
        # first moments
        self.m1_alpha = []
        self.m1_sum_beta = []
        # second moments
        self.m2_alpha = []
        self.m2_sum_beta = []
        for lay_idx in range(1, len(sum_beta)):
            # self.m1_alpha.append(torch.zeros_like(sum_beta[lay_idx]))
            # self.m2_alpha.append(torch.zeros_like(sum_beta[lay_idx]))
            self.m1_alpha.append(bigm_adam_stats.m1_alpha[lay_idx-1])
            self.m2_alpha.append(bigm_adam_stats.m2_alpha[lay_idx-1])

            self.m1_sum_beta.append([])
            self.m2_sum_beta.append([])

            self.m1_sum_beta[lay_idx-1].append(bigm_adam_stats.m1_beta_0[lay_idx-1])
            self.m1_sum_beta[lay_idx-1].append(bigm_adam_stats.m1_beta_1[lay_idx-1])
            self.m2_sum_beta[lay_idx-1].append(bigm_adam_stats.m2_beta_0[lay_idx-1])
            self.m2_sum_beta[lay_idx-1].append(bigm_adam_stats.m2_beta_1[lay_idx-1])

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def update_moments_take_projected_step(self, weights, step_size, outer_it, dual_vars, dual_vars_subg, primal_vars,
                                           clbs, cubs, nubs, l_preacts, u_preacts, cut_frequency, max_cuts, precision,
                                           opt_args):
        """
        Update the ADAM moments given the subgradients, and normal gd step size, then take the projected step from
        dual_vars.
        Update performed in place on dual_vars.
        """
        new_alpha = []
        new_sum_beta = []
        new_sum_WkIbeta = []
        new_sum_Wk1mIubeta = []
        new_sum_WkIlbeta = []

        new_alpha.append(torch.zeros_like(dual_vars.alpha[0]))
        new_sum_beta.append(torch.zeros_like(dual_vars.sum_beta[0]))
        new_sum_Wk1mIubeta.append(torch.zeros_like(dual_vars.sum_W1mIubeta[0]))
        new_sum_WkIlbeta.append(torch.zeros_like(dual_vars.sum_WIlbeta[0]))

        tau=opt_args['tau']
        for lay_idx in range(1, len(dual_vars.sum_beta)):
            # Update the ADAM moments.
            self.m1_alpha[lay_idx-1].mul_(self.coeff1).add_(1-self.coeff1, dual_vars_subg.alpha[lay_idx])
            self.m2_alpha[lay_idx-1].mul_(self.coeff2).addcmul_(1 - self.coeff2, dual_vars_subg.alpha[lay_idx], dual_vars_subg.alpha[lay_idx])

            bias_correc1 = 1 - self.coeff1 ** (outer_it + 1)
            bias_correc2 = 1 - self.coeff2 ** (outer_it + 1)
            corrected_step_size = step_size * math.sqrt(bias_correc2) / bias_correc1

            # Take the projected (non-negativity constraints) step.
            alpha_step_size = self.m1_alpha[lay_idx-1] / (self.m2_alpha[lay_idx-1].sqrt() + self.epsilon)
            new_alpha_k = (1-opt_args['tau'])*torch.clamp(dual_vars.alpha[lay_idx] + corrected_step_size * alpha_step_size, 0, None)+opt_args['tau']*dual_vars.alpha[lay_idx]
            # new_alpha_k = (1-opt_args['tau'])*torch.clamp(dual_vars.alpha[lay_idx] + corrected_step_size * alpha_step_size, 0, None)+opt_args['tau']*dual_vars.alpha_hat[lay_idx]
            new_alpha.append(new_alpha_k)

            ###############################################
            ############## UPDATING BETAS #################
            lin_k = weights[lay_idx - 1]
            W_k = lin_k.weights
            zk = primal_vars.zt[lay_idx-1]
            xk = primal_vars.xt[lay_idx]
            xkm1 = primal_vars.xt[lay_idx-1]
            l_preact = l_preacts[lay_idx].unsqueeze(1)
            u_preact = u_preacts[lay_idx].unsqueeze(1)
            cl_km1 = clbs[lay_idx - 1]
            cu_km1 = cubs[lay_idx - 1]

            # Compute the gradients for the Big-M variables.
            beta0_grad = xk - zk * u_preact
            beta1_grad = xk - lin_k.forward(xkm1) + (1 - zk) * l_preact

            # TODO: this is duplicated code (found in anderson_oracle). Remove?
            if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]:
                masked_op = anderson_optimization.MaskedConvOp(lin_k, xkm1, dual_vars.sum_beta[lay_idx])
                # Unfold the convolutional inputs into matrices containing the parts (slices) of the input forming the
                # convolution output.
                unfolded_cu_km1 = lin_k.unfold_input(cu_km1.unsqueeze(1))
                unfolded_cl_km1 = lin_k.unfold_input(cl_km1.unsqueeze(1))

                # The matrix whose matrix product with the unfolded input makes the convolutional output (after
                # reshaping to out_shape)
                unfolded_W_k = lin_k.unfold_weights()
                # u_check and l_check are now of size out_channels x slice_len x n_slices
                u_check = torch.where((unfolded_W_k > 0).unsqueeze(-1), unfolded_cu_km1, unfolded_cl_km1)
                l_check = torch.where((unfolded_W_k > 0).unsqueeze(-1), unfolded_cl_km1, unfolded_cu_km1)

                unfolded_xkm1 = lin_k.unfold_input(xkm1)  # input space unfolding

            else:
                # Fully connected layer.
                masked_op = anderson_optimization.MaskedLinearOp(lin_k)
                if lin_k.flatten_from_shape is not None:
                    cu_km1 = cu_km1.view(cu_km1.shape[0], -1)
                    cl_km1 = cl_km1.view(cl_km1.shape[0], -1)
                    xkm1 = xkm1.view(*xkm1.shape[:1], -1)
                u_check = torch.where(W_k > 0, cu_km1.unsqueeze(1), cl_km1.unsqueeze(1))
                l_check = torch.where(W_k > 0, cl_km1.unsqueeze(1), cu_km1.unsqueeze(1))
                if lin_k.flatten_from_shape is not None:
                    xkm1 = xkm1.view_as(dual_vars.sum_Wp1Ibetap1[lay_idx - 1])

            new_sum_betak = torch.zeros_like(dual_vars.sum_beta[lay_idx])
            new_sum_WkIbetak = torch.zeros_like(dual_vars.sum_Wp1Ibetap1[lay_idx - 1])
            new_sum_Wk1mIubetak = torch.zeros_like(dual_vars.sum_W1mIubeta[lay_idx])
            new_sum_WkIlbetak = torch.zeros_like(dual_vars.sum_WIlbeta[lay_idx])

            ##################################################################
            ###################### FOR ACTIVE SET ############################
            ##################################################################
            # print('check if I already exists?')
            ######################################

            Ik_list = dual_vars.I_list[lay_idx-1]
            ik_index=-1
            for ik_index in range(len(Ik_list)):
                # ... and its gradient
                if ik_index == 0:
                    ########### UPDATING BETA ###################
                    self.m1_sum_beta[lay_idx-1][ik_index].mul_(self.coeff1).add_(1-self.coeff1, beta0_grad)
                    self.m2_sum_beta[lay_idx-1][ik_index].mul_(self.coeff2).addcmul_(1 - self.coeff2, beta0_grad, beta0_grad)
                    sum_beta_step_size = self.m1_sum_beta[lay_idx-1][ik_index] / (self.m2_sum_beta[lay_idx-1][ik_index].sqrt() + self.epsilon)
                    new_sum_betak_ik = (1-opt_args['tau'])*torch.clamp(dual_vars.beta_list[lay_idx-1][ik_index] + corrected_step_size * sum_beta_step_size, 0, None).type(precision)+opt_args['tau']*dual_vars.beta_list[lay_idx-1][ik_index]
                    # new_sum_betak_ik = (1-opt_args['tau'])*torch.clamp(dual_vars.beta_list[lay_idx-1][ik_index] + corrected_step_size * sum_beta_step_size, 0, None).type(precision)+opt_args['tau']*dual_vars.beta_hat_list[lay_idx-1][ik_index]
                    #############################################
                    beta_WIl = 0
                    beta_W1mIu = (u_preact - lin_k.get_bias()) * new_sum_betak_ik
                    beta_WI = 0
                elif ik_index == 1:
                    ########### UPDATING BETA ###################
                    self.m1_sum_beta[lay_idx-1][ik_index].mul_(self.coeff1).add_(1-self.coeff1, beta1_grad)
                    self.m2_sum_beta[lay_idx-1][ik_index].mul_(self.coeff2).addcmul_(1 - self.coeff2, beta1_grad, beta1_grad)
                    sum_beta_step_size = self.m1_sum_beta[lay_idx-1][ik_index] / (self.m2_sum_beta[lay_idx-1][ik_index].sqrt() + self.epsilon)
                    new_sum_betak_ik = (1-opt_args['tau'])*torch.clamp(dual_vars.beta_list[lay_idx-1][ik_index] + corrected_step_size * sum_beta_step_size, 0, None).type(precision)+opt_args['tau']*dual_vars.beta_list[lay_idx-1][ik_index]
                    # new_sum_betak_ik = (1-opt_args['tau'])*torch.clamp(dual_vars.beta_list[lay_idx-1][ik_index] + corrected_step_size * sum_beta_step_size, 0, None).type(precision)+opt_args['tau']*dual_vars.beta_hat_list[lay_idx-1][ik_index]
                    ########################################################
                    beta_WIl = (l_preact - lin_k.get_bias()) * new_sum_betak_ik
                    beta_W1mIu = 0
                    beta_WI = lin_k.backward(new_sum_betak_ik)
                else:
                    masked_op.set_mask(Ik_list[ik_index])
                    WI_xkm1 = masked_op.forward(unfolded_xkm1 if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]
                                                else xkm1, add_bias=False)
                    nub_WIu = nubs[lay_idx - 1].unsqueeze(1) - masked_op.forward(u_check, bounds_matrix_in=True, add_bias=False)
                    W1mIu = nub_WIu - lin_k.get_bias()
                    WIl = masked_op.forward(l_check, bounds_matrix_in=True, add_bias=False)
                    exp_k_grad = xk - WI_xkm1 + (1 - zk) * WIl - zk * nub_WIu
                    ########### UPDATING BETA ###################
                    self.m1_sum_beta[lay_idx-1][ik_index].mul_(self.coeff1).add_(1-self.coeff1, exp_k_grad)
                    self.m2_sum_beta[lay_idx-1][ik_index].mul_(self.coeff2).addcmul_(1 - self.coeff2, exp_k_grad, exp_k_grad)
                    sum_beta_step_size = self.m1_sum_beta[lay_idx-1][ik_index] / (self.m2_sum_beta[lay_idx-1][ik_index].sqrt() + self.epsilon)
                    new_sum_betak_ik = (1-opt_args['tau'])*torch.clamp(dual_vars.beta_list[lay_idx-1][ik_index] + corrected_step_size * sum_beta_step_size, 0, None).type(precision)+opt_args['tau']*dual_vars.beta_list[lay_idx-1][ik_index]
                    # new_sum_betak_ik = (1-opt_args['tau'])*torch.clamp(dual_vars.beta_list[lay_idx-1][ik_index] + corrected_step_size * sum_beta_step_size, 0, None).type(precision)+opt_args['tau']*dual_vars.beta_hat_list[lay_idx-1][ik_index]
                    ########################################################
                    beta_WIl = WIl * new_sum_betak_ik
                    beta_W1mIu = W1mIu * new_sum_betak_ik
                    beta_WI = masked_op.backward(new_sum_betak_ik)

                dual_vars.beta_list[lay_idx-1][ik_index] = new_sum_betak_ik
                # update pre-computed backward passes.
                new_sum_betak = new_sum_betak + new_sum_betak_ik
                new_sum_WkIbetak = new_sum_WkIbetak + beta_WI
                new_sum_Wk1mIubetak = new_sum_Wk1mIubetak + beta_W1mIu
                new_sum_WkIlbetak = new_sum_WkIlbetak + beta_WIl

            # assert new_sum_betak.min() >= 0, 'sum_betak VIOLATION %f'%new_sum_betak.min()

            ########################################################
            ############ FOR LMO ###################################
            if (len(Ik_list) <= max_cuts and outer_it % cut_frequency < opt_args['cut_add']):
                self.m1_sum_beta[lay_idx-1].append(torch.zeros_like(dual_vars.sum_beta[lay_idx]))
                self.m2_sum_beta[lay_idx-1].append(torch.zeros_like(dual_vars.sum_beta[lay_idx]))
                self.m1_sum_beta[lay_idx-1][ik_index+1].mul_(self.coeff1).add_(1-self.coeff1, dual_vars_subg.sum_beta[lay_idx])
                self.m2_sum_beta[lay_idx-1][ik_index+1].mul_(self.coeff2).addcmul_(1 - self.coeff2, dual_vars_subg.sum_beta[lay_idx], dual_vars_subg.sum_beta[lay_idx])

                sum_beta_step_size = self.m1_sum_beta[lay_idx-1][ik_index+1] / (self.m2_sum_beta[lay_idx-1][ik_index+1].sqrt() + self.epsilon)
                M_atom_mask = torch.clamp(corrected_step_size * sum_beta_step_size, 0, None).type(precision)
                dual_vars.beta_list[lay_idx-1].append(M_atom_mask)
                dual_vars.beta_hat_list[lay_idx-1].append(M_atom_mask)
                dual_vars.I_list[lay_idx-1].append(dual_vars_subg.I_list[lay_idx])

                masked_op.set_mask(dual_vars_subg.I_list[lay_idx])
                beta_WI_lmo = masked_op.backward(M_atom_mask)
                beta_WIl_lmo = dual_vars_subg.sum_WIlbeta[lay_idx] * M_atom_mask
                beta_W1mIu_lmo = dual_vars_subg.sum_W1mIubeta[lay_idx] * M_atom_mask
                # update pre-computed backward passes.
                new_sum_betak = new_sum_betak + M_atom_mask
                new_sum_WkIbetak = new_sum_WkIbetak + beta_WI_lmo
                new_sum_Wk1mIubetak = new_sum_Wk1mIubetak + beta_W1mIu_lmo
                new_sum_WkIlbetak = new_sum_WkIlbetak + beta_WIl_lmo

            #########################################################
            #########################################################
            if type(weights[lay_idx - 1]) is utils.LinearOp and weights[lay_idx - 1].flatten_from_shape is not None:
                new_sum_betak = new_sum_betak.view_as(dual_vars.sum_beta[lay_idx])
                new_sum_WkIbetak = new_sum_WkIbetak.view_as(dual_vars.sum_Wp1Ibetap1[lay_idx - 1])
                new_sum_Wk1mIubetak = new_sum_Wk1mIubetak.view_as(dual_vars.sum_W1mIubeta[lay_idx])
                new_sum_WkIlbetak = new_sum_WkIlbetak.view_as(dual_vars.sum_WIlbeta[lay_idx])

            new_sum_beta.append(new_sum_betak)
            new_sum_WkIbeta.append(new_sum_WkIbetak)
            new_sum_Wk1mIubeta.append(new_sum_Wk1mIubetak)
            new_sum_WkIlbeta.append(new_sum_WkIlbetak)

        return CutDualVars(new_alpha, new_sum_beta, new_sum_WkIbeta, new_sum_Wk1mIubeta, new_sum_WkIlbeta, None, None,
                           None, None, beta_list=None, I_list=[])


class CutPrimalVars(anderson_optimization.PrimalVars):

    def __init__(self, xt, zt, eta=0.5, volume=0):
        """
        Given the primal vars as lists of tensors (of correct length), initialize the class with these.
        """
        self.xt = xt
        self.zt = zt
        self.xt_hat = copy.deepcopy(xt)
        self.zt_hat = copy.deepcopy(zt)
        self.eta = eta
        self.volume = volume

    @staticmethod
    def from_super_class(super_instance):
        """
        Return an instance of this class from an instance of the super class.
        """
        return CutPrimalVars(super_instance.xt, super_instance.zt)

    @staticmethod
    def mid_box_initialization(dual_vars, clbs, cubs):
        """
        Initialize the primal variables (anchor points) to the mid-point of the box constraints (halfway through each
        variable's lower and upper bounds).
        """
        primals = anderson_optimization.PrimalVars.mid_box_initialization(dual_vars, clbs, cubs)
        return CutPrimalVars.from_super_class(primals)

    def copy(self):
        """
        deep-copy the current instance
        :return: the copied class instance
        """
        return CutPrimalVars(copy.deepcopy(self.xt), copy.deepcopy(self.zt))

    @staticmethod
    def primalsk_min(lay_idx, dual_vars, clbs, cubs, precision):
        x_k_lmo = torch.where(dual_vars.fs[lay_idx] >= 0, cubs[lay_idx].unsqueeze(1), clbs[lay_idx].unsqueeze(1))
        if lay_idx > 0:
            z_k_lmo = (dual_vars.gs[lay_idx - 1] >= 0).type(precision)
        else:
            # g_k is defined from 1 to n - 1.
            z_k_lmo = None
        return x_k_lmo, z_k_lmo

    def update_primals_from_primalsk(self, lay_idx, new_xk, new_zk):
        """
        Given new values for alphas at layer lay_idx, update the dual variables and their functions.
        """
        self.xt[lay_idx] = new_xk
        self.xt_hat[lay_idx] = self.eta*self.xt_hat[lay_idx] + (1-self.eta)*new_xk
        if lay_idx > 0:
            self.zt[lay_idx-1] = new_zk
            self.zt_hat[lay_idx-1] = self.eta*self.zt_hat[lay_idx-1] + (1-self.eta)*new_zk
        if self.volume == 1:
            self.xt[lay_idx] = copy.deepcopy(self.xt_hat[lay_idx])
            if lay_idx > 0:
                self.zt[lay_idx-1] = copy.deepcopy(self.zt_hat[lay_idx-1])


class CutInit(anderson_optimization.AndersonPInit):
    """
    Parent Init class for Anderson-relaxation-based solvers.
    """
    def __init__(self, parent_duals, parent_primals):
        # parent_duals are the dual values (instance of DualVars) at parent termination
        super().__init__(parent_duals)
        self.primals = parent_primals

    def to_cpu(self):
        # Move content to cpu
        super().to_cpu()
        for varname in self.primals.__dict__:
            if isinstance(self.primals.__dict__[varname], list):
                self.primals.__dict__[varname] = [cvar.cpu() for cvar in self.primals.__dict__[varname]]

    def to_device(self, device):
        # Move content to device "device"
        super().to_device(device)
        for varname in self.primals.__dict__:
            if isinstance(self.primals.__dict__[varname], list):
                self.primals.__dict__[varname] = [cvar.to(device) for cvar in self.primals.__dict__[varname]]

    def as_stack(self, stack_size):
        # Repeat the content of this parent init to form a stack of size "stack_size"
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append(
                [pinits[0].unsqueeze(0).repeat(((stack_size,) + (1,) * (pinits.dim() - 1))) for pinits in varset])
        stacked_primal_list = []
        for varset in [self.primals.xt, self.primals.zt]:
            stacked_primal_list.append(
                [pinits[0].unsqueeze(0).repeat(((stack_size,) + (1,) * (pinits.dim() - 1))) for pinits in varset])
        return CutInit(bigm_optimization.DualVars(*stacked_dual_list), CutPrimalVars(*stacked_primal_list))

    def set_stack_parent_entries(self, parent_solution, batch_idx):
        # Given a solution for the parent problem (at batch_idx), set the corresponding entries of the stack.
        super().set_stack_parent_entries(parent_solution, batch_idx)
        for varname in self.primals.__dict__:
            if isinstance(self.primals.__dict__[varname], list):
                for x_idx in range(len(self.primals.__dict__[varname])):
                    self.primals.__dict__[varname][x_idx][2 * batch_idx] = parent_solution.primals.__dict__[varname][
                        x_idx].clone()
                    self.primals.__dict__[varname][x_idx][2 * batch_idx + 1] = parent_solution.primals.__dict__[varname][
                        x_idx].clone()

    def get_stack_entry(self, batch_idx):
        # Return the stack entry at batch_idx as a new ParentInit instance.
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append([csol[batch_idx].unsqueeze(0) for csol in varset])
        stacked_primal_list = []
        for varset in [self.primals.xt, self.primals.zt]:
            stacked_primal_list.append([csol[batch_idx].unsqueeze(0) for csol in varset])
        return CutInit(bigm_optimization.DualVars(*stacked_dual_list), CutPrimalVars(*stacked_primal_list))

    def get_lb_init_only(self):
        # Get instance of this class with only entries relative to LBs.
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append([c_init[:, -1].unsqueeze(1) for c_init in varset])
        stacked_primal_list = []
        for varset in [self.primals.xt, self.primals.zt]:
            stacked_primal_list.append([c_init[:, -1].unsqueeze(1) for c_init in varset])
        return CutInit(bigm_optimization.DualVars(*stacked_dual_list), CutPrimalVars(*stacked_primal_list))