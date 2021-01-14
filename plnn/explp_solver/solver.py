import itertools
import torch
from torch import nn
from plnn.proxlp_solver.solver import SaddleLP
from plnn.proxlp_solver import utils
from plnn.explp_solver import anderson_optimization, bigm_optimization, cut_anderson_optimization
from time import time


eps_tol=0.01


class ExpLP(SaddleLP):

    def __init__(
            self,
            layers,
            debug=False,
            params=None,
            use_preactivation=True,
            precision=torch.float,
            store_bounds_progress=-1,
            fixed_M=False,
            view_tensorboard=False,
            store_bounds_primal=False,
            max_batch=20000
    ):
        """
        :param store_bounds_progress: which layer to store bounds progress for. -1=don't.
        :param fixed_M: whether to keep M (artificial upper bound) fixed for alpha and beta, rather than update it dynamically.
        :param use_preactivation: whether to strengthen constraints (2 out of exponentially many) by using pre-activation
        :param bigm: whether to use the specialized bigm relaxation solver. Alone, instead of Anderson: "only". As initializer: "init"
        """
        self.layers = layers
        self.net = nn.Sequential(*layers)
        self.debug = debug
        self.params = params
        self.use_preactivation = use_preactivation
        if self.params["bigm"] and self.params["bigm"] not in ["only", "init"]:
            raise IOError('bigm argument supports only False, "only", "init"')
        if self.params["bigm"] and self.params["bigm_algorithm"] not in ["prox", "adam"]:
            raise IOError('bigm_algorithm supports only False, "prox", "adam"')
        self.bigm_init = self.params["bigm"] and (self.params["bigm"] == "init")
        self.bigm_only = self.params["bigm"] and (self.params["bigm"] == "only")
        self.cut_init = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "init")
        self.cut_only = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "only")
        self.max_batch = max_batch
        self.store_bounds_primal = store_bounds_primal
        self.external_init = None

        # whether to use only the planet constraints in the exponential problem (performs poorly)
        self.prox_dual_planet_only = False

        for param in self.net.parameters():
            param.requires_grad = False
        # Dummy placeholder for the trackers
        self.obj_tracker = None
        self.bound_tracker = None
        self.precision = precision
        self.fixed_M = fixed_M
        self.store_bounds_progress = store_bounds_progress
        self.view_tensorboard = view_tensorboard
        if self.view_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(comment="bigm_%s_cut_%s_i_%f_f_%f_cf_%d_c_%d" % (
            self.params["bigm"], self.params["cut"] , self.params['init_params']['initial_step_size'],
            self.params['init_params']['final_step_size'],
            self.params['cut_frequency'], self.params['max_cuts']))
        if self.bigm_only:
            if self.params["bigm_algorithm"] == "prox":
                self.optimize = self.bigm_prox_optimizer
                self.logger = utils.ProxOptimizationTrace()
            else:
                self.optimize = self.bigm_subgradient_optimizer
                self.logger = utils.OptimizationTrace()
        elif self.cut_only:
            self.logger = utils.OptimizationTrace()
            self.optimize = self.cut_anderson_optimizer

    def cut_anderson_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        # ADAM subgradient ascent for a specific big-M solver which operates directly in the dual space
        # hard-code default parameters, which are overridden by self.params
        opt_args = {
            "nb_inner_iter": 100,
            "nb_iter": 100,
            "alpha_M": 1e-3,
            "beta_M": 1e-3,
            "bigm_algorithm": "adam",
            ###
            'random_cuts': False,
            'cut_frequency': 400,
            'max_cuts': 10,
            'cut_add': 2,
            'eta': 0,
            'volume': 0,
            'tau': 0,
            ###
            'betas': (0.9, 0.999),
            'initial_step_size': 1e-3,
            'final_step_size': 1e-6,
            "init_params": {
                'nb_outer_iter': 100,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999),
                'M_factor': 1.0
            },
            'nb_outer_iter': 10,
            'bigm': 'init',
        }
        if self.cut_only:
            opt_args.update(self.params)
        else:
            opt_args.update(self.params["cut_init_params"])

        assert len(additional_coeffs) == 1
        assert len(weights) in additional_coeffs  # if the last layer's coefficients are present
        # if self.store_bounds_progress >= 0 and self.bigm_only:
        self.logger.start_timing()

        device = lower_bounds[-1].device
        input_size = tuple(lower_bounds[0].shape[1:])
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]

        # Build the clamped bounds
        clbs = [lower_bounds[0]] + [torch.clamp(bound, 0, None) for bound in lower_bounds[1:]]  # 0 to n-1
        cubs = [upper_bounds[0]] + [torch.clamp(bound, 0, None) for bound in upper_bounds[1:]]  # 0 to n-1

        # Build the naive bounds
        nubs = [lin_k.interval_forward(cl_k, cu_k)[1] for (lin_k, cl_k, cu_k) in zip(weights, clbs, cubs)]  # 1 to n

        alpha_M = [opt_args["alpha_M"]] * len(clbs)
        beta_M = [opt_args["beta_M"]] * len(clbs)

        if self.external_init is not None and type(self.external_init) is cut_anderson_optimization.CutInit:

            # Initialise bigm's duals from parent (then run supergradient ascent on them).
            bound_init = self.bigm_subgradient_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
            dual_vars, _ = cut_anderson_optimization.CutDualVars.bigm_initialization(
                self.bigm_dual_vars, weights, additional_coeffs, device, tuple(lower_bounds[0].shape[1:]), clbs, cubs,
                lower_bounds, upper_bounds, opt_args, alpha_M=alpha_M, beta_M=beta_M)
            # Initialise primals from parent, without any subgradient descent.
            primal_vars = self.external_init.primals

            ## Adam-related quantities.
            adam_stats = cut_anderson_optimization.DualADAMStats(dual_vars.sum_beta, beta1=opt_args['betas'][0],
                                                                 beta2=opt_args['betas'][1])
            # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))
            adam_stats.bigm_adam_initialization(dual_vars.sum_beta, self.bigm_adam_stats, beta1=opt_args['betas'][0],
                                                beta2=opt_args['betas'][1])
            # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))

        elif self.bigm_init:
            # Initialize alpha/beta with the output of the chosen big-m solver.
            if opt_args["bigm_algorithm"] == "prox":
                bounds = self.bigm_prox_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)

                print(f"Average bounds after init with Bigm prox: {bounds.mean().item()}")

                dual_vars, primal_vars = cut_anderson_optimization.CutDualVars.bigm_initialization(
                    self.bigm_dual_vars, weights, additional_coeffs, device, input_size, clbs, cubs, lower_bounds,
                    upper_bounds, opt_args, alpha_M=alpha_M, beta_M=beta_M)

                ## Adam-related quantities.
                adam_stats = cut_anderson_optimization.DualADAMStats(dual_vars.sum_beta, beta1=opt_args['betas'][0],
                                                                     beta2=opt_args['betas'][1])
                # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))

            else:
                # opt_args["bigm_algorithm"] == "adam"
                bounds = self.bigm_subgradient_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
                print(f"Average bounds after init with Bigm adam: {bounds.mean().item()}")

                dual_vars, primal_vars = cut_anderson_optimization.CutDualVars.bigm_initialization(
                    self.bigm_dual_vars, weights, additional_coeffs, device, input_size, clbs, cubs, lower_bounds,
                    upper_bounds, opt_args, alpha_M=alpha_M, beta_M=beta_M)

                ## Adam-related quantities.
                adam_stats = cut_anderson_optimization.DualADAMStats(dual_vars.sum_beta, beta1=opt_args['betas'][0],
                                                                     beta2=opt_args['betas'][1])
                # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))
                adam_stats.bigm_adam_initialization(dual_vars.sum_beta, self.bigm_adam_stats, beta1=opt_args['betas'][0],
                                                    beta2=opt_args['betas'][1])
                # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))

            if opt_args["bigm_algorithm"] == "prox":
                xt, zt = self.bigm_primal_vars
                primal_vars = cut_anderson_optimization.CutPrimalVars(xt, zt)
        else:
            # Initialize dual variables to all 0s, primals to mid-boxes.
            print('it is doing naive initialisation')
            dual_vars = cut_anderson_optimization.CutDualVars.naive_initialization(weights, additional_coeffs, device,
                                                                                   input_size, alpha_M=alpha_M,
                                                                                   beta_M=beta_M)
            primal_vars = cut_anderson_optimization.CutPrimalVars.mid_box_initialization(dual_vars, clbs, cubs)

            ## Adam-related quantities.
            adam_stats = cut_anderson_optimization.DualADAMStats(dual_vars.sum_beta, beta1=opt_args['betas'][0],
                                                                 beta2=opt_args['betas'][1])
            # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))

        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)

        bound_init = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
        print(f"Average bound at initialisation: {bound_init.mean().item()}")
        torch.max(best_bound, bound_init, out=best_bound)

        self.alpha_time = 0
        self.beta_time = 0
        self.primals_time = 0

        if self.debug:
            if self.view_tensorboard:
                self.writer.add_scalar('Average best bound', -best_bound.mean().item(),
                                       self.params["init_params"]["nb_outer_iter"])
                self.writer.add_scalar('Average bound', -bound_init.mean().item(), self.params["init_params"]["nb_outer_iter"])

        init_step_size = opt_args['initial_step_size']
        final_step_size = opt_args['final_step_size']

        for steps in itertools.count():
            if steps >= opt_args["nb_iter"]:
                break

            dual_vars_subg = cut_anderson_optimization.compute_dual_subgradient_adam(
                weights, clbs, cubs, nubs, dual_vars, primal_vars, steps, precision=torch.float, opt_args=opt_args)
            step_size = init_step_size + ((steps + 1) / opt_args["nb_iter"]) * (final_step_size - init_step_size)
            # normal subgradient ascent
            # dual_vars.projected_linear_combination(
            #     step_size, dual_vars_subg)

            # do adam for subgradient ascent
            dual_vars_subg_updated = adam_stats.update_moments_take_projected_step(
                weights, step_size, steps, dual_vars, dual_vars_subg, primal_vars, clbs, cubs, nubs, lower_bounds,
                upper_bounds, opt_args['cut_frequency'], opt_args['max_cuts'], precision=torch.float, opt_args=opt_args)
            dual_vars.update_from_step(weights, dual_vars_subg_updated)

            if self.debug:
                bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
                torch.max(best_bound, bound, out=best_bound)
                print(f"{steps} Average best bound: {best_bound.mean().item()}")
                print(f"{steps} Average bound: {bound.mean().item()}")
                if self.view_tensorboard:
                    self.writer.add_scalar('Average best bound', -best_bound.mean().item(),
                                           self.params["init_params"]["nb_outer_iter"] + steps + 1)
                    self.writer.add_scalar('Average bound', -bound.mean().item(),
                                           self.params["init_params"]["nb_outer_iter"] + steps + 1)

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                if steps % 10 == 0:
                    start_logging_time = time()
                    bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
                    torch.max(best_bound, bound, out=best_bound)
                    logging_time = time() - start_logging_time
                    self.logger.add_point(len(weights), best_bound.clone(), logging_time=logging_time)

            if not self.store_bounds_primal:
                bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
                torch.max(best_bound, bound, out=best_bound)
                print(f"Average best bound: {best_bound.mean().item()}")
                print(f"Average bound: {bound.mean().item()}")

        if self.cut_init:
            # store the dual vars and primal vars for possible future usage
            self.cut_dual_vars = dual_vars
            self.cut_primal_vars = primal_vars

        self.children_init = cut_anderson_optimization.CutInit(self.bigm_dual_vars, primal_vars)
        
        if self.store_bounds_primal:
            self.bounds_primal = primal_vars  # store the last matching primal
            bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
            torch.max(best_bound, bound, out=best_bound)
            nb_neurons = int(best_bound.shape[1]/2)
            print(f"Average LB improvement: {(best_bound - bound_init)[:, nb_neurons:].mean()}")

        bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
        return bound

    def bigm_subgradient_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        # ADAM subgradient ascent for a specific big-M solver which operates directly in the dual space

        # hard-code default parameters, which are overridden by self.params
        opt_args = {
            'nb_outer_iter': 100,
            'initial_step_size': 1e-3,
            'final_step_size': 1e-6,
            'betas': (0.9, 0.999)
        }
        if self.bigm_init or self.cut_init:
            opt_args.update(self.params['init_params'])
        else:
            opt_args.update(self.params)

        assert len(additional_coeffs) == 1
        assert len(weights) in additional_coeffs

        if self.store_bounds_progress >= 0 and self.bigm_only:
            self.logger.start_timing()

        device = lower_bounds[-1].device
        # Build the clamped bounds
        clbs = [lower_bounds[0]] + [torch.clamp(bound, 0, None) for bound in lower_bounds[1:]]  # 0 to n-1
        cubs = [upper_bounds[0]] + [torch.clamp(bound, 0, None) for bound in upper_bounds[1:]]  # 0 to n-1

        if self.external_init is not None and type(self.external_init) in [bigm_optimization.BigMPInit,
            cut_anderson_optimization.CutInit]:

            pinit = True
            dual_vars = self.external_init.duals
            bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            print(f"Bounds pinit: {bound.mean()}")
        else:
            pinit = False
            dual_vars = bigm_optimization.DualVars.naive_initialization(weights, additional_coeffs, device,
                                                                        lower_bounds[0].shape[1:])

        # Adam-related quantities.
        adam_stats = bigm_optimization.DualADAMStats(dual_vars.beta_0, beta1=opt_args['betas'][0], beta2=opt_args['betas'][1])
        init_step_size = opt_args['initial_step_size'] if not pinit else opt_args['initial_step_size_pinit']
        final_step_size = opt_args['final_step_size']

        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)

        if self.debug:
            obj_val = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            print(f"Average bound (and objective, they concide) at naive initialisation: {obj_val.mean().item()}")
            torch.max(best_bound, obj_val, out=best_bound)
            if self.view_tensorboard:
                self.writer.add_scalar('Average best bound', -best_bound.mean().item(), 0)
                self.writer.add_scalar('Average bound', -obj_val.mean().item(), 0)

        n_outer_iters = opt_args["nb_outer_iter"]
        for outer_it in itertools.count():
            if outer_it > n_outer_iters:
                break

            if self.debug:
                obj_val = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)

            dual_vars_subg = bigm_optimization.compute_dual_subgradient(
                weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)

            step_size = init_step_size + ((outer_it + 1) / n_outer_iters) * (final_step_size - init_step_size)

            # normal subgradient ascent
            # dual_vars.projected_linear_combination(
            #     step_size, dual_vars_subg, weights)

            # do adam for subgradient ascent
            adam_stats.update_moments_take_projected_step(
                weights, step_size, outer_it, dual_vars, dual_vars_subg)

            dual_vars.update_f_g(lower_bounds, upper_bounds)

            if self.debug:
                # This is the value "at convergence" of the dual problem
                obj_val = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                print(f"Average obj at the end of adam iteration {outer_it}: {obj_val.mean().item()}")
                torch.max(best_bound, obj_val, out=best_bound)
                if self.view_tensorboard:
                    self.writer.add_scalar('Average best bound', -best_bound.mean().item(), outer_it + 1)
                    self.writer.add_scalar('Average bound', -obj_val.mean().item(), outer_it + 1)

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress and self.bigm_only:
                if outer_it % 10 == 0:
                    start_logging_time = time()
                    bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                    logging_time = time() - start_logging_time
                    self.logger.add_point(len(weights), bound, logging_time=logging_time)

        # store the dual vars and primal vars for possible future usage
        self.bigm_dual_vars = dual_vars
        self.bigm_primal_vars = None
        self.bigm_adam_stats = adam_stats

        if self.bigm_only:
            self.children_init = bigm_optimization.BigMPInit(dual_vars)
            if self.store_bounds_primal:
                # Compute last matching primal (could make it a function...)
                nb_relu_layers = len(dual_vars.beta_0)
                xkm1, _ = bigm_optimization.layer_primal_linear_minimization(0, dual_vars.fs[0], None, clbs[0], cubs[0])
                zt = []
                xt = [xkm1]
                for lay_idx in range(1, nb_relu_layers):
                    # solve the inner problems.
                    xk, zk = bigm_optimization.layer_primal_linear_minimization(
                        lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1], clbs[lay_idx], cubs[lay_idx])
                    xt.append(xk)
                    zt.append(zk)
                self.bounds_primal = anderson_optimization.PrimalVars(xt, zt)

        bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
        return bound

    def get_lower_bound_network_input(self):
        """
        Return the input of the network that was used in the last bounds computation.
        Converts back from the conditioned input domain to the original one.
        Assumes that the last layer is a single neuron.
        """
        assert self.store_bounds_primal
        assert self.bounds_primal.xt[0].shape[1] in [1, 2], "the last layer must have a single neuron"
        l_0 = self.input_domain.select(-1, 0)
        u_0 = self.input_domain.select(-1, 1)
        net_input = (1/2) * (u_0 - l_0) * self.bounds_primal.xt[0].select(1, self.bounds_primal.xt[0].shape[1]-1) +\
                    (1/2) * (u_0 + l_0)
        return net_input

    def initialize_from(self, external_init):
        # setter to initialise from an external list of dual/primal variables (instance of AndersonPInit)
        is_cut = "cut" in self.params and (self.params["cut"] == "only")
        self.external_init = external_init