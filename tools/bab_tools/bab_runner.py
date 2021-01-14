import argparse
import torch
from plnn.branch_and_bound.relu_branch_and_bound import relu_bab
import plnn.branch_and_bound.utils as bab_utils
from tools.bab_tools.model_utils import load_cifar_1to1_exp, load_mnist_1to1_exp
from plnn.proxlp_solver.solver import SaddleLP
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from plnn.explp_solver.solver import ExpLP
import time
import pandas as pd
import os, copy
import math
import torch.multiprocessing as mp
import csv

# Pre-fixed parameters
pref_branching_thd = 0.2
pref_online_thd = 2
pref_kwbd_thd = 20
gpu = True
decision_bound = 0


def bab(gt_prop, verif_layers, domain, return_dict, timeout, batch_size, method, tot_iter,  parent_init,
        args, gurobi_dict=None, writer=None):
    epsilon = 1e-4

    if gpu:
        cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
        domain = domain.cuda()
    else:
        cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

    # use best of naive interval propagation and KW as intermediate bounds
    intermediate_net = SaddleLP(cuda_verif_layers, store_bounds_primal=False, max_batch=args.max_solver_batch)
    intermediate_net.set_solution_optimizer('best_naive_kw', None)
    anderson_bounds_net = None
    hard_crit = None
    prob_hard_crit = None

    # might need a smaller batch size for hard domains
    hard_batch_size = batch_size if args.hard_batch_size == -1 else args.hard_batch_size

    # Split domains into easy and hard, define two separate bounding methods to handle their last layer.
    if method in ["cut", "gurobi-anderson"]:

        # Set criteria for identifying subproblems as hard
        hard_crit = {
            "lb_threshold": 0.5,
            "depth_threshold": 0,  # 15
            "impr_threshold": 1e-1,
            "doms_len_threshold": 200,
            "auto": args.auto_strat,
            "hard_overhead": args.hard_overhead,  # assumed at full batch
        }

        # Set bounds net for easy domains.
        if method in ["cut"]:
            bigm_adam_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": int(tot_iter),  # cifar_oval: 180
                'initial_step_size': args.dualinit_init_step,  # cifar_oval: 1e-2
                'initial_step_size_pinit': args.dualinit_init_step / 10,
                'final_step_size': args.dualinit_fin_step,  # cifar_oval: 1e-4
                'betas': (0.9, 0.999)
            }
            bounds_net = ExpLP(cuda_verif_layers, params=bigm_adam_params, store_bounds_primal=True)
        else:
            bounds_net = LinearizedNetwork(verif_layers)

        # Set bounds net for hard domains.
        if method == "cut":
            anderson_iter = args.hard_iter  # 100
            explp_params = {
                "nb_iter": anderson_iter,
                'bigm': "init",
                'cut': "only",
                "bigm_algorithm": "adam",
                'cut_frequency': 450,
                'max_cuts': 8,
                'cut_add': args.cut_add,  # 2
                'betas': (0.9, 0.999),
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                "init_params": {
                    "nb_outer_iter": 500,#500 for our datasets, 1000 for cifar10_8_255
                    'initial_step_size': args.dualinit_init_step,
                    'initial_step_size_pinit': args.dualinit_init_step/10,
                    'final_step_size': args.dualinit_fin_step,
                    'betas': (0.9, 0.999),
                },
            }
            anderson_bounds_net = ExpLP(cuda_verif_layers, params=explp_params, fixed_M=True, store_bounds_primal=True)
            print(f"Running cut for {anderson_iter} iterations")
        elif method == "gurobi-anderson":
            anderson_bounds_net = AndersonLinearizedNetwork(
                verif_layers, mode="lp-cut", n_cuts=args.n_cuts, cuts_per_neuron=True, decision_boundary=decision_bound)

        if args.no_easy:
            # Ignore the easy problems bounding, use the hard one for all.
            bounds_net = anderson_bounds_net
            anderson_bounds_net = None

    # Use only a single last layer bounding method for all problems.
    elif method == "prox":
        bounds_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
        bounds_net.set_decomposition('pairs', 'KW')
        optprox_params = {
            'nb_total_steps': int(tot_iter),
            'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
            'initial_eta': args.eta,
            'final_eta': args.feta,
            'log_values': False,
            'maintain_primal': True
        }
        bounds_net.set_solution_optimizer('optimized_prox', optprox_params)
        print(f"Running prox with {tot_iter} steps")
    elif method == "adam":
        bounds_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
        bounds_net.set_decomposition('pairs', 'KW')
        adam_params = {
            'nb_steps': int(tot_iter),
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (0.9, 0.999),
            'log_values': False
        }
        bounds_net.set_solution_optimizer('adam', adam_params)
        print(f"Running adam with {tot_iter} steps")
    elif method == "bigm-adam":
        bigm_adam_params = {
            "bigm_algorithm": "adam",
            "bigm": "only",
            "nb_outer_iter": int(tot_iter),
            'initial_step_size': args.init_step,
            'initial_step_size_pinit': args.init_step/10,
            'final_step_size': args.fin_step,
            'betas': (0.9, 0.999)
        }
        bounds_net = ExpLP(cuda_verif_layers, params=bigm_adam_params, store_bounds_primal=True)
    elif method == "gurobi":
        bounds_net = LinearizedNetwork(verif_layers)

    # branching
    if args.branching_choice == 'heuristic':
        branching_net_name = None
    else:
        raise NotImplementedError
  
    # try:
    with torch.no_grad():
        min_lb, min_ub, ub_point, nb_states, fail_safe_ratio = relu_bab(
            intermediate_net, bounds_net, branching_net_name, domain, decision_bound, eps=epsilon,
            timeout=timeout, batch_size=batch_size, parent_init_flag=parent_init, gurobi_specs=gurobi_dict,
            anderson_bounds_net=anderson_bounds_net, writer=writer, hard_crit=hard_crit,
            hard_batch_size=hard_batch_size)

    if not (min_lb or min_ub or ub_point):
        return_dict["min_lb"] = None;
        return_dict["min_ub"] = None;
        return_dict["ub_point"] = None;
        return_dict["nb_states"] = nb_states
        return_dict["bab_out"] = "timeout"
        return_dict["fs_ratio"] = fail_safe_ratio
    else:
        return_dict["min_lb"] = min_lb.cpu()
        return_dict["min_ub"] = min_ub.cpu()
        return_dict["ub_point"] = ub_point.cpu()
        return_dict["nb_states"] = nb_states
        return_dict["fs_ratio"] = fail_safe_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', help='file to save results')
    parser.add_argument('--record_name', type=str, help='file to save results')
    parser.add_argument('--pdprops', type=str, help='pandas table with all props we are interested in')
    parser.add_argument('--timeout', type=int)
    parser.add_argument('--cpus_total', type=int, help='total number of cpus used')
    parser.add_argument('--cpu_id', type=int, help='the index of the cpu from 0 to cpus_total')
    parser.add_argument('--nn_name', type=str, help='network architecture name')
    parser.add_argument('--data', type=str, default='cifar')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--batch_size', type=int, help='batch size / 2 for how many domain computations in parallel')
    parser.add_argument('--gurobi_p', type=int, help='number of threads to use in parallelizing gurobi over domains, '
                                                     'or running the MIP solver', default=1)
    parser.add_argument('--method', type=str, choices=["prox", "adam", "gurobi", "gurobi-anderson",
        "cut", "anderson-mip", "bigm-adam"], help='method to employ for bounds (or MIP)')
    parser.add_argument('--branching_choice', type=str, choices=['heuristic'],
                        help='type of branching choice used', default='heuristic')
    parser.add_argument('--tot_iter', type=float, help='how many total iters to use for the method', default=100)
    parser.add_argument('--max_solver_batch', type=float, default=10000, help='max batch size for bounding computations')
    parser.add_argument('--parent_init', action='store_true', help='whether to initialize the code from the parent')
    parser.add_argument('--n_cuts', type=int, help='number of anderson cuts to employ (per neuron)')
    parser.add_argument('--eta', type=float)
    parser.add_argument('--feta', type=float)
    parser.add_argument('--init_step', type=float, default=1e-3)
    parser.add_argument('--fin_step', type=float, default=1e-6)

    # Anderson-based bounding methods parameters.
    parser.add_argument('--no_easy', action='store_true', help='whether to avoid the two-way bound system (easy+hard) '
                                                               'for anderson-based bounding algos')
    parser.add_argument('--auto_strat', action='store_true', help='whether to infer stratification parameters')
    parser.add_argument('--hard_iter', type=float, default=100)
    parser.add_argument('--hard_overhead', type=float, default=100)   # hard bounding overhead at full batch
    parser.add_argument('--cut_add', type=float, default=2)
    parser.add_argument('--fw_start', type=float, default=10)
    parser.add_argument('--fw_cut_iters', type=int, default=100)
    parser.add_argument('--dualinit_init_step', type=float, default=1e-2)
    parser.add_argument('--dualinit_fin_step', type=float, default=1e-4)
    parser.add_argument('--primalinit_init_step', type=float, default=1e-1)
    parser.add_argument('--primalinit_fin_step', type=float, default=1e-3)
    parser.add_argument('--hard_batch_size', type=int, default=-1)

    args = parser.parse_args()

    # initialize a file to record all results, record should be a pandas dataframe
    if args.data == 'cifar' or args.data=='cifar10':
        path = './verification_datasets/'
        result_path = './cifar_results/'

        if not os.path.exists(result_path):
            os.makedirs(result_path)

    elif args.data == 'mnist':
        path = './verification_datasets/'
        result_path = './mnist_results/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        raise NotImplementedError

    # load all properties
    if args.data=='mnist' or args.data=='cifar10':
        csvfile = open('././data/%s_test.csv'%(args.data), 'r')
        tests = list(csv.reader(csvfile, delimiter=','))
        batch_ids = range(100)
        batch_ids_run = batch_ids
        enum_batch_ids = [(bid, None) for bid in batch_ids_run]
    elif args.data in ['cifar']:
        gt_results = pd.read_pickle(path + args.pdprops)
        bnb_ids = gt_results.index
        batch_ids = bnb_ids
        enum_batch_ids = enumerate(batch_ids)

    if args.record:
        if args.record_name is not None:
            record_name = args.record_name
        else:
            method_name = ''
            columns = ["Idx", "Eps", "prop"]

            parent_init = "-pinit" if args.parent_init else ""
            algo_string = ""
            if args.method == "prox":
                algo_string += f"-eta{args.eta}-feta{args.feta}"
            elif args.method in ["adam", "cut", "bigm-adam"]:
                algo_string += f"-ilr{args.init_step},flr{args.fin_step}"
            if "cut" in args.method:
                algo_string += f"-cut_add{args.cut_add}"
            if args.method in ["cut"]:
                algo_string += f"-diilr{args.dualinit_init_step},diflr{args.dualinit_fin_step}"

            if args.method not in ["gurobi", "gurobi-anderson", "cut", "anderson-mip"]:
                algorithm_name = f"{args.method}_{int(args.tot_iter)}"
            elif args.method in ["cut"]:
                algorithm_name = f"{args.method}_{int(args.hard_iter)}"
            elif args.method == "gurobi-anderson":
                algorithm_name = f'{args.method}_{args.n_cuts}'
            else:
                algorithm_name = f'{args.method}'

            add_flags = "_no_easy" if args.no_easy else ""
            add_flags += "_auto_strat" if args.auto_strat else ""
            algorithm_name += add_flags
            
            # branching choices
            if args.method == "anderson-mip":
                method_name += f'{algorithm_name}'
                columns += [f'BSAT_{algorithm_name}', f'BBran_{algorithm_name}', f'BTime_{algorithm_name}']
            elif args.branching_choice=='heuristic':
                method_name += f'KW_{algorithm_name}{parent_init}{algo_string}'
                columns += [f'BSAT_KW_{algorithm_name}', f'BBran_KW_{algorithm_name}', f'BTime_KW_{algorithm_name}']

            if args.data=='mnist' or args.data=='cifar10':
                base_name = f'{args.nn_name}'
            else:
                base_name = f'{args.pdprops[:-4]}'
            record_name = result_path + f'{base_name}_{method_name}.pkl'

        if os.path.isfile(record_name):
            graph_df = pd.read_pickle(record_name)
        else:
            indices = list(range(len(batch_ids)))

            graph_df = pd.DataFrame(index=indices, columns=columns)
            graph_df.to_pickle(record_name)

            # skip = False

    if args.method in ["gurobi", "gurobi-anderson"]:
        if args.gurobi_p > 1:
            mp.set_start_method('spawn')  # for some reason, everything hangs w/o this
    gurobi_dict = {"gurobi": args.method in ["gurobi", "gurobi-anderson"], "p": args.gurobi_p}

    for new_idx, idx in enum_batch_ids:
        # record_info
        if args.record:
            # print(record_name)
            graph_df = pd.read_pickle(record_name)
            if pd.isna(graph_df.loc[new_idx]['Eps']) == False:
                print(f'the {new_idx}th element is done')
                # skip = True
                continue
        # if skip == True:
        #    print(f'skip the {new_idx}th element')
        #    skip = False
        #    continue

        if args.data in ['cifar']:

            imag_idx = gt_results.loc[idx]["Idx"]
            prop_idx = gt_results.loc[idx]['prop']
            eps_temp = gt_results.loc[idx]["Eps"]

            # skip the nan prop_idx or eps_temp (happens in wide.pkl, jodie's mistake, I guess)
            if (math.isnan(imag_idx) or math.isnan(prop_idx) or math.isnan(eps_temp)):
                continue

            x, verif_layers, test = load_cifar_1to1_exp(args.nn_name, int(imag_idx), int(prop_idx))
            # since we normalise cifar data set, it is unbounded now
            assert test == prop_idx
            domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)

        else:
            raise NotImplementedError

        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter(comment="%d_cut_"%(prop_idx))
        writer = None

        ### BaB
        bab_start = time.time()
        if args.method != "anderson-mip":
            gt_prop = f'idx_{imag_idx}_prop_{prop_idx}_eps_{eps_temp}'
            print(gt_prop)
            return_dict = dict()
            bab(gt_prop, verif_layers, domain, return_dict, args.timeout, args.batch_size, args.method,
                args.tot_iter, args.parent_init, args, gurobi_dict=gurobi_dict, writer=writer)

            bab_min_lb = return_dict["min_lb"]
            bab_min_ub = return_dict["min_ub"]
            bab_ub_point = return_dict["ub_point"]
            bab_nb_states = return_dict["nb_states"]
            bab_fs = return_dict["fs_ratio"]
            if bab_min_lb is None:
                if "bab_out" in return_dict:
                    bab_out = return_dict["bab_out"]
                else:
                    bab_out = 'grbError'
            else:
                if bab_min_lb >= 0:
                    print("UNSAT")
                    bab_out = "False"
                elif bab_min_ub < 0:
                    # Verify that it is a valid solution
                    print("SAT")
                    bab_out = "True"
                else:
                    print("Unknown")
                    #import pdb;
                    #pdb.set_trace()
                    bab_out = 'ET'
        else:
            # Run MIP with Anderson cuts.
            if gpu:
                cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
                domain = domain.cuda()
            else:
                cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

            # use best of naive interval propagation and KW as intermediate bounds
            intermediate_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
            intermediate_net.set_solution_optimizer('best_naive_kw', None)
            intermediate_net.define_linear_approximation(domain.unsqueeze(0))

            anderson_mip_net = AndersonLinearizedNetwork(
                verif_layers, mode="mip-exact", n_cuts=args.n_cuts, decision_boundary=decision_bound)

            cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab_utils.subproblems_to_cpu(
                domain, intermediate_net.lower_bounds, intermediate_net.upper_bounds, squeeze_interm=True)
            anderson_mip_net.build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs),
                                                      n_threads=args.gurobi_p)

            # sat_status, global_lb, bab_nb_states = anderson_mip_net.solve_mip(timeout=args.timeout)
            sat_status, global_lb, bab_nb_states = anderson_mip_net.solve_mip(timeout=args.timeout, insert_cuts=True)

            bab_out = str(sat_status) if sat_status is not None else "timeout"
            print(f"MIP SAT status: {bab_out}")

        print(f"Nb states visited: {bab_nb_states}")
        # print('bnb takes: ', bnb_time)
        print('\n')

        bab_end = time.time()
        bab_time = bab_end - bab_start
        print('total time required: ', bab_time)

        print('\n')

        if args.record:
            graph_df.loc[new_idx]["Idx"] = imag_idx
            graph_df.loc[new_idx]["Eps"] = eps_temp
            graph_df.loc[new_idx]["prop"] = prop_idx

            if args.method == "anderson-mip":
                graph_df.loc[new_idx][f"BSAT_{algorithm_name}"] = bab_out
                graph_df.loc[new_idx][f"BBran_{algorithm_name}"] = bab_nb_states
                graph_df.loc[new_idx][f"BTime_{algorithm_name}"] = bab_time
            elif args.branching_choice == "heuristic":
                graph_df.loc[new_idx][f"BSAT_KW_{algorithm_name}"] = bab_out
                graph_df.loc[new_idx][f"BBran_KW_{algorithm_name}"] = bab_nb_states
                graph_df.loc[new_idx][f"BTime_KW_{algorithm_name}"] = bab_time
            graph_df.to_pickle(record_name)


if __name__ == '__main__':
    main()
