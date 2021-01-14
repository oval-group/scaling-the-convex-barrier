import torch
import copy

import plnn.branch_and_bound.utils as bab
from plnn.branch_and_bound.branching_scores import BranchingChoice
import time
from math import floor, ceil


class ReLUDomain:
    '''
    Object representing a domain where the domain is specified by decision
    assigned to ReLUs.
    Comparison between instances is based on the values of
    the lower bound estimated for the instances.

    The domain is specified by `mask` which corresponds to a pattern of ReLUs.
    Neurons mapping to a  0 value are assumed to always have negative input (0 output slope)
          "               1                    "             positive input (1 output slope).
          "               -1 value are considered free and have no assumptions.

    For a MaxPooling unit, -1 indicates that we haven't picked a dominating input
    Otherwise, this indicates which one is the dominant one
    '''
    def __init__(self, mask, lb=-float('inf'), ub=float('inf'), lb_all=None, up_all=None, parent_solution=None,
                 parent_ub_point=None, parent_depth=0, c_imp=0, c_imp_avg=0, dec_thr=0, hard_criteria=None):
        self.mask = mask
        self.lower_bound = lb
        self.upper_bound = ub
        self.lower_all = lb_all
        self.upper_all = up_all
        self.parent_solution = parent_solution
        self.parent_ub_point = parent_ub_point
        self.depth = parent_depth + 1

        # keep running improvement average
        avg_coeff = 0.5  # need to react swiftly to branching decays
        self.impr_avg = (1 - avg_coeff) * c_imp_avg + avg_coeff * c_imp if c_imp_avg != 0 else c_imp

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound

    def to_cpu(self):
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.mask = [msk.cpu() for msk in self.mask]
        self.lower_bound = self.lower_bound.cpu()
        self.upper_bound = self.upper_bound.cpu()
        self.lower_all = [lbs.cpu() for lbs in self.lower_all]
        self.upper_all = [ubs.cpu() for ubs in self.upper_all]
        if self.parent_solution is not None:
            self.parent_solution.to_cpu()
        if self.parent_ub_point is not None:
            self.parent_ub_point = self.parent_ub_point.cpu()
        return self

    def to_device(self, device):
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.mask = [msk.to(device) for msk in self.mask]
        self.lower_bound = self.lower_bound.to(device)
        self.upper_bound = self.upper_bound.to(device)
        self.lower_all = [lbs.to(device) for lbs in self.lower_all]
        self.upper_all = [ubs.to(device) for ubs in self.upper_all]
        if self.parent_solution is not None:
            self.parent_solution.to_device(device)
        if self.parent_ub_point is not None:
            self.parent_ub_point = self.parent_ub_point.to(device)
        return self


def relu_bab(intermediate_net, bounds_net, branching_net_name, domain, decision_bound, eps=1e-4, sparsest_layer=0,
             timeout=float("inf"), batch_size=5, parent_init_flag=True, gurobi_specs=None,
             branching_threshold=0.2, anderson_bounds_net=None, writer=None, hard_crit=None, hard_batch_size=5):
    '''
    Uses branch and bound algorithm to evaluate the global minimum
    of a given neural network. Splits according to KW.
    Does ReLU activation splitting (not domain splitting, the domain will remain the same throughout)

    Assumes that the last layer is a single neuron.

    `intermediate_net`: Neural Network class, defining the `get_upper_bound`, `define_linear_approximation` functions.
                        Network used to get intermediate bounds.
    `bounds_net`      : Neural Network class, defining the `get_upper_bound`, `define_linear_approximation` functions.
                        Network used to get the final layer bounds, given the intermediate ones.
    `eps`           : Maximum difference between the UB and LB over the minimum
                      before we consider having converged
    `decision_bound`: If not None, stop the search if the UB and LB are both
                      superior or both inferior to this value.
    `batch_size`: The number of domain lower/upper bounds computations done in parallel at once (on a GPU) is
                    batch_size*2
    `parent_init_flag`: whether to initialize every optimization from its parent node
    `gurobi_specs`: dictionary containing whether ("gurobi") gurobi needs to be used (executes on "p" cpu)
    'hard_crit': dictionary containing the hardness criteria for subproblems
    Returns         : Lower bound and Upper bound on the global minimum,
                      as well as the point where the upper bound is achieved
    '''
    nb_visited_states = 0
    fail_safe_ratio = -1
    hard_task = False
    start_time = time.time()

    if gurobi_specs:
        gurobi_dict = dict(gurobi_specs)
        p = gurobi_dict["p"]
        gurobi = gurobi_dict["gurobi"]
    else:
        p = 1
        gurobi = False
    if gurobi and p > 1:
        send_nets = bounds_net if anderson_bounds_net is None else (bounds_net, anderson_bounds_net)
        cpu_servers, server_queue, instruction_queue, barrier = bab.spawn_cpu_servers(p, send_nets)
        gurobi_dict.update({'server_queue': server_queue, 'instruction_queue': instruction_queue,
                            'barrier': barrier, 'cpu_servers': cpu_servers})
    else:
        gurobi_dict.update({'server_queue': None, 'instruction_queue': None, 'barrier': None, 'cpu_servers': None})

    # do initial computation for the network as it is (batch of size 1: there is only one domain)
    # get intermediate bounds
    intermediate_net.define_linear_approximation(domain.unsqueeze(0))
    intermediate_lbs = copy.deepcopy(intermediate_net.lower_bounds)
    intermediate_ubs = copy.deepcopy(intermediate_net.upper_bounds)

    if intermediate_lbs[-1] > decision_bound or intermediate_ubs[-1] < decision_bound:
        bab.join_children(gurobi_dict, timeout)
        return intermediate_lbs[-1], intermediate_ubs[-1], \
               intermediate_net.get_lower_bound_network_input(), nb_visited_states, fail_safe_ratio

    print('computing last layer bounds')
    # compute last layer bounds with a more expensive network
    if not gurobi:
        bounds_net.build_model_using_bounds(domain.unsqueeze(0), (intermediate_lbs, intermediate_ubs))
    else:
        cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab.subproblems_to_cpu(
            domain, intermediate_lbs, intermediate_ubs, squeeze_interm=True)
        bounds_net.build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs))
    global_lb, global_ub = bounds_net.compute_lower_bound(counterexample_verification=True)

    # Stratified bounding related.
    if anderson_bounds_net is not None and hard_crit["auto"]:
        # Automatically infer stratification parameters by estimating the hard bounding gain.
        if not gurobi:
            anderson_bounds_net.build_model_using_bounds(domain.unsqueeze(0), (intermediate_lbs, intermediate_ubs))
        else:
            anderson_bounds_net.build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs))
        updated_lb = anderson_bounds_net.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
        hard_lb_impr = (updated_lb - global_lb).mean()
        print(f"Hard bounding improvement over looser LB: {hard_lb_impr}")
        hard_crit["hard_lb_impr"] = hard_lb_impr.cpu()

    intermediate_lbs[-1] = global_lb
    intermediate_ubs[-1] = global_ub
    bounds_net_device = global_lb.device
    intermediate_net_device = domain.device

    # retrieve bounds info from the bounds network
    global_ub_point = bounds_net.get_lower_bound_network_input()
    global_ub = bounds_net.net(global_ub_point)

    # retrieve which relus are active/passing/ambiguous
    bounds_net.relu_mask = [c_mask.to(bounds_net_device) for c_mask in intermediate_net.relu_mask]
    updated_mask = intermediate_net.relu_mask
    parent_init = bounds_net.children_init.get_lb_init_only()  # we don't use the UB as init

    print(f"Global LB: {global_lb}; Global UB: {global_ub}")
    print('decision bound', decision_bound)
    if global_lb > decision_bound or global_ub < decision_bound:
        bab.join_children(gurobi_dict, timeout)
        return global_lb, global_ub, global_ub_point, nb_visited_states, fail_safe_ratio

    candidate_domain = ReLUDomain(updated_mask, lb=global_lb, ub=global_ub, lb_all=intermediate_lbs,
                                  up_all=intermediate_ubs, parent_solution=parent_init, dec_thr=decision_bound,
                                  hard_criteria=hard_crit).to_cpu()

    domains = [candidate_domain]
    anderson_domains = []
    anderson = False
    anderson_buffer = False
    # initialise branching related terms
    branching_tools = BranchingChoice(updated_mask, sparsest_layer, intermediate_net.weights, branching_net_name)

    infeasible_count = 0
    n_iter = 0
    while global_ub - global_lb > eps:
        print(f"New batch at {time.time() - start_time}[s]")
        n_iter += 1
        # Check if we have run out of time.
        if time.time() - start_time > timeout:
            bab.join_children(gurobi_dict, timeout)
            return None, None, None, nb_visited_states, fail_safe_ratio

        ## since branching decisions are processed in batches, we collect domain info with the following lists
        orig_ub_stacks_current = []
        orig_lb_stacks_current = []
        orig_mask_stacks_current = []

        print(f"Number of domains {len(domains)} Number of anderson domains {len(anderson_domains)}")
        # Determine whether the current verification problem is hard (once it's set as such, it'll remain so)
        hard_task = bab.is_hard_problem(domains, hard_crit, hard_task)

        effective_batch_size = min(batch_size, len(domains))

        if writer is not None:
            writer.add_scalar('domains', len(domains), n_iter)
            writer.add_scalar('anderson_domains', len(anderson_domains), n_iter)
        print(f"effective_batch_size {effective_batch_size}")

        # effective_batch_size*2 as every candidate domain is split in two different ways
        splitted_lbs_stacks = [lbs[0].to(intermediate_net_device).unsqueeze(0).repeat(((effective_batch_size*2,) + (1,) * (lbs.dim() - 1)))
                               for lbs in bounds_net.lower_bounds]
        splitted_ubs_stacks = [ubs[0].to(intermediate_net_device).unsqueeze(0).repeat(((effective_batch_size*2,) + (1,) * (ubs.dim() - 1)))
                               for ubs in bounds_net.upper_bounds]
        splitted_domain = domain.unsqueeze(0).expand(((effective_batch_size*2,) + (-1,) * domain.dim()))
        depth_list = []
        parent_lb_list = []
        impr_avg_list = []

        domains[0].parent_solution.to_device(intermediate_net_device)
        parent_init_stacks = domains[0].parent_solution.as_stack(effective_batch_size*2)

        # List of sets storing for each layer index, the batch entries that are splitting a ReLU there
        branching_layer_log = []
        for _ in range(len(intermediate_net.lower_bounds)-1):
            branching_layer_log.append(set())

       ###picks not 1 problem but effective_batch_size number of problems/domains.
        for batch_idx in range(effective_batch_size):
            # Pick a domain to branch over and remove that from our current list of
            # domains. Also, potentially perform some pruning on the way.
            candidate_domain = bab.pick_out(domains, global_ub.cpu() - eps).to_device(intermediate_net_device)
            # Generate new, smaller domains by splitting over a ReLU
            mask = candidate_domain.mask
            orig_lbs = candidate_domain.lower_all
            orig_ubs = candidate_domain.upper_all

            # collect branching related information
            # print(f"node depth: {candidate_domain.depth}")
            # print(f"node impr avg: {candidate_domain.impr_avg} -- current impr {candidate_domain.c_imp}")
            depth_list.extend([candidate_domain.depth, candidate_domain.depth])
            parent_lb_list.extend([candidate_domain.lower_bound, candidate_domain.lower_bound])
            impr_avg_list.extend([candidate_domain.impr_avg, candidate_domain.impr_avg])
            orig_lb_stacks_current.append(orig_lbs)
            orig_ub_stacks_current.append(orig_ubs)
            orig_mask_stacks_current.append(mask)

            # get parent's dual solution from the candidate domain
            parent_init_stacks.set_stack_parent_entries(candidate_domain.parent_solution, batch_idx)

        # Compute branching choices
        # branching will return IndexError in case no ambiguous ReLU is left (won't happen for large nets).
        branching_decision_list, _ = branching_tools.heuristic_branching_decision(orig_lb_stacks_current, orig_ub_stacks_current, orig_mask_stacks_current)

        for batch_idx, branching_decision in enumerate(branching_decision_list):
            branching_layer_log[branching_decision[0]] |= {2*batch_idx, 2*batch_idx+1}
            orig_lbs  =  orig_lb_stacks_current[batch_idx]
            orig_ubs  =  orig_ub_stacks_current[batch_idx]

            for choice in [0, 1]:
                # print(f'splitting decision: {branching_decision} - choice {choice}')
                # Find the upper and lower bounds on the minimum in the domain
                # defined by n_mask_i
                nb_visited_states += 1

                # split the domain with the current branching decision
                splitted_lbs_stacks, splitted_ubs_stacks = update_bounds_from_split(
                    branching_decision, choice, orig_lbs, orig_ubs, 2*batch_idx + choice, splitted_lbs_stacks,
                    splitted_ubs_stacks)

        print(f"Running Nb states visited: {nb_visited_states}")
        print(f"N. infeasible nodes {infeasible_count}")

        relu_start = time.time()
        # compute the bounds on the batch of splits, at once
        dom_ub_temp, dom_lb_temp, dom_ub_point_temp, updated_mask_temp, dom_lb_all_temp, dom_ub_all_temp, dual_solutions_temp = compute_bounds(
            intermediate_net, bounds_net, branching_layer_log, splitted_domain, splitted_lbs_stacks, splitted_ubs_stacks,
            parent_init_stacks, parent_init_flag, gurobi_dict
        )

        dom_ub=dom_ub_temp; dom_lb=dom_lb_temp; dom_ub_point=dom_ub_point_temp
        updated_mask= updated_mask_temp; dom_lb_all= dom_lb_all_temp
        dom_ub_all= dom_ub_all_temp; dual_solutions= dual_solutions_temp

        # update the global upper bound (if necessary) comparing to the best of the batch
        batch_ub, batch_ub_point_idx = torch.min(dom_ub, dim=0)
        batch_ub_point = dom_ub_point[batch_ub_point_idx]
        if batch_ub < global_ub:
            global_ub = batch_ub
            global_ub_point = batch_ub_point

        for batch_idx in range(updated_mask[0].shape[0]):
            current_tot_ambi_nodes = 0
            for layer_mask in updated_mask:
                current_tot_ambi_nodes += torch.sum(layer_mask[batch_idx] == -1).item()
            # print(f"total number of ambiguous nodes: {current_tot_ambi_nodes}")

        # sequentially add all the domains to the queue (ordered list)
        batch_global_lb = dom_lb[0]
        for batch_idx in range(dom_lb.shape[0]):
            print('dom_lb: ', dom_lb[batch_idx])
            print('dom_ub: ', dom_ub[batch_idx])

            if dom_lb[batch_idx] == float('inf') or dom_ub[batch_idx] == float('inf') or \
                    dom_lb[batch_idx] > dom_ub[batch_idx]:
                infeasible_count += 1

            elif dom_lb[batch_idx] < min(global_ub, decision_bound):
                c_dom_lb_all = [lb[batch_idx].unsqueeze(0) for lb in dom_lb_all]
                c_dom_ub_all = [ub[batch_idx].unsqueeze(0) for ub in dom_ub_all]
                c_updated_mask = [msk[batch_idx].unsqueeze(0) for msk in updated_mask]

                c_dual_solutions = dual_solutions.get_stack_entry(batch_idx)
                dom_to_add = ReLUDomain(
                    c_updated_mask, lb=dom_lb[batch_idx].unsqueeze(0), ub=dom_ub[batch_idx].unsqueeze(0),
                    lb_all=c_dom_lb_all, up_all=c_dom_ub_all, parent_solution=c_dual_solutions,
                    parent_depth=depth_list[batch_idx], c_imp_avg=impr_avg_list[batch_idx],
                    c_imp=dom_lb[batch_idx].item() - parent_lb_list[batch_idx].item(), dec_thr=decision_bound,
                    hard_criteria=hard_crit
                ).to_cpu()

                # the hard queue is filled only if this problem has been marked as hard
                filling_hard_batch = anderson_bounds_net is not None and not (anderson or anderson_buffer) and hard_task
                # if the problem is hard, add "difficult" domains to the hard queue
                if filling_hard_batch and bab.is_difficult_domain(dom_to_add, hard_crit, global_ub, decision_bound):
                    bab.add_domain(dom_to_add, anderson_domains)
                else:
                    if anderson_buffer:
                        # use a buffer so that when the buffer is empty, we can use the hard problem's parent init
                        bab.add_domain(dom_to_add, anderson_domains)
                    else:
                        bab.add_domain(dom_to_add, domains)

                batch_global_lb = min(dom_lb[batch_idx], batch_global_lb)

        relu_end = time.time()
        print('A batch of relu splits requires: ', relu_end - relu_start)
        if writer is not None:
            writer.add_scalar('relu-splits_time', relu_end - relu_start, n_iter)
        # Update global LB.
        if len(domains) + len(anderson_domains) > 0:
            lb_candidate = anderson_domains[0] if anderson_domains else domains[0]
            lb_candidate = min(lb_candidate, domains[0]) if domains else lb_candidate
            global_lb = lb_candidate.lower_bound.to(bounds_net_device)
        else:
            # If we've run out of domains, it means we included no newly splitted domain
            global_lb = torch.ones_like(global_lb) * (decision_bound + eps) if batch_global_lb > global_ub \
                else batch_global_lb
        # Remove domains clearly on the right side of the decision threshold: our goal is to which side of it is the
        # minimum, no need to know more for these domains.
        prune_value = min(global_ub.cpu() - eps, decision_bound + eps)
        domains = bab.prune_domains(domains, prune_value)

        if anderson_domains or anderson_buffer:
            anderson_domains = bab.prune_domains(anderson_domains, prune_value)

        if len(domains) == 0 and anderson_bounds_net is not None and (not anderson):
            domains = anderson_domains
            anderson_domains = []
            # Check whether it's worth doing anderson domains
            print('shifting to anderson domains')
            anderson_bounds_net.lower_bounds = bounds_net.lower_bounds
            anderson_bounds_net.upper_bounds = bounds_net.upper_bounds
            bounds_net = anderson_bounds_net
            batch_size = hard_batch_size
            if parent_init_flag and not anderson_buffer:
                # use a buffer so that when the buffer is empty, we can use the hard problem's parent init
                anderson_buffer = True
            else:
                anderson = True
                if parent_init_flag:
                    anderson_buffer = False
            if gurobi:
                bab.gurobi_switch_bounding_net(gurobi_dict)

        print(f"Current: lb:{global_lb}\t ub: {global_ub}")
        if writer is not None:
            writer.add_scalar('lower_bound', global_lb, n_iter)
            writer.add_scalar('upper_bound', global_ub, n_iter)
        # Stopping criterion
        if global_lb >= decision_bound:
            break
        elif global_ub < decision_bound:
            break

    bab.join_children(gurobi_dict, timeout)

    print(f"Terminated in {time.time() - start_time}[s]; {nb_visited_states} nodes.")
    print(f"Infeasible count: {infeasible_count}")

    return global_lb, global_ub, global_ub_point, nb_visited_states, fail_safe_ratio


def update_bounds_from_split(decision, choice, old_lbs, old_ubs, batch_idx, splitted_lbs_stacks, splitted_ubs_stacks):
    """
    Given a ReLU branching decision and bounds for all the activations, clip the bounds according to the decision.
    Update performed in place in the list of lower/upper bound stacks (batches of lower/upper bounds)
    :param decision: tuples (x_idx, node)
    :param choice: 0/1 for whether to clip on a blocking/passing ReLU
    :param old_lbs: list of tensors for the (pre-activation) lower bounds relative to all the activations of the network
    :param old_ubs:list of tensors for the (pre-activation) upper bounds relative to all the activations of the network
    :param splitted_lbs_stacks: batched lower bounds to update with the splitted ones at batch_idx
    :param splitted_ubs_stacks: batched upper bounds to update with the splitted ones at batch_idx
    """
    new_ubs = copy.deepcopy(old_ubs)
    new_lbs = copy.deepcopy(old_lbs)

    assert new_lbs[0].shape[0] == 1

    if decision is not None:
        change_idx = decision[0] + 1
        # upper_bound for the corresponding relu is forced to be 0
        if choice == 0:
            # blocking ReLU obtained by setting the pre-activation UB to 0
            new_ubs[change_idx].view(-1)[decision[1]] = 0
        else:
            # passing ReLU obtained by setting the pre-activation LB to 0
            new_lbs[change_idx].view(-1)[decision[1]] = 0

    for x_idx in range(len(splitted_lbs_stacks)):
        splitted_lbs_stacks[x_idx][batch_idx] = new_lbs[x_idx]
        splitted_ubs_stacks[x_idx][batch_idx] = new_ubs[x_idx]
    return splitted_lbs_stacks, splitted_ubs_stacks


def compute_bounds(intermediate_net, bounds_net, branching_layer_log, splitted_domain, splitted_lbs,
                   splitted_ubs, parent_init_stacks, parent_init_flag, gurobi_dict):
    """
    Split domain according to branching decision and compute all the necessary quantities for it.
    Splitting on the input domain will never happen as it'd be done on l1-u1, rather than l0-u0 (representing the
    conditioned input domain). So conditioning is not problematic, here.
    :param intermediate_net: Network used for intermediate bounds
    :param bounds_net: Network used for last bounds
    :param branching_layer_log: List of sets storing for each layer index, the set of batch entries that are
        splitting a ReLU there (stored like x_idx-1)
    :param choice: 0/1 for whether to clip on a blocking/passing ReLU
    :param splitted_lbs: list of tensors for the (pre-activation) lower bounds relative to all the activations of the
    network, for all the domain batches
    :param splitted_ubs:list of tensors for the (pre-activation) upper bounds relative to all the activations of the
        network, for all the domain batches
    :param parent_init_stacks:list of tensors to use as dual variable initialization in the last layer solver
    :return: domain UB, domain LB, net input point that yielded UB, updated ReLU mask, updated old_lbs, updated old_ubs
    :param parent_init_flag: whether to initialize the bounds optimisation from the parent node
    :param gurobi_dict: dictionary containing information for gurobi's (possibly parallel) execution
    """
    # update intermediate bounds after the splitting
    splitted_lbs, splitted_ubs = compute_intermediate_bounds(
        intermediate_net, branching_layer_log, splitted_domain, splitted_lbs, splitted_ubs)

    # update and retrieve which relus are active/passing/ambiguous (need to rebuild the model with all the batch)
    intermediate_net.build_model_using_bounds(splitted_domain, (splitted_lbs, splitted_ubs))
    intermediate_net.update_relu_mask()
    updated_mask = intermediate_net.relu_mask

    # get the new last-layer bounds after the splitting
    if not gurobi_dict["gurobi"]:
        # compute all last layer bounds in parallel
        if parent_init_flag:
            bounds_net.initialize_from(parent_init_stacks)
        bounds_net.build_model_using_bounds(splitted_domain, (splitted_lbs, splitted_ubs))
        # here, not computing upper bounds to save memory (and time for anderson-based bounding)
        updated_lbs = bounds_net.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
        splitted_lbs[-1] = torch.max(updated_lbs, splitted_lbs[-1])
        # evaluate the network at the lower bound point
        dom_ub_point = bounds_net.get_lower_bound_network_input()
        dual_solutions = bounds_net.children_init
    else:
        # compute them one by one
        splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions = compute_last_bounds_cpu(
            bounds_net, splitted_domain, splitted_lbs, splitted_ubs, gurobi_dict)

    # retrieve bounds info from the bounds network: the lower bounds are the output of the bound calculation, the upper
    # bounds are computed by evaluating the network at the lower bound points.
    dom_lb_all = splitted_lbs
    dom_ub_all = splitted_ubs
    dom_lb = splitted_lbs[-1]

    # TODO: do we need any alternative upper bounding strategy for the dual algorithms?
    dom_ub = bounds_net.net(dom_ub_point)

    # check that the domain upper bound is larger than its lower bound. If not, infeasible domain (and mask).
    # return +inf as a consequence to have the bound pruned.
    primal_feasibility = bab.check_primal_infeasibility(dom_lb_all, dom_ub_all, dom_lb, dom_ub)
    dom_lb = torch.where(~primal_feasibility, float('inf') * torch.ones_like(dom_lb), dom_lb)
    dom_ub = torch.where(~primal_feasibility, float('inf') * torch.ones_like(dom_ub), dom_ub)

    return dom_ub, dom_lb, dom_ub_point, updated_mask, dom_lb_all, dom_ub_all, dual_solutions


def compute_intermediate_bounds(intermediate_net, branching_layer_log, splitted_domain, intermediate_lbs,
                                intermediate_ubs):
    # compute intermediate bounds for the current batch, leaving out unnecessary computations
    # (those before the splitted relus)

    # get minimum layer idx where branching is happening
    min_branching_layer = len(intermediate_net.weights)
    for branch_lay_idx in range(len(branching_layer_log)):
        if branching_layer_log[branch_lay_idx]:
            min_branching_layer = branch_lay_idx
            break

    # List of sets storing for each layer index, the batch entries that are splitting a ReLU there or onwards
    cumulative_branching_layer_log = [None] * (len(intermediate_net.lower_bounds)-1)
    for branch_lay_idx in range(len(branching_layer_log)):
        cumulative_branching_layer_log[branch_lay_idx] = branching_layer_log[branch_lay_idx]
        if branch_lay_idx > 0:
            cumulative_branching_layer_log[branch_lay_idx] |= cumulative_branching_layer_log[branch_lay_idx-1]

    # TODO: this was +1 in the PLNN-verification-private codebase, but this should be correct (and is more efficient)
    for x_idx in range(min_branching_layer+2, len(intermediate_net.lower_bounds)):

        active_batch_ids = list(cumulative_branching_layer_log[x_idx-2])
        sub_batch_intermediate_lbs = [lbs[active_batch_ids] for lbs in intermediate_lbs]
        sub_batch_intermediate_ubs = [ubs[active_batch_ids] for ubs in intermediate_ubs]

        intermediate_net.build_model_using_bounds(
            splitted_domain[active_batch_ids],
            (sub_batch_intermediate_lbs, sub_batch_intermediate_ubs))
        updated_lbs, updated_ubs = intermediate_net.compute_lower_bound(
            node=(x_idx, None), counterexample_verification=True)

        # retain best bounds and update intermediate bounds from batch
        intermediate_lbs[x_idx][active_batch_ids] = torch.max(updated_lbs, intermediate_lbs[x_idx][active_batch_ids])
        intermediate_ubs[x_idx][active_batch_ids] = torch.min(updated_ubs, intermediate_ubs[x_idx][active_batch_ids])

    return intermediate_lbs, intermediate_ubs


def compute_last_bounds_cpu(bounds_net, splitted_domain, splitted_lbs, splitted_ubs, gurobi_dict):
    # Compute the last layer bounds on (multiple, if p>1) cpu over the batch domains (used for Gurobi).

    # Retrieve execution specs.
    p = gurobi_dict["p"]
    server_queue = gurobi_dict["server_queue"]
    instruction_queue = gurobi_dict["instruction_queue"]
    barrier = gurobi_dict["barrier"]

    if p == 1:
        batch_indices = list(range(splitted_lbs[0].shape[0]))
        cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs = bab.subproblems_to_cpu(
            splitted_domain, splitted_lbs, splitted_ubs)
        splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions = bab.compute_last_bounds_sequentially(
            bounds_net, cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs, batch_indices)
    else:
        # Full synchronization after every batch.
        barrier.wait()

        cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs = bab.subproblems_to_cpu(
            splitted_domain, splitted_lbs, splitted_ubs, share=True)

        max_batch_size = cpu_splitted_lbs[0].shape[0]
        c_batch_size = int(ceil(max_batch_size / float(p)))
        busy_processors = int(ceil(max_batch_size / float(c_batch_size))) - 1
        idle_processors = p - (busy_processors+1)

        # Send bounding jobs to the busy cpu servers.
        for sub_batch_idx in range(busy_processors):
            start_batch_index = sub_batch_idx * c_batch_size
            end_batch_index = min((sub_batch_idx + 1) * c_batch_size, max_batch_size)
            slice_indices = list(range(start_batch_index, end_batch_index))
            instruction_queue.put((cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs, slice_indices))
        # Keep the others idle.
        for _ in range(idle_processors):
            instruction_queue.put(("idle",))

        # Execute the last sub-batch of bounds on this cpu core.
        slice_indices = list(range((busy_processors) * c_batch_size, max_batch_size))
        splitted_lbs, splitted_ubs, c_dom_ub_point, c_dual_solutions = bab.compute_last_bounds_sequentially(
            bounds_net, cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs, slice_indices, share=True)

        # Gather by-products of bounding in the same format returned by a gpu-batched bounds computation.
        dom_ub_point = c_dom_ub_point[0].unsqueeze(0).repeat(((max_batch_size,) + (1,) * (c_dom_ub_point.dim() - 1)))
        dual_solutions = c_dual_solutions.as_stack(max_batch_size)
        dom_ub_point[slice_indices] = c_dom_ub_point
        dual_solutions.set_stack_parent_entries(c_dual_solutions, slice_indices)

        for _ in range(busy_processors):
            # Collect bounding jobs from cpu servers.
            splitted_lbs, splitted_ubs, c_dom_ub_point, c_dual_solutions, slice_indices = \
                server_queue.get(True)

            # Gather by-products of bounding in the same format returned by a gpu-batched bounds computation.
            dom_ub_point[slice_indices] = c_dom_ub_point
            dual_solutions.set_stack_parent_entries(c_dual_solutions, slice_indices)

    return splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions
