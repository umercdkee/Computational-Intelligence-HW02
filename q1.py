import xml.etree.ElementTree as ET
import math
import random

#code to read data from file
def read_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    name    = root.find('info/name').text
    dataset = root.find('info/dataset').text

    nodes = {}
    for node in root.findall('network/nodes/node'):
        node_id = int(node.get('id'))
        nodes[node_id] = {
            'type': int(node.get('type')),
            'cx':   float(node.find('cx').text),
            'cy':   float(node.find('cy').text),
        }

    capacity = float(root.find('fleet/vehicle_profile/capacity').text)
    depot    = int(float(root.find('fleet/vehicle_profile/departure_node').text))

    requests = {}
    for req in root.findall('requests/request'):
        node_id = int(req.get('node'))
        requests[node_id] = float(req.find('quantity').text)

    return {
        'name':     name,
        'dataset':  dataset,
        'nodes':    nodes,
        'capacity': capacity,
        'depot':    depot,
        'requests': requests,
    }


#using coordinates to calculate distance between each node
def build_distance_matrix(nodes):

    distances = {}
    ids = list(nodes.keys())
    for i in ids:
        for j in ids:
            if i == j:
                distances[(i, j)] = 0.0
            else:
                dx = nodes[i]['cx'] - nodes[j]['cx']
                dy = nodes[i]['cy'] - nodes[j]['cy']
                distances[(i, j)] = math.sqrt(dx * dx + dy * dy)
    return distances #dictionary storing distance between node i and node j


#initializing pheronome and visibility matrices for each edge using distances calculated above
def initialise_matrices(node_ids, distances, tau0):
    tau = {}
    eta = {}
    for i in node_ids:
        for j in node_ids:
            if i == j or distances[(i, j)] == 0: #edge case: node i is same as node j
                tau[(i, j)] = 0.0
                eta[(i, j)] = 0.0
            else:
                tau[(i, j)] = tau0 #setting initial tau values to an arbitrary small constant
                eta[(i, j)] = 1.0 / distances[(i, j)] #eta formula
    return tau, eta


#function to update transition probabilities for all nodes
def transition_probabilities(current, allowed, tau, eta, alpha, beta):

    numerators = {}
    for j in allowed: #updating probabilities for all customers that the truck is allowed to visit
        t = tau[(current, j)] ** alpha
        e = eta[(current, j)] ** beta
        numerators[j] = t * e 

    total = sum(numerators.values())

    if total == 0:
        prob = 1.0 / len(allowed)
        return [(j, prob) for j in allowed]

    return [(j, numerators[j] / total) for j in allowed]


#using roulette wheel scheme to construct ants based on transition probabilities
#random number generated and probabilities for each customer summed up until they are greater than or equal to the random number
def roulette_select(probabilities):
    r = random.random()
    cumulative = 0.0
    for city, prob in probabilities:
        cumulative += prob
        if r <= cumulative:
            return city
    return probabilities[-1][0]


#constructing one full ant
def build_solution(depot, customers, demands, capacity,
                   tau, eta, alpha, beta):

    unvisited = set(customers)   #tabu list of customers not yet visited by ant
    routes          = []
    loads           = []

    while unvisited:
        #initializing a new truck route from the depot
        current_node  = depot
        route_load    = 0.0
        route_stops   = []

        while True:
            #customers that still fit in this truck
            allowed = [
                c for c in unvisited
                if route_load + demands[c] <= capacity
            ]

            if not allowed:
                break   #truck is full or no allowed customer left

            #computing transition probabilities and picking next stop
            probs       = transition_probabilities(current_node, allowed,
                                                   tau, eta, alpha, beta)
            next_node   = roulette_select(probs)

            #updating truck route and relevant variables  
            route_stops.append(next_node)
            route_load  += demands[next_node]
            unvisited.discard(next_node)
            current_node = next_node
        
        #updating ant
        routes.append(route_stops)
        loads.append(route_load)

    return routes, loads


#calculating fitness of each ant. fitness function is being minimized here
def solution_distance(routes, depot, distances):
    total = 0.0
    for route in routes:
        if not route: #skip truck without route
            continue
        prev = depot #initial point
        for node in route: #summing all distances
            total += distances[(prev, node)]
            prev = node
        total += distances[(prev, depot)]   #distance to return to depot
    return total


#updating pheromones for all edges
def update_pheromones(tau, all_solutions, node_ids, depot, distances, rho, Q):

    delta_tau = {(i, j): 0.0 for i in node_ids for j in node_ids} #matrix to store pheronome deposits

    for routes, total_dist in all_solutions:
        if total_dist == 0:
            continue
        deposit = Q / total_dist #total deposit of each ant 

        for route in routes: #spreading deposit of each ant over all its routes
            if not route: #skipping trucks without routes
                continue
            path = [depot] + route + [depot] #adding depot to start and end of route
            for k in range(len(path) - 1): 
                i, j = path[k], path[k + 1]
                delta_tau[(i, j)] += deposit
                delta_tau[(j, i)] += deposit   #updating the matrix to store pheronome deposits for each edge
    
    for i in node_ids:
        for j in node_ids:
            #evaporating old pheronome and adding new deposits
            if i != j:
                tau[(i, j)] = rho * tau[(i, j)] + delta_tau[(i, j)]

    return tau

#helper function to print results. doesn't return anything, just prints. thank you Claude.
def format_route(depot, route, demands, distances):
    if not route:
        return "  (empty route)"
    path       = [depot] + route + [depot]
    dist       = sum(distances[(path[k], path[k+1])] for k in range(len(path)-1))
    load       = sum(demands[c] for c in route)
    stops_str  = " → ".join(str(n) for n in path)
    return f"  {stops_str}   [load={load:.0f}, dist={dist:.2f}]"

#another helper function to print things tameez se
def print_solution(routes, depot, demands, distances, label = "Solution"):
    total = solution_distance(routes, depot, distances)
    print(f"\n  {label}  |  Vehicles used: {len(routes)}  |  Total distance: {total:.2f}")
    for idx, route in enumerate(routes):
        print(f"    Vehicle {idx+1}: {format_route(depot, route, demands, distances)}")



def run_aco(data,
            num_ants       = 20,
            alpha          = 1.0,
            beta           = 2.0,
            rho            = 0.5,
            Q              = 100.0,
            num_iterations = 100,
            tau0           = 0.01,
            verbose        = True):

    nodes     = data['nodes']
    depot     = data['depot']
    demands   = data['requests']        
    capacity  = data['capacity']
    customers = [nid for nid in nodes if nid != depot]

    distances         = build_distance_matrix(nodes)
    node_ids          = list(nodes.keys())
    tau, eta          = initialise_matrices(node_ids, distances, tau0)

    best_routes  = None
    best_dist    = math.inf

    if verbose:
        print("=" * 65)
        print("  ANT COLONY OPTIMISATION — Vehicle Routing Problem")
        print("=" * 65)
        print(f"  Instance    : {data['name']}  ({data['dataset']})")
        print(f"  Nodes       : {len(nodes)}  (depot = {depot})")
        print(f"  Customers   : {len(customers)}")
        print(f"  Capacity    : {capacity}")
        print(f"  Ants        : {num_ants}")
        print(f"  α (alpha)   : {alpha}")
        print(f"  β (beta)    : {beta}")
        print(f"  ρ (rho)     : {rho}")
        print(f"  Q           : {Q}")
        print(f"  Iterations  : {num_iterations}")

    #main
    for iteration in range(1, num_iterations + 1):

        all_solutions = []

        #each ant being constructed
        for _ in range(num_ants):
            routes, loads = build_solution(
                depot, customers, demands, capacity,
                tau, eta, alpha, beta
            )
            dist = solution_distance(routes, depot, distances)
            all_solutions.append((routes, dist))

            if dist < best_dist:
                best_dist   = dist
                best_routes = [r[:] for r in routes]   #deep copy of the best routes

        #updating pheromones
        tau = update_pheromones(
            tau, all_solutions, node_ids, depot, distances, rho, Q
        )

        #logging results
        if verbose:
            iter_best_dist   = min(d for _, d in all_solutions)
            iter_best_routes = next(r for r, d in all_solutions
                                    if d == iter_best_dist)
            improved = "  ← NEW BEST" if iter_best_dist < best_dist + 1e-9 \
                                         and iter_best_dist == best_dist else ""
            print(f"\n{'─'*65}")
            print(f"  Iteration {iteration:>3}  |  "
                  f"Iter best: {iter_best_dist:.2f}  |  "
                  f"Global best: {best_dist:.2f}{improved}")

    #final results
    print("\n" + "=" * 65)
    print("  FINAL RESULT")
    print("=" * 65)
    print_solution(best_routes, depot, demands, distances,
                   label="Best solution found")
    print("=" * 65)

    return best_routes, best_dist


def run_grid_search(data, param_grid, num_iterations=100, tau0=0.01, seed=42):
    """
    Runs run_aco() varying one parameter at a time from a baseline.
    The first value in each param_grid list is the baseline value.

    param_grid keys: alpha, beta, rho, num_ants

    Prints a ranked summary table at the end.
    Returns a list of result dicts sorted best -> worst.
    """
    baseline = {
        'alpha':    param_grid['alpha'][0],
        'beta':     param_grid['beta'][0],
        'rho':      param_grid['rho'][0],
        'num_ants': param_grid['num_ants'][0],
    }

    # Build configs: baseline + one-at-a-time variations
    configs = [dict(baseline)]
    for param, values in param_grid.items():
        for val in values[1:]:
            cfg = dict(baseline)
            cfg[param] = val
            if cfg not in configs:
                configs.append(cfg)

    total   = len(configs)
    results = []

    print("=" * 75)
    print("  PARAMETER GRID SEARCH")
    print("=" * 75)
    print(f"  Baseline  ->  alpha={baseline['alpha']}  beta={baseline['beta']}  "
          f"rho={baseline['rho']}  ants={baseline['num_ants']}")
    print(f"  Varying one parameter at a time")
    print(f"  Iterations per run  : {num_iterations}")
    print(f"  Total configurations: {total}")

    for run_idx, cfg in enumerate(configs, start=1):
        alpha    = cfg['alpha']
        beta     = cfg['beta']
        rho      = cfg['rho']
        num_ants = cfg['num_ants']

        changed = [p for p in baseline if cfg[p] != baseline[p]]
        tag = "BASELINE" if not changed else f"vary {', '.join(changed)}"

        print(f"\n{'─'*75}")
        print(f"  Run {run_idx}/{total}  [{tag}]  "
              f"alpha={alpha}  beta={beta}  rho={rho}  ants={num_ants}")
        print(f"{'─'*75}")

        random.seed(seed)

        _, best_dist = run_aco(
            data,
            num_ants       = num_ants,
            alpha          = alpha,
            beta           = beta,
            rho            = rho,
            Q              = 100.0,
            num_iterations = num_iterations,
            tau0           = tau0,
            verbose        = False,
        )
        results.append({
            'run':       run_idx,
            'tag':       tag,
            'alpha':     alpha,
            'beta':      beta,
            'rho':       rho,
            'num_ants':  num_ants,
            'best_dist': best_dist,
        })

        print(f"  Best distance: {best_dist:.2f}")

    # Summary table ranked best -> worst
    results.sort(key=lambda x: x['best_dist'])

    print("\n" + "=" * 75)
    print("  RESULTS SUMMARY  (ranked best -> worst)")
    print("=" * 75)
    print(f"  {'Rank':<5} {'alpha':<7} {'beta':<6} {'rho':<6} {'Ants':<6} "
          f"{'Best Dist':<12}")
    print("  " + "-" * 71)
    for rank, r in enumerate(results, start=1):
        marker = "  <- BEST"  if rank == 1            else ""
        marker = "  <- WORST" if rank == len(results) else marker
        print(f"  {rank:<5} {r['alpha']:<7} {r['beta']:<6} {r['rho']:<6} "
              f"{r['num_ants']:<6} {r['best_dist']:<12.2f}  "
              f"{r['tag']}{marker}")
    print("=" * 75)

    return results

if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "test_instances/A-n32-k05.xml"

    data = read_file(filepath)

    #first value in each list is the baseline.
    #all other values are tested one at a time while the rest stay at baseline.
    param_grid = {
        'alpha':    [2,   1,   3,   5, 2  ],
        'beta':     [2,   1,   3,   5, 2 ],
        'rho':      [0.5, 0.2, 0.7, 0.9, 0.7],
        'num_ants': [10,  20,  40,  80, 40 ],
    }

    run_grid_search(
        data,
        param_grid     = param_grid,
        num_iterations = 100,
        tau0           = 0.01,
        seed           = 42,
    )
