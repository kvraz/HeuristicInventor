import random
import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        return list(range(n))

    # === NEW STATE VARIABLES ===
    # 1. node_crystallization_potential: Represents how stable or well-integrated a node is
    #    into its current tour position and connections. Higher values indicate higher stability.
    #    Nodes with lower potential are candidates for 'dissolution' (re-evaluation).
    node_crystallization_potential = np.ones(n, dtype=float) * 1.0

    # 2. inter_node_attraction_matrix: A global historical memory matrix that captures the
    #    accumulated desirability for any two nodes (i, j) to be adjacent in the tour.
    #    It's updated based on successful (low-cost) adjacencies, making it non-Markovian.
    inter_node_attraction_matrix = np.full((n, n), 0.1, dtype=float)

    # 3. tour_fluidity: A global non-Markovian parameter that controls the 'aggressiveness'
    #    of the refinement phase. Starts high for exploration and gradually decreases over time
    #    to promote stability in later iterations.
    tour_fluidity = 1.0

    # 4. node_flux_bias: A node-specific historical memory. Nodes that have been
    #    recently moved (part of a dissolution or segment flux) will have a higher bias,
    #    making them less likely to be immediately selected for another move, promoting
    #    exploration of different parts of the tour. This introduces a non-Markovian element
    #    influencing selection diversity.
    node_flux_bias = np.zeros(n, dtype=float)

    # 5. node_resonance_field: A measure of a node's local connectivity
    #    harmony and its global potential for forming good connections with non-neighboring nodes.
    #    Higher values indicate better overall integration and potential. It combines local
    #    and global structural desirability.
    node_resonance_field = np.ones(n, dtype=float) * 1.0

    # 6. segment_dysfunction_map: (NEW state variable) A symmetric N x N matrix
    #    that tracks the accumulated "badness" or "inefficiency" of the path segment
    #    between any two nodes (u,v) in the current tour. Higher values indicate
    #    a more problematic segment (as defined by its endpoints) that needs reconfiguration.
    #    It is a non-Markovian memory of structural weaknesses beyond immediate neighbors,
    #    influencing the selection of segments for the 'Segment Detachment and Re-Fusion' move.
    segment_dysfunction_map = np.full((n, n), 0.1, dtype=float) # Initialize with small base dysfunction

    # === PARAMETERS FOR THE CUSTOM OBJECTIVE FUNCTIONS ===
    alpha_attraction = 2.0      # Weight for inter-node attraction gain
    beta_distance_penalty = 1.0 # Weight for distance increase cost
    gamma_crystallization = 0.5 # Weight for node crystallization potential/internal stability influence
    delta_flux_penalty = 0.2    # Weight for penalizing highly fluxed nodes in selection probabilities
    eta_resonance = 0.5         # Weight for node resonance field influence in scores/selection

    # === PARAMETERS FOR STATE VARIABLE UPDATES ===
    attraction_reward_factor = 0.5  # Factor to increase attraction for newly formed segments
    crystallization_increase_factor = 0.1 # Factor to increase potential for nodes involved in good moves
    crystallization_decay_rate = 0.99 # Multiplicative decay rate for node potential and attraction matrix
    fluidity_decay_rate = 0.995     # Multiplicative decay rate for tour_fluidity
    flux_increase_factor = 0.2      # Factor to increase node_flux_bias upon a move
    flux_decay_rate = 0.9           # Decay rate for node_flux_bias (can be faster than crystallization)
    resonance_decay_rate = 0.985    # Decay rate for node_resonance_field
    discomfort_decay_rate = 0.97    # NEW: Decay rate for segment_dysfunction_map
    discomfort_increase_factor = 0.1 # NEW: Factor to increase dysfunction for problematic segments/gaps
    discomfort_decrease_factor = 0.1 # NEW: Factor to decrease dysfunction for well-formed segments/connections

    epsilon = 1e-6 # Small constant to prevent division by zero in divisions by distance

    # Initial tour construction: Start with a random city
    start_city = random.randint(0, n - 1)
    current_tour = [start_city]
    visited = [False] * n
    visited[start_city] = True

    # --- PHASE 1: Incremental Crystallization (Construction) ---
    # This phase iteratively adds all remaining unvisited cities to form an initial complete tour.
    # The selection of which city to add and where to insert it is guided by the
    # Insertion Crystallization Score (ICS), which incorporates the novel state variables.
    while len(current_tour) < n:
        best_k_candidate = -1
        best_insertion_idx = -1
        max_crystallization_score = -np.inf
        
        unvisited_cities = [i for i in range(n) if not visited[i]]

        if not unvisited_cities:
            break

        for k_candidate in unvisited_cities:
            # Iterate through all possible insertion points in the current tour
            for j in range(len(current_tour)):
                u = current_tour[j]
                v = current_tour[(j + 1) % len(current_tour)] # Handles wrap-around for the tour

                dist_uk = distance_matrix[u, k_candidate]
                dist_kv = distance_matrix[k_candidate, v]
                dist_uv = distance_matrix[u, v]

                # Calculate Insertion Crystallization Score (ICS)
                # This is the custom objective function optimized during construction.
                # It balances historical attraction, immediate distance cost, and node stability.
                attraction_gain = inter_node_attraction_matrix[u, k_candidate] + inter_node_attraction_matrix[k_candidate, v]
                distance_cost = dist_uk + dist_kv - dist_uv
                
                # Potential influence: penalize candidate with low potential, reward neighbors with high potential
                potential_influence = (node_crystallization_potential[u] + node_crystallization_potential[v]) / 2 - node_crystallization_potential[k_candidate]
                
                ics = (alpha_attraction * attraction_gain) - (beta_distance_penalty * distance_cost) + (gamma_crystallization * potential_influence)

                if ics > max_crystallization_score:
                    max_crystallization_score = ics
                    best_k_candidate = k_candidate
                    best_insertion_idx = j + 1

        if best_k_candidate != -1:
            # Perform Segment Insertion: The novel move rule during construction
            
            # Before insertion: old connection (u, v) is broken
            u_old = current_tour[(best_insertion_idx - 1 + len(current_tour)) % len(current_tour)]
            v_old = current_tour[best_insertion_idx % len(current_tour)] # Node at best_insertion_idx

            # NEW: Increase dysfunction for the broken segment (u_old, v_old) as it's modified
            segment_dysfunction_map[u_old, v_old] += discomfort_increase_factor
            segment_dysfunction_map[v_old, u_old] += discomfort_increase_factor

            current_tour.insert(best_insertion_idx, best_k_candidate)
            visited[best_k_candidate] = True

            # Update novel state variables: Non-Markovian mechanism
            prev_node_in_tour = current_tour[(best_insertion_idx - 1 + len(current_tour)) % len(current_tour)]
            next_node_in_tour = current_tour[(best_insertion_idx + 1) % len(current_tour)]
            
            # 1. Reward inter_node_attraction_matrix for successful new connections.
            #    Reward is inversely proportional to distance, prioritizing efficient links.
            inter_node_attraction_matrix[prev_node_in_tour, best_k_candidate] += attraction_reward_factor / (distance_matrix[prev_node_in_tour, best_k_candidate] + epsilon)
            inter_node_attraction_matrix[best_k_candidate, prev_node_in_tour] += attraction_reward_factor / (distance_matrix[prev_node_in_tour, best_k_candidate] + epsilon)
            inter_node_attraction_matrix[best_k_candidate, next_node_in_tour] += attraction_reward_factor / (distance_matrix[best_k_candidate, next_node_in_tour] + epsilon)
            inter_node_attraction_matrix[next_node_in_tour, best_k_candidate] += attraction_reward_factor / (distance_matrix[best_k_candidate, next_node_in_tour] + epsilon)

            # 2. Update node_crystallization_potential. Increase potential for nodes involved in good insertions.
            local_distance_increase = (distance_matrix[prev_node_in_tour, best_k_candidate] + distance_matrix[best_k_candidate, next_node_in_tour]
                                       - distance_matrix[prev_node_in_tour, next_node_in_tour])
            
            node_crystallization_potential[best_k_candidate] += crystallization_increase_factor / (1 + max(0, local_distance_increase))
            node_crystallization_potential[prev_node_in_tour] += crystallization_increase_factor / (1 + max(0, local_distance_increase))
            node_crystallization_potential[next_node_in_tour] += crystallization_increase_factor / (1 + max(0, local_distance_increase))
            
            # 4. Update node_flux_bias for the newly inserted node and its neighbors.
            node_flux_bias[best_k_candidate] += flux_increase_factor
            node_flux_bias[prev_node_in_tour] += flux_increase_factor / 2 # Less for neighbors
            node_flux_bias[next_node_in_tour] += flux_increase_factor / 2

            # NEW: Decrease dysfunction for the newly formed good connections
            segment_dysfunction_map[prev_node_in_tour, best_k_candidate] = max(0.01, segment_dysfunction_map[prev_node_in_tour, best_k_candidate] - discomfort_decrease_factor)
            segment_dysfunction_map[best_k_candidate, prev_node_in_tour] = max(0.01, segment_dysfunction_map[best_k_candidate, prev_node_in_tour] - discomfort_decrease_factor)
            segment_dysfunction_map[best_k_candidate, next_node_in_tour] = max(0.01, segment_dysfunction_map[best_k_candidate, next_node_in_tour] - discomfort_decrease_factor)
            segment_dysfunction_map[next_node_in_tour, best_k_candidate] = max(0.01, segment_dysfunction_map[next_node_in_tour, best_k_candidate] - discomfort_decrease_factor)

        # 3. Global decay of state variables after each city insertion. This is a non-Markovian 'forgetting' mechanism.
        node_crystallization_potential *= crystallization_decay_rate
        inter_node_attraction_matrix *= crystallization_decay_rate
        tour_fluidity *= fluidity_decay_rate
        node_flux_bias *= flux_decay_rate
        node_resonance_field *= resonance_decay_rate
        segment_dysfunction_map *= discomfort_decay_rate # NEW: Decay segment dysfunction
        
        # Ensure state variables remain within reasonable bounds (non-negative, not excessively large)
        node_crystallization_potential = np.maximum(node_crystallization_potential, 0.01)
        inter_node_attraction_matrix = np.maximum(inter_node_attraction_matrix, 0.01)
        node_flux_bias = np.maximum(node_flux_bias, 0.0)
        node_resonance_field = np.maximum(node_resonance_field, 0.01)
        segment_dysfunction_map = np.maximum(segment_dysfunction_map, 0.01)

    # --- PHASE 2: Adaptive Tour Reorganization (Refinement) ---
    # This phase performs iterative refinements using three novel move types:
    # 1. Segment Detachment & Re-Fusion (now with dysfunction-based selection)
    # 2. Dyad Re-Embedding (operates on directed pairs of nodes)
    # 3. Single Node Dissolution & Re-Crystallization (fine-tuning)
    # The choice between these mechanisms is dynamic, influenced by tour_fluidity, making it adaptive.

    num_refinement_iterations = n * 20 # Increased iterations for N=20 to allow more refinement steps

    for iteration in range(num_refinement_iterations):
        # Global decay of state variables at the start of each refinement iteration
        node_crystallization_potential *= crystallization_decay_rate
        inter_node_attraction_matrix *= crystallization_decay_rate
        tour_fluidity *= fluidity_decay_rate
        node_flux_bias *= flux_decay_rate
        node_resonance_field *= resonance_decay_rate
        segment_dysfunction_map *= discomfort_decay_rate # NEW: Decay segment dysfunction
        
        # Ensure state variables remain within reasonable bounds
        tour_fluidity = max(0.01, min(1.0, tour_fluidity))
        node_crystallization_potential = np.maximum(node_crystallization_potential, 0.01)
        inter_node_attraction_matrix = np.maximum(inter_node_attraction_matrix, 0.01)
        node_flux_bias = np.maximum(node_flux_bias, 0.0)
        node_resonance_field = np.maximum(node_resonance_field, 0.01)
        segment_dysfunction_map = np.maximum(segment_dysfunction_map, 0.01)

        # Update node_resonance_field based on current tour and other state variables
        current_tour_nodes_arr = np.array(current_tour)
        for i_idx in range(n):
            city_i = current_tour_nodes_arr[i_idx]
            prev_city = current_tour_nodes_arr[(i_idx - 1 + n) % n]
            next_city = current_tour_nodes_arr[(i_idx + 1) % n]

            # Local harmony: current connections based on attraction and distance
            local_harmony = (inter_node_attraction_matrix[city_i, prev_city] + inter_node_attraction_matrix[city_i, next_city]) \
                            - (distance_matrix[city_i, prev_city] + distance_matrix[city_i, next_city])

            # Global potential: attraction to other nodes not in immediate vicinity, scaled by distance
            global_potential = 0.0
            for other_city in range(n):
                if other_city != city_i and other_city != prev_city and other_city != next_city:
                    global_potential += inter_node_attraction_matrix[city_i, other_city] / (1.0 + distance_matrix[city_i, other_city])

            node_resonance_field[city_i] = local_harmony + (tour_fluidity * global_potential * 0.1) # tour_fluidity scales exploration for global potential

        # Adaptive choice between Segment Detachment, Dyad Re-Embedding, and Single Node Re-Crystallization
        segment_move_raw_weight = tour_fluidity
        dyad_move_raw_weight = (1.0 - abs(tour_fluidity - 0.5) * 2)
        single_node_move_raw_weight = (1.0 - tour_fluidity)

        segment_move_raw_weight = max(0.05, segment_move_raw_weight * 1.5)
        dyad_move_raw_weight = max(0.05, dyad_move_raw_weight * 1.5)
        single_node_move_raw_weight = max(0.05, single_node_move_raw_weight * 1.5)
        
        total_weight = segment_move_raw_weight + dyad_move_raw_weight + single_node_move_raw_weight
        prob_segment = segment_move_raw_weight / total_weight
        prob_dyad = dyad_move_raw_weight / total_weight
        prob_single_node = single_node_move_raw_weight / total_weight
        
        move_choice = random.random()

        if move_choice < prob_segment and len(current_tour) > 3:
            # --- Perform Segment Detachment and Re-Fusion ---
            # NEW: Segment selection is now biased by segment_dysfunction_map,
            # targeting segments that have accumulated high 'badness' scores.
            
            # 1. Select a segment defined by endpoints (u, v) based on high segment_dysfunction_map[u,v]
            segment_candidate_scores = []
            segment_candidates = []
            
            # Iterate through all possible segments in the current tour (defined by a start node and length)
            # Minimum segment length 2 (path of 3 nodes, e.g., A-B-C, segment B-C).
            # Max segment length up to N-2 (to leave at least 2 nodes in the rest of the tour for re-insertion)
            for i in range(n):
                u_idx_in_tour = i
                u = current_tour[u_idx_in_tour]
                
                # Segment length L: 2 to N-2 (leaving at least 2 nodes outside the segment)
                for L in range(2, n - 1): # L is the number of nodes in the segment
                    v_idx_in_tour = (u_idx_in_tour + L - 1) % n
                    v = current_tour[v_idx_in_tour]

                    # Score for this potential segment: high dysfunction + low endpoint stability = good candidate
                    segment_score = segment_dysfunction_map[u, v] \
                                    * (1.0 - node_crystallization_potential[u]) \
                                    * (1.0 - node_crystallization_potential[v]) \
                                    / (1.0 + delta_flux_penalty * node_flux_bias[u] + delta_flux_penalty * node_flux_bias[v] + epsilon)
                    
                    segment_candidate_scores.append(max(epsilon, segment_score))
                    segment_candidates.append((u_idx_in_tour, v_idx_in_tour, L))
            
            if not segment_candidates or np.sum(segment_candidate_scores) < epsilon:
                continue

            segment_probs = np.array(segment_candidate_scores) / np.sum(segment_candidate_scores)
            chosen_segment_tuple = random.choices(segment_candidates, weights=segment_probs, k=1)[0]
            
            u_idx_in_tour, v_idx_in_tour, L = chosen_segment_tuple
            
            # Extract the actual segment nodes (u -> ... -> v)
            segment_nodes_original = []
            for k in range(L):
                segment_nodes_original.append(current_tour[(u_idx_in_tour + k) % n])

            segment_nodes_reversed = segment_nodes_original[::-1]

            # Store nodes surrounding the segment *before* detachment
            prev_node_of_segment_idx = (u_idx_in_tour - 1 + n) % n
            next_node_of_segment_idx = (v_idx_in_tour + 1) % n
            prev_node_of_segment = current_tour[prev_node_of_segment_idx]
            next_node_of_segment = current_tour[next_node_of_segment_idx]

            # 3. Detach the segment: Create a temporary tour without it
            temp_tour = []
            for i in range(n):
                if current_tour[i] not in segment_nodes_original:
                    temp_tour.append(current_tour[i])
            
            # Update state variables due to detachment
            for node in segment_nodes_original:
                node_crystallization_potential[node] = 0.01 # Reset potential as it's 'dissolved'
                node_flux_bias[node] += flux_increase_factor # Mark segment nodes as recently moved
            
            # NEW: Increase dysfunction for the newly formed gap, as a segment was removed.
            segment_dysfunction_map[prev_node_of_segment, next_node_of_segment] += discomfort_increase_factor
            segment_dysfunction_map[next_node_of_segment, prev_node_of_segment] += discomfort_increase_factor

            # Reward the direct connection formed by closing the gap, if it's an improvement
            if distance_matrix[prev_node_of_segment, next_node_of_segment] < (distance_matrix[prev_node_of_segment, segment_nodes_original[0]] + distance_matrix[segment_nodes_original[-1], next_node_of_segment]):
                inter_node_attraction_matrix[prev_node_of_segment, next_node_of_segment] += attraction_reward_factor / (distance_matrix[prev_node_of_segment, next_node_of_segment] + epsilon)
                inter_node_attraction_matrix[next_node_of_segment, prev_node_of_segment] += attraction_reward_factor / (distance_matrix[prev_node_of_segment, next_node_of_segment] + epsilon)
            
            # 4. Re-Fusion Evaluation (Segment Re-Integration Score - SRS)
            max_srs = -np.inf
            best_insertion_idx = -1
            best_segment_orientation = None

            for s_orientation in [segment_nodes_original, segment_nodes_reversed]:
                first_node_in_s = s_orientation[0]
                last_node_in_s = s_orientation[-1]

                for j in range(len(temp_tour)): # Iterate over all possible insertion points
                    u_in_temp = temp_tour[j]
                    v_in_temp = temp_tour[(j + 1) % len(temp_tour)]

                    dist_s_u = distance_matrix[first_node_in_s, u_in_temp]
                    dist_s_v = distance_matrix[last_node_in_s, v_in_temp]
                    dist_uv = distance_matrix[u_in_temp, v_in_temp]

                    # Internal segment cost (sum of current edge distances within segment)
                    internal_segment_distance = sum(distance_matrix[s_orientation[k], s_orientation[k+1]] for k in range(L-1)) if L > 1 else 0

                    # External connection attraction from historical memory
                    external_attraction_gain = inter_node_attraction_matrix[u_in_temp, first_node_in_s] + inter_node_attraction_matrix[last_node_in_s, v_in_temp]

                    # Total distance change for this insertion
                    total_new_distance_cost = dist_s_u + internal_segment_distance + dist_s_v - dist_uv

                    # Internal stability of segment based on historical attraction for its internal edges
                    internal_segment_stability_bonus = sum(inter_node_attraction_matrix[s_orientation[k], s_orientation[k+1]] for k in range(L-1)) if L > 1 else 0

                    # Add resonance influence for segment re-integration
                    resonance_influence_srs = (node_resonance_field[u_in_temp] + node_resonance_field[v_in_temp]) / 2 + (node_resonance_field[first_node_in_s] + node_resonance_field[last_node_in_s]) / 2

                    srs = (alpha_attraction * external_attraction_gain) - (beta_distance_penalty * total_new_distance_cost) + (gamma_crystallization * internal_segment_stability_bonus) + (eta_resonance * resonance_influence_srs * 0.5)

                    if srs > max_srs:
                        max_srs = srs
                        best_insertion_idx = j + 1
                        best_segment_orientation = s_orientation
            
            # 5. Re-Fuse the segment at the best found position and orientation
            if best_segment_orientation is not None and best_insertion_idx != -1:
                # Construct the new tour list
                new_tour_list = temp_tour[:best_insertion_idx] + best_segment_orientation + temp_tour[best_insertion_idx:]
                current_tour = new_tour_list
                
                # Update state variables after successful re-fusion
                new_prev_node = current_tour[(best_insertion_idx - 1 + n) % n]
                new_next_node = current_tour[(best_insertion_idx + L) % n] # Node after the entire segment

                # Reward external connections of the re-fused segment
                inter_node_attraction_matrix[new_prev_node, best_segment_orientation[0]] += attraction_reward_factor / (distance_matrix[new_prev_node, best_segment_orientation[0]] + epsilon)
                inter_node_attraction_matrix[best_segment_orientation[0], new_prev_node] += attraction_reward_factor / (distance_matrix[new_prev_node, best_segment_orientation[0]] + epsilon)
                inter_node_attraction_matrix[best_segment_orientation[-1], new_next_node] += attraction_reward_factor / (distance_matrix[best_segment_orientation[-1], new_next_node] + epsilon)
                inter_node_attraction_matrix[new_next_node, best_segment_orientation[-1]] += attraction_reward_factor / (distance_matrix[best_segment_orientation[-1], new_next_node] + epsilon)

                # Reward internal segment connections if it was a good re-fusion
                for k in range(L-1):
                    node1 = best_segment_orientation[k]
                    node2 = best_segment_orientation[k+1]
                    inter_node_attraction_matrix[node1, node2] += attraction_reward_factor / (distance_matrix[node1, node2] + epsilon)
                    inter_node_attraction_matrix[node2, node1] += attraction_reward_factor / (distance_matrix[node1, node2] + epsilon)

                # Update crystallization potential and flux bias for all involved nodes
                local_distance_change_effect = (distance_matrix[new_prev_node, best_segment_orientation[0]] + distance_matrix[best_segment_orientation[-1], new_next_node]
                                            - distance_matrix[prev_node_of_segment, next_node_of_segment]) # Simplified measure
                
                # Increase potential for segment nodes if re-integration was good
                for node in best_segment_orientation:
                    node_crystallization_potential[node] += crystallization_increase_factor / (1 + max(0, local_distance_change_effect)) * 2 
                    node_flux_bias[node] += flux_increase_factor # Mark segment nodes as moved

                node_crystallization_potential[new_prev_node] += crystallization_increase_factor / (1 + max(0, local_distance_change_effect))
                node_crystallization_potential[new_next_node] += crystallization_increase_factor / (1 + max(0, local_distance_change_effect))
                node_flux_bias[new_prev_node] += flux_increase_factor / 2
                node_flux_bias[new_next_node] += flux_increase_factor / 2

                # NEW: Decrease dysfunction for the successful re-fused segment boundaries and its internal path
                segment_dysfunction_map[new_prev_node, best_segment_orientation[0]] = max(0.01, segment_dysfunction_map[new_prev_node, best_segment_orientation[0]] - discomfort_decrease_factor)
                segment_dysfunction_map[best_segment_orientation[0], new_prev_node] = max(0.01, segment_dysfunction_map[best_segment_orientation[0], new_prev_node] - discomfort_decrease_factor)
                segment_dysfunction_map[best_segment_orientation[-1], new_next_node] = max(0.01, segment_dysfunction_map[best_segment_orientation[-1], new_next_node] - discomfort_decrease_factor)
                segment_dysfunction_map[new_next_node, best_segment_orientation[-1]] = max(0.01, segment_dysfunction_map[new_next_node, best_segment_orientation[-1]] - discomfort_decrease_factor)
                
                # For the entire segment's path (u to v)
                segment_dysfunction_map[best_segment_orientation[0], best_segment_orientation[-1]] = max(0.01, segment_dysfunction_map[best_segment_orientation[0], best_segment_orientation[-1]] - discomfort_decrease_factor)
                segment_dysfunction_map[best_segment_orientation[-1], best_segment_orientation[0]] = max(0.01, segment_dysfunction_map[best_segment_orientation[-1], best_segment_orientation[0]] - discomfort_decrease_factor)

        elif move_choice < prob_segment + prob_dyad and len(current_tour) > 2:
            # --- Perform Dyad Re-Embedding (NEW MOVE RULE) ---
            # This move selects a directed pair of adjacent nodes (u,v) (a 'dyad'), detaches them,
            # and re-inserts them (possibly reversed) into a new optimal position in the remaining tour.
            
            # 1. Identify a "Disrupted Dyad" (u,v) for re-embedding.
            # Probabilistic selection favoring nodes with low resonance, low crystallization, and not recently fluxed.
            dyad_selection_scores = (1.0 - node_resonance_field[current_tour_nodes_arr]) * (1.0 - node_crystallization_potential[current_tour_nodes_arr]) / (1.0 + delta_flux_penalty * node_flux_bias[current_tour_nodes_arr] + epsilon)
            dyad_selection_scores = np.maximum(dyad_selection_scores, epsilon)
            dyad_selection_probs = dyad_selection_scores / np.sum(dyad_selection_scores)
            
            u_idx_in_tour = random.choices(range(n), weights=dyad_selection_probs, k=1)[0]
            u = current_tour[u_idx_in_tour]
            v = current_tour[(u_idx_in_tour + 1) % n] # The successor of u is v, forming the dyad (u,v)

            if u == v: # Should not happen in a valid tour of > 1 node
                continue

            # 2. Detach the Dyad (u,v) from the tour.
            prev_node_of_u_idx = (u_idx_in_tour - 1 + n) % n
            next_node_of_v_idx = (u_idx_in_tour + 2) % n # Node after v
            prev_node_of_u = current_tour[prev_node_of_u_idx]
            next_node_of_v = current_tour[next_node_of_v_idx]

            temp_tour_dyad_removed = []
            for i in range(n):
                if current_tour[i] != u and current_tour[i] != v:
                    temp_tour_dyad_removed.append(current_tour[i])
            
            if len(temp_tour_dyad_removed) < 1: # Must have at least one node to insert the dyad next to
                continue

            # Update state variables due to detachment
            node_crystallization_potential[u] = 0.01
            node_crystallization_potential[v] = 0.01
            node_flux_bias[u] += flux_increase_factor
            node_flux_bias[v] += flux_increase_factor
            node_flux_bias[prev_node_of_u] += flux_increase_factor / 2
            node_flux_bias[next_node_of_v] += flux_increase_factor / 2

            # NEW: Increase dysfunction for the newly formed gap due to dyad removal
            segment_dysfunction_map[prev_node_of_u, next_node_of_v] += discomfort_increase_factor
            segment_dysfunction_map[next_node_of_v, prev_node_of_u] += discomfort_increase_factor

            # Reward closing gap
            if distance_matrix[prev_node_of_u, next_node_of_v] < (distance_matrix[prev_node_of_u, u] + distance_matrix[v, next_node_of_v]):
                inter_node_attraction_matrix[prev_node_of_u, next_node_of_v] += attraction_reward_factor / (distance_matrix[prev_node_of_u, next_node_of_v] + epsilon)
                inter_node_attraction_matrix[next_node_of_v, prev_node_of_u] += attraction_reward_factor / (distance_matrix[prev_node_of_u, next_node_of_v] + epsilon)
            
            # 3. Find Best Re-Embedding Position (Dyad Re-Embedding Score - DRES)
            max_dres = -np.inf
            best_re_embedding_idx = -1
            best_dyad_orientation = None # (u,v) or (v,u)

            dyad_orientations = [[u,v], [v,u]]
            for dyad_orientation in dyad_orientations:
                if not dyad_orientation or len(dyad_orientation) < 2: # Check if dyad_orientation is valid
                    continue
                first_node_in_dyad = dyad_orientation[0]
                second_node_in_dyad = dyad_orientation[1]
                internal_dyad_distance = distance_matrix[first_node_in_dyad, second_node_in_dyad]
                
                for j in range(len(temp_tour_dyad_removed)):
                    x_in_temp = temp_tour_dyad_removed[j]
                    y_in_temp = temp_tour_dyad_removed[(j + 1) % len(temp_tour_dyad_removed)]

                    dist_x_first = distance_matrix[x_in_temp, first_node_in_dyad]
                    dist_second_y = distance_matrix[second_node_in_dyad, y_in_temp]
                    dist_xy = distance_matrix[x_in_temp, y_in_temp]

                    # Custom objective: Dyad Re-Embedding Score (DRES)
                    external_attraction_gain = inter_node_attraction_matrix[x_in_temp, first_node_in_dyad] + inter_node_attraction_matrix[second_node_in_dyad, y_in_temp]
                    
                    total_new_distance_cost = dist_x_first + internal_dyad_distance + dist_second_y - dist_xy

                    # Resonance influence: Prefer connecting to high resonance nodes (x,y)
                    resonance_influence = (node_resonance_field[x_in_temp] + node_resonance_field[y_in_temp]) / 2

                    # Crystallization influence for the dyad nodes themselves (u,v)
                    dyad_crystallization_influence = (node_crystallization_potential[first_node_in_dyad] + node_crystallization_potential[second_node_in_dyad]) / 2

                    dres = (alpha_attraction * external_attraction_gain) - (beta_distance_penalty * total_new_distance_cost) + (eta_resonance * resonance_influence) + (gamma_crystallization * dyad_crystallization_influence)

                    if dres > max_dres:
                        max_dres = dres
                        best_re_embedding_idx = j + 1
                        best_dyad_orientation = dyad_orientation
            
            # 4. Re-Embed the dyad
            if best_dyad_orientation is not None and best_re_embedding_idx != -1:
                new_tour_list_dyad = temp_tour_dyad_removed[:best_re_embedding_idx] + best_dyad_orientation + temp_tour_dyad_removed[best_re_embedding_idx:]
                current_tour = new_tour_list_dyad

                new_prev_node_dyad = current_tour[(best_re_embedding_idx - 1 + n) % n]
                new_next_node_dyad = current_tour[(best_re_embedding_idx + 2) % n]

                inter_node_attraction_matrix[new_prev_node_dyad, best_dyad_orientation[0]] += attraction_reward_factor / (distance_matrix[new_prev_node_dyad, best_dyad_orientation[0]] + epsilon)
                inter_node_attraction_matrix[best_dyad_orientation[0], new_prev_node_dyad] += attraction_reward_factor / (distance_matrix[new_prev_node_dyad, best_dyad_orientation[0]] + epsilon)
                inter_node_attraction_matrix[best_dyad_orientation[1], new_next_node_dyad] += attraction_reward_factor / (distance_matrix[best_dyad_orientation[1], new_next_node_dyad] + epsilon)
                inter_node_attraction_matrix[new_next_node_dyad, best_dyad_orientation[1]] += attraction_reward_factor / (distance_matrix[best_dyad_orientation[1], new_next_node_dyad] + epsilon)

                inter_node_attraction_matrix[best_dyad_orientation[0], best_dyad_orientation[1]] += attraction_reward_factor / (distance_matrix[best_dyad_orientation[0], best_dyad_orientation[1]] + epsilon)
                inter_node_attraction_matrix[best_dyad_orientation[1], best_dyad_orientation[0]] += attraction_reward_factor / (distance_matrix[best_dyad_orientation[0], best_dyad_orientation[1]] + epsilon)

                local_distance_change_effect_dyad = (distance_matrix[new_prev_node_dyad, best_dyad_orientation[0]] + distance_matrix[best_dyad_orientation[1], new_next_node_dyad]
                                                    - distance_matrix[new_prev_node_dyad, new_next_node_dyad])
                
                node_crystallization_potential[best_dyad_orientation[0]] += crystallization_increase_factor / (1 + max(0, local_distance_change_effect_dyad)) * 2
                node_crystallization_potential[best_dyad_orientation[1]] += crystallization_increase_factor / (1 + max(0, local_distance_change_effect_dyad)) * 2
                node_crystallization_potential[new_prev_node_dyad] += crystallization_increase_factor / (1 + max(0, local_distance_change_effect_dyad))
                node_crystallization_potential[new_next_node_dyad] += crystallization_increase_factor / (1 + max(0, local_distance_change_effect_dyad))
                
                node_flux_bias[best_dyad_orientation[0]] += flux_increase_factor
                node_flux_bias[best_dyad_orientation[1]] += flux_increase_factor
                node_flux_bias[new_prev_node_dyad] += flux_increase_factor / 2
                node_flux_bias[new_next_node_dyad] += flux_increase_factor / 2

                # NEW: Decrease dysfunction for the re-embedded dyad and its new connections
                segment_dysfunction_map[new_prev_node_dyad, best_dyad_orientation[0]] = max(0.01, segment_dysfunction_map[new_prev_node_dyad, best_dyad_orientation[0]] - discomfort_decrease_factor)
                segment_dysfunction_map[best_dyad_orientation[0], new_prev_node_dyad] = max(0.01, segment_dysfunction_map[best_dyad_orientation[0], new_prev_node_dyad] - discomfort_decrease_factor)
                segment_dysfunction_map[best_dyad_orientation[1], new_next_node_dyad] = max(0.01, segment_dysfunction_map[best_dyad_orientation[1], new_next_node_dyad] - discomfort_decrease_factor)
                segment_dysfunction_map[new_next_node_dyad, best_dyad_orientation[1]] = max(0.01, segment_dysfunction_map[new_next_node_dyad, best_dyad_orientation[1]] - discomfort_decrease_factor)
                segment_dysfunction_map[best_dyad_orientation[0], best_dyad_orientation[1]] = max(0.01, segment_dysfunction_map[best_dyad_orientation[0], best_dyad_orientation[1]] - discomfort_decrease_factor)
                segment_dysfunction_map[best_dyad_orientation[1], best_dyad_orientation[0]] = max(0.01, segment_dysfunction_map[best_dyad_orientation[1], best_dyad_orientation[0]] - discomfort_decrease_factor)

        else:
            # --- Perform Single Node Dissolution and Re-Crystallization (Fallback/Fine-tuning) ---
            # This is the existing node-level move, used when fluidity is lower or larger moves are not chosen.
            if len(current_tour) < 3: # Need at least 3 nodes to remove one and re-insert
                continue

            # 1. Identify a 'Weakly Crystallized Node' (WCN) for potential dissolution.
            # Selection bias for WCN: probabilistic, inversely related to potential and resonance, directly related to flux bias.
            wcn_scores = (1.0 - node_crystallization_potential[current_tour_nodes_arr]) * (1.0 - node_resonance_field[current_tour_nodes_arr]) / (1.0 + delta_flux_penalty * node_flux_bias[current_tour_nodes_arr] + epsilon)
            wcn_scores = np.maximum(wcn_scores, epsilon)
            wcn_probs = wcn_scores / np.sum(wcn_scores)
            
            wcn_idx_in_tour = random.choices(range(n), weights=wcn_probs, k=1)[0]
            wcn = current_tour[wcn_idx_in_tour]

            # Decision to dissolve: probabilistic, influenced by `tour_fluidity` (non-Markovian)
            # and the WCN's `node_crystallization_potential` and `node_flux_bias`.
            dissolution_score = tour_fluidity * (1 - node_crystallization_potential[wcn]) / (1 + delta_flux_penalty * node_flux_bias[wcn] + epsilon)
            dissolution_probability = max(0.01, min(0.9, dissolution_score)) # Clamp probability to reasonable range

            if random.random() < dissolution_probability:
                
                # 2. Dissolve the WCN from its current position in the tour.
                prev_node_wcn = current_tour[(wcn_idx_in_tour - 1 + len(current_tour)) % len(current_tour)]
                next_node_wcn = current_tour[(wcn_idx_in_tour + 1) % len(current_tour)]
                
                current_tour.pop(wcn_idx_in_tour)
                
                # Update state variables due to dissolution
                node_crystallization_potential[wcn] = 0.01 # Reset WCN's potential to a base level
                node_flux_bias[wcn] += flux_increase_factor # Mark WCN as recently moved
                node_flux_bias[prev_node_wcn] += flux_increase_factor / 2
                node_flux_bias[next_node_wcn] += flux_increase_factor / 2
                
                # NEW: Increase dysfunction for the gap created by WCN dissolution
                segment_dysfunction_map[prev_node_wcn, next_node_wcn] += discomfort_increase_factor
                segment_dysfunction_map[next_node_wcn, prev_node_wcn] += discomfort_increase_factor

                # Penalize the neighbors if the local structure was poor (e.g., if closing the gap actually reduced distance)
                if distance_matrix[prev_node_wcn, next_node_wcn] < (distance_matrix[prev_node_wcn, wcn] + distance_matrix[wcn, next_node_wcn]):
                    node_crystallization_potential[prev_node_wcn] *= (1 - crystallization_increase_factor)
                    node_crystallization_potential[next_node_wcn] *= (1 - crystallization_increase_factor)
                    inter_node_attraction_matrix[prev_node_wcn, next_node_wcn] += attraction_reward_factor / (distance_matrix[prev_node_wcn, next_node_wcn] + epsilon)
                    inter_node_attraction_matrix[next_node_wcn, prev_node_wcn] += attraction_reward_factor / (distance_matrix[prev_node_wcn, next_node_wcn] + epsilon)

                # 3. Re-Crystallize (Re-insert) the WCN into the remaining tour.
                max_recrystallization_score = -np.inf
                best_recrystallization_idx = -1
                
                for j in range(len(current_tour)): # Iterate over all possible insertion points in the smaller tour
                    u = current_tour[j]
                    v = current_tour[(j + 1) % len(current_tour)]

                    dist_wcn_u = distance_matrix[wcn, u]
                    dist_wcn_v = distance_matrix[wcn, v]
                    dist_uv = distance_matrix[u, v]

                    # Calculate RICS (same formula as ICS, but for re-insertion of an existing node)
                    recrystallization_attraction_gain = inter_node_attraction_matrix[u, wcn] + inter_node_attraction_matrix[wcn, v]
                    recrystallization_distance_cost = dist_wcn_u + dist_wcn_v - dist_uv
                    
                    recrystallization_potential_influence = (node_crystallization_potential[u] + node_crystallization_potential[v]) / 2 - node_crystallization_potential[wcn]
                    
                    # Add resonance influence for single node re-crystallization
                    recrystallization_resonance_influence = (node_resonance_field[u] + node_resonance_field[v]) / 2 - node_resonance_field[wcn]

                    rics = (alpha_attraction * recrystallization_attraction_gain) - (beta_distance_penalty * recrystallization_distance_cost) \
                         + (gamma_crystallization * recrystallization_potential_influence) + (eta_resonance * recrystallization_resonance_influence)

                    if rics > max_recrystallization_score:
                        max_recrystallization_score = rics
                        best_recrystallization_idx = j + 1
                
                # If a valid insertion point was found
                if best_recrystallization_idx != -1:
                    current_tour.insert(best_recrystallization_idx, wcn)

                    # Update state variables after re-crystallization
                    new_prev_node = current_tour[(best_recrystallization_idx - 1 + len(current_tour)) % len(current_tour)]
                    new_next_node = current_tour[(best_recrystallization_idx + 1) % len(current_tour)]
                    
                    inter_node_attraction_matrix[new_prev_node, wcn] += attraction_reward_factor / (distance_matrix[new_prev_node, wcn] + epsilon)
                    inter_node_attraction_matrix[wcn, new_prev_node] += attraction_reward_factor / (distance_matrix[new_prev_node, wcn] + epsilon)
                    inter_node_attraction_matrix[wcn, new_next_node] += attraction_reward_factor / (distance_matrix[wcn, new_next_node] + epsilon)
                    inter_node_attraction_matrix[new_next_node, wcn] += attraction_reward_factor / (distance_matrix[wcn, new_next_node] + epsilon)

                    local_distance_increase = (distance_matrix[new_prev_node, wcn] + distance_matrix[wcn, new_next_node]
                                            - distance_matrix[new_prev_node, new_next_node])
                    
                    # Update crystallization potential, giving a higher reward for a successful re-insertion
                    node_crystallization_potential[wcn] += crystallization_increase_factor / (1 + max(0, local_distance_increase)) * 2 
                    node_crystallization_potential[new_prev_node] += crystallization_increase_factor / (1 + max(0, local_distance_increase))
                    node_crystallization_potential[new_next_node] += crystallization_increase_factor / (1 + max(0, local_distance_increase))
                    
                    # Update node_flux_bias
                    node_flux_bias[wcn] += flux_increase_factor
                    node_flux_bias[new_prev_node] += flux_increase_factor / 2
                    node_flux_bias[new_next_node] += flux_increase_factor / 2

                    # NEW: Decrease dysfunction for the successful re-crystallized node and its new connections
                    segment_dysfunction_map[new_prev_node, wcn] = max(0.01, segment_dysfunction_map[new_prev_node, wcn] - discomfort_decrease_factor)
                    segment_dysfunction_map[wcn, new_prev_node] = max(0.01, segment_dysfunction_map[wcn, new_prev_node] - discomfort_decrease_factor)
                    segment_dysfunction_map[wcn, new_next_node] = max(0.01, segment_dysfunction_map[wcn, new_next_node] - discomfort_decrease_factor)
                    segment_dysfunction_map[next_node_wcn, wcn] = max(0.01, segment_dysfunction_map[next_node_wcn, wcn] - discomfort_decrease_factor)
                    
    return current_tour