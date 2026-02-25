from EPEC_model.EPEC import EPEC
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_scaled_setup(n_players: int, P_max_ref: list, demand_ref: float, cost_ref: list):
    """
    Generates ordered generator parameters for n_players,
    preserving the relative pattern from P_max_ref
    Keeps total system capacity constant for comparable social welfare.
    """

    # --- Reference pattern base case ---
    base_pattern = np.array(P_max_ref)
    base_pattern_sum = base_pattern.sum()
    base_pattern = base_pattern / base_pattern.sum()  # normalize

    # --- Interpolate to match n_players ---
    x_base = np.linspace(0, 1, len(base_pattern))
    x_new = np.linspace(0, 1, n_players)
    pattern_scaled = np.interp(x_new, x_base, base_pattern)

    # Normalize so total capacity is constant (≈165)
    Pmax = pattern_scaled / pattern_scaled.sum() * base_pattern_sum
    Pmin = np.zeros(n_players)

    # --- Costs: follow same increasing pattern ---
    base_cost_min = np.array(cost_ref)

    cost_pattern = np.interp(x_new, x_base, base_cost_min)

    cost = np.round(cost_pattern, 1)

    demand = demand_ref

    return (
        Pmin.tolist(),
        Pmax.round(1).tolist(),
        cost.tolist(),
        demand,
    )


def run_multiple_player_setups(max_players: int, P_max_ref: list, cost_ref: list, demand_ref: float, segments: int):   

    epec_results = {}

    players_list = range(4, max_players + 1)
    convergence_rate = []
    worst_poa_list   = []
    worst_poa_conv_list = []
    highest_cci_list = []
    highest_cci_conv_list = []

    for n_players in players_list:
        print(f"\n--- Running EPEC for {n_players} players ---")
        Pmin, Pmax, cost, demand = generate_scaled_setup(n_players=n_players, P_max_ref=P_max_ref, demand_ref=demand_ref, cost_ref=cost_ref)
        print("Pmin:", Pmin)
        print("Pmax:", Pmax)
        print("Cost:", cost)
        print("Demand:", demand)

        epec = EPEC(Pmin = Pmin, 
                    Pmax = Pmax, 
                    demand = demand, 
                    cost = cost, 
                    segments = segments, 
                    exercise = "4_multi_players"
                    )
        
        share_converged, worst_poa, worst_poa_conv, highest_cci, highest_cci_conv = epec.iterate_cost_combinations()
        convergence_rate.append(share_converged)
        worst_poa_list.append(worst_poa)
        worst_poa_conv_list.append(worst_poa_conv)
        highest_cci_list.append(highest_cci)
        highest_cci_conv_list.append(highest_cci_conv)
        epec_results[n_players] = epec

        # Save merit order curve for each setup of players for run_id = 0 to show the merit order curve change with players
        epec.plot_merit_order_curve(run_id = 0, num_players=n_players)

    # # --- Plot convergence rate vs number of players ---
    plt.figure(figsize=(8, 5))
    plt.plot(players_list, convergence_rate, marker='o')
    plt.xlabel('Number of Players')
    plt.ylabel('Convergence Rate (%)')
    # plt.title('EPEC Convergence Rate vs Number of Players')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('Assignment_scripts/figures/4_multi_players', exist_ok=True)
    plt.savefig('Assignment_scripts/figures/4_multi_players/convergence_rate_vs_players.png', dpi=300)
    plt.show()

    # # --- Plot worst PoA vs number of players ---
    plt.figure(figsize=(8, 5))
    plt.plot(players_list, worst_poa_list, marker='o', color='orange', label='All Runs')
    plt.plot(players_list, worst_poa_conv_list, marker='o', color='red', label='Converged Runs')
    plt.xlabel('Number of Players')
    plt.ylabel('Worst Price of Anarchy (PoA)')
    plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', 
        ncol=4, framealpha=0.9, fontsize=10)
    # plt.title('Worst PoA vs Number of Players')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('Assignment_scripts/figures/4_multi_players', exist_ok=True)
    plt.savefig('Assignment_scripts/figures/4_multi_players/worst_poa_vs_players.png', dpi=300)
    plt.show()

    # # --- Plot highest CCI vs number of players ---
    plt.figure(figsize=(8, 5))
    plt.plot(players_list, highest_cci_list, marker='o', color='green', label='All Runs')
    plt.plot(players_list, highest_cci_conv_list, marker='o', color='darkred', label='Converged Runs')
    plt.xlabel('Number of Players')
    plt.ylabel('Highest Consumer Cost Inflation (CCI)')
    plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', 
        ncol=4, framealpha=0.9, fontsize=10)
    # plt.title('Highest CCI vs Number of Players')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('Assignment_scripts/figures/4_multi_players', exist_ok=True)
    plt.savefig('Assignment_scripts/figures/4_multi_players/highest_cci_vs_players.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    
    # 4 setup
    Pmin = [ 0,  0,  0,  0]
    Pmax = [55, 60, 65, 70]

    demand = 160

    cost = [80, 100, 120, 140]

    segments = 3

    epec = EPEC(Pmin = Pmin, 
                    Pmax = Pmax, 
                    demand = demand, 
                    cost = cost, 
                    segments = segments, 
                    exercise = "4"
                    )
    epec.iterate_cost_combinations()

    for run_id in epec.results:
        epec.plot_merit_order_curve(run_id = run_id)
        epec.plot_clearing_price_over_iterations(run_id = run_id)

    epec.plot_PoA()

    # Run multiple player setups with only two segments
    max_players = 10
    segments = 2
    run_multiple_player_setups(max_players=max_players, P_max_ref=Pmax, cost_ref=cost, demand_ref=demand, segments=segments)