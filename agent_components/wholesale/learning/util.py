from agent_components.wholesale.util import get_sum_purchased_for_ts, calculate_missing_energy


def calculate_balancing_needed(purchases, realized_usage):
    # appending final balancing costs for broker for any missing energy
    if len(purchases) == 0:
        balancing_needed = -1 * realized_usage
    else:
        energy_sum = get_sum_purchased_for_ts(purchases)
        balancing_needed = calculate_missing_energy(energy_sum, realized_usage)
    return balancing_needed