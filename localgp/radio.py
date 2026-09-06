"""An explicit exploratory radio model, not an exact paper reproduction."""
import numpy as np
from scipy.spatial.distance import cdist


def channel_gain(positions, users, config):
    horizontal = cdist(np.asarray(positions), np.asarray(users))
    distance = np.sqrt(horizontal**2 + config.altitude**2)
    elevation = np.degrees(np.arcsin(config.altitude / distance))
    p_los = 1 / (1 + config.los_c * np.exp(-config.los_d * (elevation - config.los_c)))
    loss = config.reference_loss_db + 10 * config.path_loss_exponent * np.log10(distance)
    loss += (1 - p_los) * config.nlos_extra_db
    return 10**(-loss / 10)


def service_metrics(positions, users, config):
    gains = channel_gain(positions, users, config)
    # At most one serving AeBS: strongest eligible uplink identifies association.
    association = gains.argmax(axis=0)
    user_ids = np.arange(len(users))
    served = gains[association, user_ids] * config.mu_power >= config.association_threshold
    loads = np.bincount(association[served], minlength=len(positions))
    downlink = config.aebs_power * gains
    useful = downlink[association, user_ids]
    interference = np.maximum(0, downlink.sum(axis=0) - useful)
    sinr = useful / (config.noise_power + interference)
    rates = np.zeros(len(users))
    rates[served] = config.bandwidth / loads[association[served]] * np.log2(1 + sinr[served])
    return dict(coverage=float(served.mean()), throughput_bps=float(rates.sum()),
                mean_served_sinr=float(sinr[served].mean()) if served.any() else 0.0,
                served_users=int(served.sum()), loads=loads.tolist(),
                association=np.where(served, association, -1).tolist())
