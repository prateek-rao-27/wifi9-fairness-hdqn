from .sim import network_data_rate
from .utils import path_loss_2g, path_loss_5g, path_loss_6g
import jax
from .extension_functions import tx_power_indices_to_tx_power 
from .constants import MIN_TX_POWER, MAX_TX_POWER, CHANNEL_WIDTH_2G, CHANNEL_WIDTH_5G, CHANNEL_WIDTH_6G, DEFAULT_SIGMA, DEFAULT_TX_POWER
import jax.numpy as jnp
from functools import partial 
@partial(jax.jit, static_argnames=("return_per_sta","n_tx_power_levels"))
def network_data_rate_mlo(
        key: jax.random.PRNGKey,
        pos: jax.Array,
        walls: jax.Array,
        link_ap_sta, 
        mcs: jax.Array | None = None,
        sigma: float = DEFAULT_SIGMA,
        # path_loss_fn: Callable = default_path_loss
        # key: jax.random.PRNGKey, 
        # pos, 
        # walls, 
        n_tx_power_levels = 4,
        return_per_sta: bool = False, 
    ):

    """
    Compute the aggregate Multi-Link Operation (MLO) network data rate across 2.4 GHz, 5 GHz, and 6 GHz links.
    This function evaluates per-band data rates using a shared network model (`network_data_rate`) and
    returns their sum. For each band, transmission power indices are converted to actual transmit powers,
    a separate PRNG subkey is used, and the corresponding channel width and path-loss model are applied.
    Args:
        key (jax.random.PRNGKey):
            Random key used for stochastic components in the rate computation. It is split internally
            into three subkeys (2G/5G/6G).
        pos (jax.Array):
            Node positions array of shape `(n_nodes, ...)`, consumed by `network_data_rate`.
        walls (jax.Array):
            Wall attenuation matrix of shape `(n_nodes, n_nodes)`. If `None`, a zero matrix is used.
        link_ap_sta:
            Per-band link configuration container with three entries (2G, 5G, 6G). Each entry must include:
            - `"tx_matrix"`: transmission matrix for the band
            - `"tx_power_indices"`: discrete transmit-power indices mapped to actual power values
        mcs (jax.Array | None, optional):
            Optional MCS configuration passed through to the underlying rate model.
        sigma (float, optional):
            Noise/uncertainty parameter passed to `network_data_rate`. Defaults to `DEFAULT_SIGMA`.
        n_tx_power_levels (int, optional):
            Number of discrete transmit-power levels used when converting power indices.
            Defaults to `4`.
    Returns:
        jax.Array:
            Total network data rate, computed as:
            `data_rate_2g + data_rate_5g + data_rate_6g`.
    Notes:
        - Uses per-band constants (`CHANNEL_WIDTH_2G`, `CHANNEL_WIDTH_5G`, `CHANNEL_WIDTH_6G`) and
            path-loss functions (`path_loss_2g`, `path_loss_5g`, `path_loss_6g`).
        - Internally JIT-compiles a partially applied `network_data_rate` function for repeated calls.
    """

    n_nodes = pos.shape[0]

    tx_matrices = link_ap_sta[0]
    tx_power_indices = link_ap_sta[1]    


    if (walls == None):
        walls = jnp.zeros((n_nodes, n_nodes), dtype=float)

    net_data_rate_mlo_1 = partial(network_data_rate, pos=pos, walls=walls, mcs=mcs, sigma=sigma)
    
    key_2g, key_5g, key_6g = jax.random.split(key, 3)    

    tx_power_array_2g = tx_power_indices_to_tx_power(tx_power_indices[0], n_tx_power_levels=n_tx_power_levels, min_tx_power=MIN_TX_POWER, max_tx_power=MAX_TX_POWER)
    tx_power_array_5g = tx_power_indices_to_tx_power(tx_power_indices[1], n_tx_power_levels=n_tx_power_levels, min_tx_power=MIN_TX_POWER, max_tx_power=MAX_TX_POWER)
    tx_power_array_6g = tx_power_indices_to_tx_power(tx_power_indices[2], n_tx_power_levels=n_tx_power_levels, min_tx_power=MIN_TX_POWER, max_tx_power=MAX_TX_POWER)
    
    # tx_power_array_2g = jnp.ones(shape=(n_nodes, ), dtype=jnp.float32) * DEFAULT_TX_POWER
    # tx_power_array_5g = jnp.ones(shape=(n_nodes, ), dtype=jnp.float32) * DEFAULT_TX_POWER
    # tx_power_array_6g = jnp.ones(shape=(n_nodes, ), dtype=jnp.float32) * DEFAULT_TX_POWER

    data_rate_2g = net_data_rate_mlo_1(key_2g, tx=tx_matrices[0], tx_power=tx_power_array_2g, channel_width=CHANNEL_WIDTH_2G, path_loss_fn=path_loss_2g, return_internals = return_per_sta)
    data_rate_5g = net_data_rate_mlo_1(key_5g, tx=tx_matrices[1], tx_power=tx_power_array_5g, channel_width=CHANNEL_WIDTH_5G, path_loss_fn=path_loss_5g, return_internals = return_per_sta)
    data_rate_6g = net_data_rate_mlo_1(key_6g, tx=tx_matrices[2], tx_power=tx_power_array_6g, channel_width=CHANNEL_WIDTH_6G, path_loss_fn=path_loss_6g, return_internals = return_per_sta)
    
    if return_per_sta:
        total_rate = (
        data_rate_2g[0]
        + data_rate_5g[0]
        + data_rate_6g[0]
        )
        return(
            total_rate,(data_rate_2g[1].average_data_rate, data_rate_5g[1].average_data_rate, data_rate_6g[1].average_data_rate,), tx_matrices
        )

        per_tx_rate = (
           data_rate_2g[1].average_data_rate
           + data_rate_5g[1].average_data_rate
           + data_rate_6g[1].average_data_rate
        )
        if return_per_sta:
          print("TX matrix shape:", tx_matrices.shape)
          print("TX matrix:")
          print(tx_matrices)

        return total_rate, per_tx_rate

    return data_rate_2g + data_rate_5g + data_rate_6g