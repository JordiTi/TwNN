from shared.layers import SpikingLayerConfig

BASEFUNCTIONS_LAYER_CONFIG = SpikingLayerConfig(
    threshold=2.0,
    tau_mem=30.0,
    tau_syn=15.0,
    tauref = 50.0,
    tau_rise = 20.0,
    tau_decay = 15.0,
    beta = 1.0,
    chunk_size=1,   
    weight_clip=1.0,
)

