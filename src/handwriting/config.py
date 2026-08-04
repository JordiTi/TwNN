from shared.layers import SpikingLayerConfig

HANDWRITING_LAYERs_CONFIG = SpikingLayerConfig(
    threshold=10.0,
    tau_mem=25.0,
    tau_syn=25.0,
    tauref = 25.0,
    tau_rise = 25.0,
    tau_decay = 25.0,
    beta = 1.0,
    chunk_size=1,   
    weight_clip=1.0,
)
