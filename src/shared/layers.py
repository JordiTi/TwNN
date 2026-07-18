import numpy as np
from dataclasses import dataclass


@dataclass
class SpikingLayerConfig:
    threshold: float = 10.0
    urest: float = 0.0
    tau_pre_1: float = 25.0
    tau_pre_2: float = 25.0
    tau_syn: float = 25.0
    tau_mem: float = 25.0
    tau_elig_rise: float = 25.0
    tau_elig_decay: float = 25.0
    beta: float = 1.0
    weight_clip: float = 1.0
    chunk_size: float = 1


class SpikingLayer:
    """
    LIF spiking layer trained with SuperSpike surrogate gradient + eligibility trace.
    Weight updates use a manual three-factor rule (eligibility trace x error),
    not standard backprop.

    Shapes:
        weights: (ninputs, noutputs)
        Vs, Isyns, surrogategradient: (noutputs,)
        pre_trace1, pre_trace2: (ninputs,)
        lambda_1, lambda_2: (ninputs, noutputs)
    """

    def __init__(self, ninputs, noutputs, config=None, debug=False):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.debug = debug
        self.history = {"V": [], "spikes": []} if debug else None

        cfg = config or SpikingLayerConfig()
        self.cfg = cfg
        self.threshold = cfg.threshold
        self.urest = cfg.urest
        self.beta = cfg.beta
        self.weight_clip = cfg.weight_clip
        self.finalchunktime = cfg.chunk_size

        self.weights = np.random.uniform(-1, 1, [ninputs, noutputs])

        self.Vs = np.ones(noutputs) * self.urest
        self.Isyns = np.ones(noutputs) * self.urest

        self.pre_trace1 = np.zeros(ninputs, dtype=np.float32)
        self.pre_trace2 = np.zeros(ninputs, dtype=np.float32)
        self.lambda_1 = np.zeros([ninputs, noutputs])
        self.lambda_2 = np.zeros([ninputs, noutputs])
        self.surrogategradient = 0

        self.chunktime = 0
        self.dw_running = np.zeros([ninputs, noutputs])

        self._precompute_decay_constants(cfg)

    def _precompute_decay_constants(self, cfg):
        self.exp_pre_1 = np.exp(-1 / cfg.tau_pre_1)
        self.exp_pre_2 = np.exp(-1 / cfg.tau_pre_2)
        self.exp_syn = np.exp(-1 / cfg.tau_syn)
        self.exp_mem = np.exp(-1 / cfg.tau_mem)
        self.exp_elig_rise = np.exp(-1 / cfg.tau_elig_rise)
        self.exp_elig_decay = np.exp(-1 / cfg.tau_elig_decay)

        self.one_minus_exp_pre_2 = 1 - self.exp_pre_2
        self.one_minus_exp_mem = 1 - self.exp_mem
        self.one_minus_exp_elig_rise = 1 - self.exp_elig_rise
        self.one_minus_exp_elig_decay = 1 - self.exp_elig_decay

    # ---- forward / state dynamics ----

    def update_state(self, S, dt=1):
        """S: (ninputs,) input spikes. dt: elapsed time this step (chunk counter only).
        Returns spikes: (noutputs,) bool."""
        self._update_traces(S)
        self._update_membrane(S)
        self._update_surrogate_gradient()
        self._update_eligibility_trace()
        return self._check_spikes()

    def _update_traces(self, S):
        self.pre_trace1 = self.pre_trace1 * self.exp_pre_1 + S
        self.pre_trace2 = (
            self.pre_trace2 * self.exp_pre_2
            + self.one_minus_exp_pre_2 * self.pre_trace1
        )

    def _update_membrane(self, S):
        self.Isyns = self.Isyns * self.exp_syn + np.dot(self.weights.T, S)
        self.Vs = self.Vs * self.exp_mem + self.Isyns * self.one_minus_exp_mem

    def _update_surrogate_gradient(self):
        self.surrogategradient = 1 / (
            (1 + np.abs(self.beta * (self.Vs - self.threshold))) ** 2
        )

    def _update_eligibility_trace(self):
        outer_hebbian = np.outer(self.pre_trace2, self.surrogategradient)
        outer_hebbian[outer_hebbian < 1e-7] = 0  # numerical floor
        self.lambda_1 = (
            self.lambda_1 * self.exp_elig_rise
            + self.one_minus_exp_elig_rise * outer_hebbian
        )
        self.lambda_2 = (
            self.lambda_2 * self.exp_elig_decay
            + self.one_minus_exp_elig_decay * self.lambda_1
        )

    def _check_spikes(self):
        spikes = self.Vs > self.threshold
        self.Vs[spikes] = 0
        if self.debug:
            self.history["V"].append(self.Vs.copy())
            self.history["spikes"].append(spikes.copy())
        return spikes

    # ---- weight update ----

    def _accumulate_and_apply(self, dw, lr, dt):
        """Shared chunked-update logic: accumulate dw, apply every `chunk_size` of elapsed time."""
        self.chunktime += dt
        self.dw_running += dw
        if self.chunktime >= self.finalchunktime:
            self.chunktime = 0
            self.weights += lr * self.dw_running
            self.weights = np.clip(self.weights, -self.weight_clip, self.weight_clip)
            self.dw_running = np.zeros([self.ninputs, self.noutputs])

    def update_weight(self, error, lr, dt=1):
        raise NotImplementedError  # subclasses define error routing

    def reset(self):
        self.Vs = np.ones(self.noutputs) * self.urest
        self.surrogategradient = 0
        self.lambda_1 = np.zeros([self.ninputs, self.noutputs])
        self.lambda_2 = np.zeros([self.ninputs, self.noutputs])
        self.pre_trace1 = np.zeros(self.ninputs, dtype=np.float32)
        self.pre_trace2 = np.zeros(self.ninputs, dtype=np.float32)
        if self.debug:
            self.history = {"V": [], "spikes": []}


class HiddenLayer(SpikingLayer):
    def __init__(self, ninputs, noutputs, nalphamneurons, config=None, **kwargs):
        super().__init__(ninputs, noutputs, config=config, **kwargs)
        # Fixed random DFA feedback matrix -- never trained
        self.B = np.random.uniform(-1, 1, [nalphamneurons, noutputs])

    def update_weight(self, error, lr, dt=1):
        """error: (nalphamneurons,) output-layer error, projected via fixed B."""
        delta = np.dot(error, self.B)
        dw = np.multiply(self.lambda_2.T, delta)
        self._accumulate_and_apply(dw, lr, dt)


class OutputLayer(SpikingLayer):
    def update_weight(self, error, lr, dt=1):
        """error: (noutputs,) direct output error, no projection needed."""
        dw = np.multiply(self.lambda_2.T, error)
        self._accumulate_and_apply(dw, lr, dt)
