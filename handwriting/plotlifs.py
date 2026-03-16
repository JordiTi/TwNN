import numpy as np
import matplotlib.pyplot as plt

# simulation time
dt = 0.0001
T = 1.0
t = np.arange(0, T, dt)

# biologically realistic synaptic kernel (difference of exponentials)
def syn_kernel(t, tau_r, tau_d):
    k = np.exp(-t/tau_d) - np.exp(-t/tau_r)
    k[t < 0] = 0
    k[k < 0] = 0  # enforce non-negative
    return k   # normalize peak to 1

# generate poisson spike train
def poisson_spikes(rate_hz):
    spikes = (np.random.rand(len(t)) < rate_hz * dt).astype(float)
    return spikes

# filter spikes
def synaptic_filter(spikes, tau_r, tau_d):
    tk = np.arange(0, 1, dt)
    kernel = syn_kernel(tk, tau_r, tau_d)
    y = np.convolve(spikes, kernel)[:len(spikes)]
    return y

# parameter sets (rise, decay)
params = [
    (0.005, 0.008, 7),   # fast synapse
    (0.01, 0.05, 2),   # medium
    (0.02, 0.08, 3)      # slow
]

fig, axes = plt.subplots(4, 1, figsize=(8,6), sharex=True)
allamps = []
allspikes = []
for i, ax in enumerate(axes):
    if i == 3:
            ax.plot(t, np.sum(np.array(allamps), axis=0), linewidth=2)
            ax.set_ylim(bottom=0)
            ax.set_ylim(top=15)
            ax.set_ylabel("Synaptic value")
    else:   
        tau_r, tau_d, amp = params[i]


        spikes = poisson_spikes(rate_hz=np.random.randint(15,30))
        spikes[5000:] = 0
        allspikes.append(spikes)
        syn = synaptic_filter(spikes*amp, tau_r, tau_d)
        allamps.append(syn)
        spike_times = t[spikes > 0]
        ax.plot(t, syn, linewidth=2)
        ax.vlines(spike_times, 0,  4, linewidth=2, color="orange")

        ax.set_ylim(bottom=0)
        ax.set_ylim(top=15)
        ax.set_ylabel("Synaptic value")
        ax.set_title(f"τ_r={tau_r*1000:.1f} ms, τ_d={tau_d*1000:.1f} ms")

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
np.savetxt("sumoftwitches.txt", np.array(allamps) )
np.savetxt("allspikes.txt", np.array(allspikes) )
plt.savefig("sumoftwitches.pdf", dpi=300, format="pdf")
plt.show()