import matplotlib.pyplot as plt

def draw_bias_variances(out_batches):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(221)
    ax.set_xlim(-np.pi, np.pi)
    nbins = 100

    res = plt.hist(out_batches.numpy()[:, 0, 0],
                   bins=nbins, histtype='step', label="First Layer", lw=2, log=True)
    res = plt.hist(out_batches.numpy()[:, 1, 0],
                   bins=nbins, histtype='step', label="Second Layer", lw=2)
    res = plt.hist(out_batches.numpy()[:, 8, 0],
                   bins=nbins, histtype='step', label="Last Layer", lw=2)
    res = plt.hist(out_batches.numpy()[:, :, 0].flatten(),
                   bins=nbins, histtype='step', label="All predictions", lw=2)
    ax.legend()
    ax.set_xlabel("Error in $\phi$ [rad]")
    ax.set_ylabel('predicted hits')

    # ax3 = fig.add_subplot(223)
    # res = plt.hist(np.sqrt(out_batches.numpy()[:, 0, 2])*np.sign(out_batches.numpy()[:, 0, 0]),
    #                bins=nbins, histtype='step', label="First Layer", lw=2, log=True)
    # res = plt.hist(np.sqrt(out_batches.numpy()[:, 1, 2])*np.sign(out_batches.numpy()[:, 1, 0]),
    #                bins=nbins, histtype='step', label="Second Layer", lw=2)
    # res = plt.hist(np.sqrt(out_batches.numpy()[:, 8, 2])*np.sign(out_batches.numpy()[:, 8, 0]),
    #                bins=nbins, histtype='step', label="Last Layer", lw=2)
    # res = plt.hist(np.sqrt(out_batches.numpy()[:, :, 2].flatten())*np.sign(out_batches.numpy()[:, :, 0].flatten()),
    #                bins=nbins, histtype='step', label="All predictions", lw=2)
    # ax3.legend()
    # ax3.set_xlabel("$\sigma_\phi$ [rad]")
    # ax3.set_ylabel('predicted hits')
    ax3 = fig.add_subplot(223)
    ax3.set_xlim(0, 12.5)
    res = plt.hist(np.sqrt(out_batches.numpy()[:, 0, 2]),
                   bins=nbins, histtype='step', label="First Layer", lw=2, log=True)
    res = plt.hist(np.sqrt(out_batches.numpy()[:, 1, 2]),
                   bins=nbins, histtype='step', label="Second Layer", lw=2)
    res = plt.hist(np.sqrt(out_batches.numpy()[:, 8, 2]),
                   bins=nbins, histtype='step', label="Last Layer", lw=2)
    res = plt.hist(np.sqrt(out_batches.numpy()[:, :, 2].flatten()),
                   bins=nbins, histtype='step', label="All predictions", lw=2)
    ax3.legend()
    ax3.set_xlabel("$\sigma_\phi$ [rad]")
    ax3.set_ylabel('predicted hits')



    ax2 = fig.add_subplot(222)
    ax2.set_xlim(-3000, 3000)
    res = plt.hist(out_batches.numpy()[:, 0, 1],
                   bins=nbins, histtype='step', label="First Layer", lw=2, log=True)
    res = plt.hist(out_batches.numpy()[:, 1, 1],
                   bins=nbins, histtype='step', label="Second Layer", lw=2)
    res = plt.hist(out_batches.numpy()[:, 8, 1],
                   bins=nbins, histtype='step', label="Last Layer", lw=2)
    res = plt.hist(out_batches.numpy()[:, :, 1].flatten(),
                   bins=nbins, histtype='step', label="All predictions", lw=2)
    # res = plt.hist(out_batches.numpy()[:, 8, 0]*sigma_phi+mean_phi, bins=100, histtype='step')
    ax2.legend()
    ax2.set_xlabel("Error in $Z$ [mm]")
    ax2.set_ylabel('predicted hits')


    # ax4 = fig.add_subplot(224)
    # ax4.set_xlim(-5000, 5000)
    # res = plt.hist(np.sqrt(out_batches.numpy()[:, 0, 3])*np.sign(out_batches.numpy()[:, 0, 1]),
    #                bins=nbins, histtype='step', label="First Layer", lw=2, log=True)
    # res = plt.hist(np.sqrt(out_batches.numpy()[:, 1, 3])*np.sign(out_batches.numpy()[:, 1, 1]),
    #                bins=nbins, histtype='step', label="Second Layer", lw=2)
    # res = plt.hist(np.sqrt(out_batches.numpy()[:, 8, 3])*np.sign(out_batches.numpy()[:, 8, 1]),
    #                bins=nbins, histtype='step', label="Last Layer", lw=2)
    # res = plt.hist(np.sqrt(out_batches.numpy()[:, :, 3].flatten())*np.sign(out_batches.numpy()[:, :, 1].flatten()),
    #                bins=nbins, histtype='step', label="All predictions", lw=2)
    # # res = plt.hist(out_batches.numpy()[:, 8, 0]*sigma_phi+mean_phi, bins=100, histtype='step')
    # ax4.legend()
    # ax4.set_xlabel("$\sigma_Z$ [mm]")
    # ax4.set_ylabel('predicted hits')
    ax4 = fig.add_subplot(224)
    ax4.set_xlim(0, 5000)
    res = plt.hist(np.sqrt(out_batches.numpy()[:, 0, 3]),
                   bins=nbins, histtype='step', label="First Layer", lw=2, log=True)
    res = plt.hist(np.sqrt(out_batches.numpy()[:, 1, 3]),
                   bins=nbins, histtype='step', label="Second Layer", lw=2)
    res = plt.hist(np.sqrt(out_batches.numpy()[:, 8, 3]),
                   bins=nbins, histtype='step', label="Last Layer", lw=2)
    res = plt.hist(np.sqrt(out_batches.numpy()[:, :, 3].flatten()),
                   bins=nbins, histtype='step', label="All predictions", lw=2)
    # res = plt.hist(out_batches.numpy()[:, 8, 0]*sigma_phi+mean_phi, bins=100, histtype='step')
    ax4.legend()
    ax4.set_xlabel("$\sigma_Z$ [mm]")
    ax4.set_ylabel('predicted hits')