import matplotlib.pyplot as plt
import torch
import numpy as np

from train import batch_size

def cal_res(model, test_track):
    """
    calculate predicted residual and variances
    of the model, for this test_track.
    test_track's size: [batch_size, n_hits, 3]"""

    print("test track size:", test_track.shape)
    n_events = test_track.shape[0]
    n_batches = int(n_events/batch_size)
    print("number of batches:", n_batches)

    with torch.no_grad():
        output_list = []
        for ibatch in range(n_batches):
            start = ibatch*batch_size
            end = start + batch_size
            test_t = torch.from_numpy(test_track[start:end, :-1, :])
            target_t = torch.from_numpy(test_track[start:end, 1:, 1:])

            output_tmp = model(test_t)
            output_tmp = output_tmp.contiguous().view(-1, output_tmp.size(-1))
            output_tmp[:, 0:2] = output_tmp[:, 0:2] - target_t.contiguous().view(-1, target_t.size(-1))
            output_list.append(output_tmp)

        print("number of output items:", len(output_list))
        output = torch.cat(output_list)
        print(output.size())
        return output


def get_output(model, test_track):
    """
    calculate predicted output for this test_track.
    test_track's size: [batch_size, n_hits, 3]"""

    print("test track size:", test_track.shape)
    n_events = test_track.shape[0]
    n_batches = int(n_events/batch_size)
    print("number of batches:", n_batches)

    with torch.no_grad():
        output_list = []
        for ibatch in range(n_batches):
            start = ibatch*batch_size
            end = start + batch_size
            test_t = torch.from_numpy(test_track[start:end, :-1, :])

            output_tmp = model(test_t)
            output_list.append(output_tmp)

        print("number of output items:", len(output_list))
        output = torch.cat(output_list)
        print(output.size())
        return output


def plot(truth, prediction):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    target = truth[1:, 1]
    predict = prediction[:, 0]
    err = np.sqrt(prediction[:, 2])
    ax.errorbar(np.arange(9), target, fmt='-*', lw=2, ms=10, label='target')
    ax.errorbar(np.arange(9), predict, yerr=err, fmt='.', lw=2, ms=10, label='prediction')
    ax.set_ylim(-3, 3)
    ax.set_ylabel('$\phi$')
    ax.set_xlabel('layer')
    ax.legend()

    ax1 = fig.add_subplot(122)
    target2 = truth[1:, 2]
    predict2 = prediction[:, 1]
    err2 = np.sqrt(prediction[:, 3])
    ax1.errorbar(np.arange(9), target2, fmt='-*', lw=2, ms=10, label='target')
    ax1.errorbar(np.arange(9), predict2, yerr=err2, fmt='.', lw=2, ms=10, label='prediction')
    ax1.set_ylim(-3, 3)
    ax1.set_ylabel('$Z$')
    ax1.set_xlabel('layer')
    ax1.legend()
    return fig


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
