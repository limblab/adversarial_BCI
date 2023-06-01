import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import matplotlib
import pickle
from wiener_filter import format_data, test_wiener_filter

rcParams['font.family'] = 'Arial'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

def emg_preprocessing(trial_emg, EMG_names):
    """
    Function to clip and normalize each channel of EMG envelopes
    :param trial_emg: list of lists containing EMG envelopes for each successful trial
    :return: list of lists containing the pre-processed EMG envelopes for each successful trial
    """
    
    # First, keep EMG channels that are good across all recorded sessions
    EMG_names_good = ['EMG_ECRb', 'EMG_ECRl', 'EMG_ECU', 'EMG_EDCr', 'EMG_FCR', 'EMG_FCU', 'EMG_FDP'] 
    idx_emg = [EMG_names.index(x) for x in EMG_names_good]
    for trial, val_trial in enumerate(trial_emg):
        trial_emg[trial] = val_trial[:, idx_emg]
    
    trial_emg_np = np.concatenate(trial_emg)

    # EMG clipping
    outlier = np.mean(trial_emg_np, axis=0) + 6 * np.std(trial_emg_np, axis=0)
    for i, val in enumerate(trial_emg):
        for ii in range(len(outlier)):
            trial_emg[i][:, ii] = trial_emg[i][:, ii].clip(max=outlier[ii])

    # EMG normalization
    trial_emg_np_baseline = np.percentile(trial_emg_np, 2, axis=0)
    trial_emg_np_max = np.percentile(trial_emg_np - trial_emg_np_baseline, 90, axis=0)
    for i, val in enumerate(trial_emg):
        trial_emg[i] = (val - trial_emg_np_baseline) / trial_emg_np_max

    return trial_emg, EMG_names_good

def spike_preprocessing(unit_names1, unit_names2, spike1, spike2):
    """
    unit_names1: unit names in the first dataset
    unit_names2: unit names in the second dataset
    spike1: a list, the spike counts for each trial in the first dataset
    spike2: a list, the spike counts for each trial in the second dataset
    """
    all_unit_names = np.sort(list(set(unit_names1)|set(unit_names2)))
    N_unit = len(all_unit_names)
    
    idx = [list(all_unit_names).index(e) for e in unit_names1]
    spike1_ = [np.zeros((s.shape[0], N_unit)) for s in spike1]
    for k in range(len(spike1)):
        spike1_[k][:, idx] = spike1[k]
        
    idx = [list(all_unit_names).index(e) for e in unit_names2]
    spike2_ = [np.zeros((s.shape[0], N_unit)) for s in spike2]
    for k in range(len(spike2)):
        spike2_[k][:, idx] = spike2[k]
    return spike1_, spike2_
    
def plot_train_logs(path, file_name, epochs, log_scale = False):
    with open(path + file_name, 'rb') as fp:
        train_log = pickle.load(fp)
    
    idx = np.append(np.where(np.diff(train_log['epoch']) == 1)[0], len(train_log['epoch'])-1)
    N = len(train_log['decoder r2 wiener'])
    acc_idx = [idx[i*10-1] for i in range(1, N+1)]

    x_tick_label_numbers = np.arange(50, epochs+50, 50)
    epoch_idx = [np.where(np.array(train_log['epoch']) == each-1)[0][-1] for each in x_tick_label_numbers]

    plt.figure('train_log', figsize = (8, 5))
    plt.xlabel('Epochs', fontsize = 24)
    plt.xticks(epoch_idx, labels = x_tick_label_numbers, fontsize = 20)

    ax1 = plt.subplot(111)
    plt.plot(train_log['loss G1'], color = 'r')
    plt.plot(train_log['loss G2'], color = 'deeppink')

    plt.plot(train_log['loss D1'], color = 'blue')
    plt.plot(train_log['loss D2'], color = 'deepskyblue')

    # plt.plot(train_log['loss cycle 121'], color = 'green')
    # plt.plot(train_log['loss cycle 212'], color = 'lawngreen')
    ax1.set_ylabel('Losses', fontsize = 24)
    plt.yticks(fontsize = 20)

    ax2 = ax1.twinx()
    ax2.plot(acc_idx, train_log['decoder r2 wiener'], 'darkgray', linewidth = 4)
    ax2.set_ylim([-0.5, 1])
    ax2.set_ylabel('Decoding accuracy ($R^2$)', fontsize = 24)
    plt.yticks(fontsize = 20)
    if log_scale == True:
        plt.xscale('log')
    plt.tight_layout()
    plt.savefig('./images/training log.pdf')

def plot_actual_and_pred_EMG(fig_title, spike, EMG, decoder, bin_size, n_lags, num_trials, r2_list, EMG_names, color):
    """
    This function is to plot the actual and predicted EMGs in a trial-based manner
    """
    N = len(EMG_names)
    idx = random.sample(range(len(spike)), num_trials)
    p_spike, p_EMG = [spike[i] for i in idx], [EMG[i] for i in idx] # Sampling a certain number of trials for plotting
    actual_EMG_, pred_EMG_ = [], []
    for each in zip(p_spike, p_EMG):
        x, y = format_data(each[0], each[1], n_lags)
        pred_y = test_wiener_filter(x, decoder)
        actual_EMG_.append(y)
        pred_EMG_.append(pred_y)
    trial_len = [len(each) for each in actual_EMG_]
    T = np.cumsum(trial_len)
    # ---- plot ---- #
    fig = plt.figure(fig_title, figsize = (5, 6))
    plt.title(fig_title, fontsize = 14)
    for i in range(N):
        ax = plt.subplot(N, 1, i + 1)
        plt.plot(np.concatenate(actual_EMG_)[:, i], 'k', linewidth = 1.5)
        plt.plot(np.concatenate(pred_EMG_)[:, i], color, linewidth = 1.5)
        for each in T[:-1]:
            plt.axvline(each, linestyle = '--', color = 'gray')
        ylim = plt.gca().get_ylim()[1]
        plt.axis('off')
        plt.text(20, 0.9*ylim, EMG_names[i][4:], fontsize = 12)
        plt.text(150, 0.9*ylim, str(r2_list[i])[:4], fontsize = 12, color = color)
    plt.tight_layout()
    
def list_to_nparray(X):
    """
    This function converts a list of np.array (each array is a single-trial) into a single np.array where trials are concatenated
    """
    n_col = np.size(X[0],1)
    Y = np.empty((0, n_col))
    for each in X:
        Y = np.vstack((Y, each))
    return Y




