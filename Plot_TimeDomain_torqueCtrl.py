import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_performance_metrics(i_d, i_q, T, T_ref):
    """
    This function will calculate the performance metrics of the measured data i_d, i_q, T on a given torque profile T_ref.
    All input data needs to be normalized!
    """

    gamma = 0
    T_lim = 200
    i_n_max = 240
    i_lim = 270
    torque_boundary = 5 / T_lim
    id_boundary = 15 / i_lim
    dangerzone_boundary = i_n_max / i_lim
    test = True

    squared_error_torque = 0
    absolute_error_torque = 0
    squared_stator_current = 0
    r_sum = 0
    for _i_d, _i_q, _T, _T_ref in zip(i_d, i_q, T, T_ref):

        current_total = np.sqrt(_i_d ** 2 + _i_q ** 2)
        e_T_abs = np.abs(_T_ref - _T)

        squared_error_torque += (e_T_abs / 2) ** 2
        absolute_error_torque += (e_T_abs / 2)
        squared_stator_current += current_total ** 2

        term = False
        if current_total > 1 and not test:
            rew = -1
            term = True
        elif current_total > dangerzone_boundary:
            reward_offset = - (1 - gamma)
            rew = (1 - (current_total - dangerzone_boundary) / (1 - dangerzone_boundary)) * (1 - gamma) / 2 + reward_offset
        elif _i_d > id_boundary:
            reward_offset = - (1 - gamma) / 2
            rew = (1 - (_i_d - id_boundary) / (dangerzone_boundary - id_boundary)) * (1 - gamma) / 2 + reward_offset
        elif e_T_abs > torque_boundary:
            reward_offset = 0
            rew = (1 - e_T_abs / 2) * (1 - gamma) / 2 + reward_offset
        else:
            reward_offset = (1 - gamma) / 2
            rew = (1 - current_total) * (1 - gamma) / 2 + reward_offset

        r_sum += rew

    print(f"Mean Reward: {r_sum / len(T)}")
    print(f"Torque MSE: {squared_error_torque / len(T)}")
    print(f"Torque MAE: {absolute_error_torque / len(T)}")
    print(f"Current RMS: {np.sqrt(squared_stator_current / len(T))}")
    print(f"in episode of length {len(T)}")

    return r_sum / len(T)

def current_mtpc(T_star, motor_parameters):
    """
    This function will calculate optimal maximum torque per current operating points i_d_mtpc, i_q_mtpc for a given
    reference torque T_ref and a given permanent magnet synchronous motor (defined by constant parameters).
    It is not checked whether these operating points are reachable or not.
    """

    delta_l = motor_parameters["l_d"] - motor_parameters["l_q"]
    psi_p = motor_parameters["psi_p"]
    p = motor_parameters["p"]

    a = (3 / 2 * p) ** 2 * delta_l ** 3
    b = (3 / 2 * p) ** 2 * 3 * delta_l ** 2 * psi_p
    c = (3 / 2 * p) ** 2 * 3 * delta_l * psi_p ** 2
    d = (3 / 2 * p) ** 2 * psi_p ** 3
    e = - delta_l * T_star ** 2

    p1 = 2 * c ** 3 - 9 * b * c * d + 27 * a * d ** 2 + 27 * b ** 2 * e - 72 * a * c * e
    p2 = p1 + np.sqrt(- 4 * (c ** 2 - 3 * b * d + 12 * a * e) ** 3 + p1 ** 2)
    p3 = (c ** 2 - 3 * b * d + 12 * a * e) / (3 * a * (p2 / 2) ** (1 / 3)) + (p2 / 2) ** (1 / 3) / (3 * a)
    p4 = np.sqrt(b ** 2 / (4 * a ** 2) - 2 * c / (3 * a) + p3)
    p5 = b ** 2 / (2 * a ** 2) - 4 * c / (3 * a) - p3
    p6 = (- b ** 3 / a ** 3 + 4 * b * c / a ** 2 - 8 * d / a) / (4 * p4)

    i_d_mtpc = - b / (4 * a) - p4 / 2 - (np.sqrt(p5 - p6)) / 2
    i_q_mtpc = T_star / (3 / 2 * p * (psi_p + delta_l * i_d_mtpc))

    return i_d_mtpc, i_q_mtpc


def plot_episode(training_folder, episode_number, episode_type="training_episode"):
    """
    This function creates a PDF plot of the specified episode and saves it to the "Plots" folder.
    A "Plots" folder will be created if necessary.
    """

    folder_name = training_folder
    file_name = episode_type
    nb = episode_number

    # load the specified set of measured data
    path = folder_name + "/" + file_name + "_" + str(nb) + ".hdf5"
    with h5py.File(path, "r") as f:
        tau = np.copy(f['tau'])
        lim = np.copy(f['limits'])

        obs = np.transpose(np.copy(f['observations']))
        rews = np.copy(f['rewards'])
        acts = np.copy(f['actions'])

    try:
        # load the training history, if available (validation / testing episodes have no history)
        with h5py.File(folder_name + "/" + "history" + ".hdf5", "r") as f:
            hist = np.copy(f['history'])
    except:
        hist = [0]

    plt.subplots(2, 2, figsize=(25, 15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=2)
    plt.suptitle(folder_name)

    t = np.arange(0, len(obs[0]), 1)
    t = t * tau * 1e3

    # define the motor parameters
    params = {"p": 3,
              "r_s": 17.932e-3,
              "l_d": 0.37e-3,
              "l_q": 1.2e-3,
              "psi_p": 65.65e-3}
    torque_tolerance = 5 # in Nm

    # calculate the performance in the specified measurement
    calculate_performance_metrics(obs[5], obs[6], obs[1], obs[14])

    # mechanical observations
    obs[0] = lim[0] * obs[0]
    obs[1] = lim[1] * obs[1]

    # current observations
    obs[2] = lim[2] * obs[2]
    obs[3] = lim[2] * obs[3]
    obs[4] = lim[2] * obs[4]
    obs[5] = lim[2] * obs[5]
    obs[6] = lim[2] * obs[6]

    # voltage observations
    obs[7] = lim[7] * obs[7]
    obs[8] = lim[7] * obs[8]
    obs[9] = lim[7] * obs[9]
    obs[10] = lim[7] * obs[10]
    obs[11] = lim[7] * obs[11]

    # electrical motor angle
    obs[12] = obs[12] * np.pi

    # reference torque
    obs[14] = lim[1] * obs[14]

    # calculate optimal MTPC operating points
    i_d_mtpc, i_q_mtpc = current_mtpc(obs[14], motor_parameters=params)

    # plot d-current and optimal d-current
    plt.subplot(12, 3, (1, 7))
    plt.plot(t, obs[5], label=r'$i_\mathrm{d}$')
    plt.plot(t, i_d_mtpc, linestyle='dashed', color="black", label=r'$i_\mathrm{d,MTPC}$')
    plt.title(r'$i_\mathrm{d}$')
    plt.xlabel(r'$t / \mathrm{ms}$')
    plt.ylabel(r'$i / \mathrm{A}$')
    plt.grid(True)
    plt.legend()

    # plot q-current and optimal q-current
    plt.subplot(12, 3, (10, 16))
    plt.plot(t, obs[6], label=r'$i_\mathrm{q}$')
    plt.plot(t, i_q_mtpc, linestyle='dashed', color="black", label=r'$i_\mathrm{q,MTPC}$')
    plt.title(r'$i_\mathrm{q}$')
    plt.xlabel(r'$t / \mathrm{ms}$')
    plt.ylabel(r'$i / \mathrm{A}$')
    plt.grid(True)
    plt.legend()

    # plot stator-fixed, three-phase abc-currents
    plt.subplot(12, 3, (3, 9))
    plt.plot(t, obs[2], label=r'$i_\mathrm{a}$')
    plt.plot(t, obs[3], label=r'$i_\mathrm{b}$')
    plt.plot(t, obs[4], label=r'$i_\mathrm{c}$')
    plt.title(r'$i_\mathrm{rst}$')
    plt.xlabel(r'$t / \mathrm{ms}$')
    plt.ylabel(r'$i / \mathrm{A}$')
    plt.grid(True)
    plt.legend()

    # plot the drive torque, the reference torque and the defined tolerance zone
    plt.subplot(12, 3, (12, 18))
    plt.plot(t, obs[1], label=r'$T$')
    plt.plot(t, obs[14], label=r'$T^*$')
    plt.plot(t, obs[14] + torque_tolerance, linestyle='dashed', color="black")
    plt.plot(t, obs[14] - torque_tolerance, linestyle='dashed', color="black")
    plt.hlines(lim[1], xmin=t[0], xmax=t[-1], label=r'$T_{\mathrm{max}}$')
    plt.hlines(-lim[1], xmin=t[0], xmax=t[-1], label=r'$T_{\mathrm{min}}$')
    plt.title(r'Torque')
    plt.xlabel(r'$t / \mathrm{ms}$')
    plt.ylabel(r'$T / (\mathrm{N} \cdot \mathrm{m})$')
    plt.ylim([- lim[1] - 5, lim[1] + 5])
    plt.grid(True)
    plt.legend()

    # plot the rewards over the course of the episode
    plt.subplot(12, 3, (19, 25))
    plt.plot(t, rews, linewidth=0.5)
    plt.title(r'Reward')
    plt.xlabel(r'$t / \mathrm{ms}$')
    plt.ylabel(r'$r / 1$')
    plt.ylim([-1.1, 1.1])
    plt.grid(True)

    # plot the reward history (the development of average reward over the course of all episodes) ~= "learning curve"
    plt.subplot(12, 3, (28, 34))
    plt.plot(hist, label="episode average reward")
    plt.plot(pd.Series(hist, name='reward_history').rolling(50).mean(), label=r'moving average (50 episodes)')
    plt.title(r'Reward History')
    plt.xlabel(r'Episode')
    plt.ylabel(r'$\frac{1}{K}\sum r$')
    plt.grid(True)
    plt.legend()
    try:
        plt.axvline(x=nb, ymin=-1000, ymax=5, color="red")
        plt.scatter(nb, hist[nb], color="red")
    except:
        pass

    # plot the currents in stator-fixed, two-phase alpha beta coordinates
    i_abc = np.array([obs[2], obs[3], obs[4]])
    T_23 = np.array([[2 / 3,     -1 / 3,          -1 / 3],
                     [0, 1 / np.sqrt(3), -1 / np.sqrt(3)]])
    i_alphabeta = np.matmul(T_23, i_abc)

    plt.subplot(12, 3, (20, 35))
    plt.plot(i_alphabeta[0], i_alphabeta[1])
    circle = plt.Circle((0, 0), lim[2], color='cyan')
    plt.gcf().gca().add_artist(circle)
    plt.title(r'$i_\mathrm{\alpha \beta}$')
    plt.xlabel(r'$i_\mathrm{\alpha} / \mathrm{A}$')
    plt.ylabel(r'$i_\mathrm{\beta} / \mathrm{A}$')
    plt.xlim([-320, 320])
    plt.ylim([-320, 320])
    for i in range(1, 7, 1):
        radius = 270
        angle = 2 * np.pi / 6 * (i - 1)
        switch_order = [0, 4, 6, 2, 3, 1, 5, 7]
        plt.text(radius * np.cos(angle), radius * np.sin(angle), switch_order[i], ha="center", va="center")
    plt.text(0, 0, "0,7", ha="center", va="center")
    plt.grid(True)
    plt.legend()

    # plot the currents in rotor-fixed, two-phase dq coordinates
    plt.subplot(12, 3, (2, 17))
    rect = plt.Rectangle((-280, -280), 560, 560, color="red", alpha=0.5)
    plt.gcf().gca().add_artist(rect)
    circle = plt.Circle((0, 0), lim[2], color='white', fill=True)
    plt.gcf().gca().add_artist(circle)
    plt.plot(obs[5], obs[6], label=r"momentary current")
    circle = plt.Circle((0, 0), lim[2], color='black', fill=False)
    plt.gcf().gca().add_artist(circle)
    circle = plt.Circle((0, 0), 240, color='black', fill=False, linestyle="--")
    plt.gcf().gca().add_artist(circle)
    plt.title(r'$i_\mathrm{dq}$')
    plt.xlim([-280, 280])
    plt.ylim([-280, 280])
    plt.xlabel(r'$i_\mathrm{d} / \mathrm{A}$')
    plt.ylabel(r'$i_\mathrm{q} / \mathrm{A}$')
    plt.grid(True)
    plt.scatter(-params["psi_p"] / params["l_d"], 0, marker="x", color="black", label=r"short cirquit point", zorder=10)

    # annotate optimal currents (in dq)
    count = 0
    _, idx = np.unique(obs[14], return_index=True)
    plt.plot([], [], [], linestyle="-", color="red", linewidth=0.5, label=r"$T^*$")
    for _T in obs[14][np.sort(idx)]:
        i_d_mtpc, i_q_mtpc = current_mtpc(_T, motor_parameters=params)
        plt.text(i_d_mtpc, i_q_mtpc, str(count), zorder=11)
        plt.scatter(i_d_mtpc, i_q_mtpc, color="yellow", zorder=10)
        count += 1
        i_d = np.linspace(-280, 280, 2000)
        denom = 1.5 * params["p"] * (params["psi_p"] + (params["l_d"] - params["l_q"]) * i_d)
        i_q = _T / denom
        i_s = np.sqrt(i_d ** 2 + i_q ** 2)
        gap_idx = np.argmin(i_q)
        mask = i_s < 240
        plt.plot(i_d[0:gap_idx][mask[0:gap_idx]], i_q[0:gap_idx][mask[0:gap_idx]],
                 linestyle="-",
                 color="red",
                 linewidth=0.5)
        plt.plot(i_d[gap_idx:][mask[gap_idx:]], i_q[gap_idx:][mask[gap_idx:]],
                 linestyle="-",
                 color="red",
                 linewidth=0.5)

    # plot MTPC trajectory (in dq)
    i_total = np.linspace(-240, 240, 1000)
    _p = params["psi_p"] / (2 * (params["l_d"] - params["l_q"]))
    _q = - i_total ** 2 / 2
    i_d_opt = - _p / 2 - np.sqrt((_p / 2) ** 2 - _q)
    i_q_opt = np.sqrt(i_total ** 2 - i_d_opt ** 2)
    plt.plot(i_d_opt, i_q_opt, color="black")
    i_q_opt = - np.sqrt(i_total ** 2 - i_d_opt ** 2)
    plt.plot(i_d_opt, i_q_opt, color="black", label=r"MTPC")

    # plot voltage limit ellipsis (in dq)
    i_d = np.linspace(-280, 280, 1000)
    u, c = np.unique(obs[0], return_counts=True)
    dup = u[c > 5]
    w_el = dup * params["p"]
    for _w_el in w_el:
        V_s = lim[7] * 2 / np.sqrt(3)
        i_q_plus = np.sqrt(V_s ** 2 / (_w_el ** 2 * params["l_q"] ** 2) - (params["l_d"] ** 2) / (params["l_q"] ** 2) * (i_d + params["psi_p"] / params["l_d"]) ** 2)
        i_q_minus = - np.sqrt(V_s ** 2 / (_w_el ** 2 * params["l_q"] ** 2) - (params["l_d"] ** 2) / (params["l_q"] ** 2) * (i_d + params["psi_p"] / params["l_d"]) ** 2)
        plt.plot(i_d, i_q_plus, color="cyan")
        plt.plot(i_d, i_q_minus, color="cyan", label=r"available voltage")

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),
              loc='upper right',
              ncol=1,
              #mode="expand",
              borderaxespad=0.0)

    # plot the (mechanical) angular motor speed
    plt.subplot(12, 3, (21, 24))
    plt.plot(t, obs[0])
    plt.title(r'Mechanical Angular Velocity')
    plt.xlabel(r'$t / \mathrm{ms}$')
    plt.ylabel(r'$\omega_\mathrm{me} / \frac{1}{\mathrm{s}}$')
    plt.grid(True)

    # plot the selected actions / switching states over time
    plt.subplot(12, 3, (27, 30))
    plt.title(r"Action")
    plt.scatter(t, acts)
    plt.xlabel(r'$t / \mathrm{ms}$')
    plt.ylabel(r"Switch state")
    plt.grid(True)

    # plot the electrical motor angle
    plt.subplot(12, 3, (33, 36))
    plt.title(r"Electrical Angle")
    plt.plot(t, obs[12])
    plt.xlabel(r'$t / \mathrm{ms}$')
    plt.ylabel(r"$\epsilon_\mathrm{el}$")
    plt.grid(True)

    # save the plot as PDF to the "Plots" directory
    Path('Plots').mkdir(parents=True, exist_ok=True)
    plotName = 'Plots/' + file_name + "_" + str(nb) + '.pdf'
    plt.savefig(plotName, bbox_inches='tight')

    plt.close()

