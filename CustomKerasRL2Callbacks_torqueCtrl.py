from rl.callbacks import Callback
import timeit
import numpy as np
import h5py

class randomSpeedProfile:
    """
    this class allows to integrate speed ramps during training,
    utilization of these speed ramps did not seem expedient in some initial tests,
    that is why the changeProbability (probability of a ramp occuring) was set to zero for this setup

    accordingly, this function is only responsible for changing the speed for every episode,
    during the episode the speed is kept constant
    """

    def __init__(self, epsLength, maxSpeed, changeProbability=0, rampDuration=0.3):
        self.epsLength = epsLength
        self.maxSpeed = maxSpeed
        self.changeProbability = changeProbability
        self.rampDuration = rampDuration

        self.now_speed = np.random.uniform(-1, 1)
        self.next_speed = self.now_speed
        self.old_speed = self.now_speed
        self.upcoming_speed = self.now_speed

    def randomProfile(self, t):

        if (t <= 50e-6):
            self.now_speed = self.upcoming_speed
            self.next_speed = self.now_speed
            self.old_speed = self.now_speed
            self.t_ramp_end = 0
            self.upcoming_speed = np.random.uniform(-1, 1)

        if self.old_speed == self.next_speed and np.random.uniform() < self.changeProbability:
            self.old_speed = self.next_speed
            self.next_speed = np.random.uniform(-1, 1)
            self.t_ramp_start = t
            self.t_ramp_end = t + self.rampDuration

        if t < self.t_ramp_end:
            self.now_speed = (t - self.t_ramp_start) * (self.next_speed - self.old_speed) / self.rampDuration + self.old_speed
        else:
            self.old_speed = self.next_speed

        return self.now_speed * self.maxSpeed


class StoreEpisodeLogger(Callback):

    """
    this callback is used for three purposes:
    1) changes to the environment at runtime (e.g. changes of reference point)
    2) changes to the agent at runtime (e.g. changes learning rate)
    3) creating logfiles of the training episodes to monitor learning progress (also for plotting)
    """

    def __init__(self,
                 folder_name,
                 file_name,
                 tau,
                 limits,
                 training,
                 lr_max,
                 lr_min,
                 nb_steps_start,
                 nb_steps_reduction,
                 speed_generator,
                 create_eps_logs = False,
                 test = False,):

        self.folder_name = folder_name
        self.file_name = file_name
        self.tau = tau
        self.limits = limits
        self.training = training
        self.create_eps_logs = create_eps_logs

        self.lr_max = lr_max
        self.lr_min = lr_min
        self.nb_steps_start = nb_steps_start
        self.nb_steps_reduction = nb_steps_reduction

        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}

        self.step = 0
        self.test = test

        # since torque and speed are normalized at this point, an upper boundary of 1 refers to the maximum allowed torque / speed
        self.torque_high = 1
        self.torque_low = - self.torque_high

        self.speed_high = 1
        self.speed_low = - self.speed_high

        self.speed_generator = speed_generator

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        print('done')

    def on_episode_begin(self, episode, logs):

        # randomize reference torque at the start of each episode
        self.env.env.env.reference_generator._reference_value = np.random.uniform(self.torque_low,
                                                                                  self.torque_high)

        # set flag to reset the motor at the next on_step_begin call
        self.resample_state = True

        # initialize the logging buffers
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []

    def on_episode_end(self, episode, logs):

        # compute the mean reward for the finished episode
        mean_rew = np.mean(self.rewards[episode])

        # log the training process in the form of text output
        if self.training:
            if not self.test:
                print(self.folder_name + " Steps learned = {} / {}, episode = {}, mean reward = {}".
                      format(logs['nb_steps'], self.params['nb_steps'], episode, mean_rew))

            # a reward history file will log the mean reward over the course of all training episodes
            if episode == 0:
                history = np.array([])
            else:
                try:
                    with h5py.File(self.folder_name + "/" + "history" + ".hdf5", "r") as f:
                        history = np.copy(f['history'])
                except:
                    history = np.array([])

            with h5py.File(self.folder_name + "/" + "history" + ".hdf5", "w") as f:
                history = np.append(history, mean_rew)
                hist = f.create_dataset("history", data=history)

        # create a logfile at the end of each episode, important for plotting of episodes
        if self.create_eps_logs:
            with h5py.File(self.folder_name + "/" + self.file_name + "_" + str(episode) + ".hdf5", "w") as f:
                tau = f.create_dataset("tau", data=self.tau)
                lim = f.create_dataset("limits", data=self.limits)

                obs = f.create_dataset("observations", data=self.observations[episode])
                rews = f.create_dataset("rewards", data=self.rewards[episode])
                acts = f.create_dataset("actions", data=self.actions[episode])

        # clear the logging buffers
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]

    def on_step_begin(self, step, logs={}):

        # if the episode is for testing / validation, the initial state is the zero vector
        if self.test:
            if self.resample_state:
                i_q_0 = 0
                i_d_0 = 0
                eps_0 = 0
                omega_0 = 0
                self.env.env.env.reference_generator._reference_value = 0
                self.env.env.env.physical_system._ode_solver.set_initial_value(np.array([omega_0, i_d_0, i_q_0, eps_0]))
                self.resample_state = False

            # the validation profile has a defined torque reference profile
            else:
                niveau0 = 0
                niveau1 = 0.375 #0.5
                niveau2 = 0.75 #1

                if (step % 25000) <= 5000:
                    self.env.env.env.reference_generator._reference_value = niveau0
                elif (step % 25000) <= 10000:
                    self.env.env.env.reference_generator._reference_value = niveau1
                elif (step % 25000) <= 15000:
                    self.env.env.env.reference_generator._reference_value = niveau2
                elif (step % 25000) <= 20000:
                    self.env.env.env.reference_generator._reference_value = -niveau2
                elif (step % 25000) <= 25000:
                    self.env.env.env.reference_generator._reference_value = -niveau1

        # if the episode is for training, there is a chance that the reference torque is changed during the episode
        else:
            if np.random.uniform() < 0.001:
                self.env.env.env.reference_generator._reference_value = np.random.uniform(self.torque_low,
                                                                                          self.torque_high)

            # if the environment has to be reset, use exploring starts
            if self.resample_state:
                eps_0 = np.random.uniform(-1, 1) * np.pi
                omega_0 = self.speed_generator.upcoming_speed * self.env.env.env.physical_system.limits[0]

                # the utilized exploring starts strategy uses parameter knowledge to speed up the training, which is helpful but not obligatory
		# the agent itself has no parameter knowledge
                guaranteed_in_ellipsis = True
                if guaranteed_in_ellipsis:
                    psi_p = self.env.env.env.physical_system.electrical_motor.motor_parameter["psi_p"]
                    l_d = self.env.env.env.physical_system.electrical_motor.motor_parameter["l_d"]
                    l_q = self.env.env.env.physical_system.electrical_motor.motor_parameter["l_q"]
                    p = self.env.env.env.physical_system.electrical_motor.motor_parameter["p"]
                    u_dc = self.limits[-1]
                    dc_link_d = u_dc / (np.sqrt(3) * l_d * np.abs(omega_0 * p))
                    i_d_upper = np.clip(- psi_p / l_d + dc_link_d, None, 230)
                    i_d_lower = np.clip(- psi_p / l_d - dc_link_d, -230, None)
                    i_d_0 = np.random.uniform(i_d_lower, i_d_upper)
                    i_q_upper = np.clip(np.sqrt((u_dc / (np.sqrt(3) * omega_0 * p * l_q)) ** 2 -
                                        (l_d / l_q * (i_d_0 + psi_p / l_d)) ** 2), None, np.sqrt(230 ** 2 - i_d_0 ** 2))
                    i_q_lower = np.clip(- i_q_upper, - np.sqrt(230 ** 2 - i_d_0 ** 2), None)
                    i_q_0 = np.random.uniform(i_q_lower, i_q_upper)

                # one can also use exploring starts in a completely random fashion,
                # but some initial states might be uncontrollable, the controller will not gain useful experiences in that case
                # one will probably need more training episodes in this case
                else:
                    i_q_0 = np.random.uniform(-0.7, +0.7) * 240
                    i_d_0 = np.random.uniform(-0.7, +0.7) * 240

                # reset the motor system in the given way
                self.env.env.env.physical_system._ode_solver.set_initial_value(np.array([omega_0, i_d_0, i_q_0, eps_0]))
                self.resample_state = False


            # changes the agent's learning rate if necessary
            if self.step > self.nb_steps_start and self.step < (self.nb_steps_start + self.nb_steps_reduction):
                lr_slope = self.lr_max - self.lr_min
                new_lr = self.lr_max - lr_slope / self.nb_steps_reduction * (self.step - self.nb_steps_start)
                self.model.trainable_model.optimizer._hyper['learning_rate'] = new_lr

    def on_step_end(self, step, logs):

        # save the new state transition to the logging buffer
        episode = logs['episode']
        self.observations[episode].append(self.env.env._obs_logger) # (logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])

        # count the steps
        self.step += 1