import numpy as np
import matplotlib.pyplot as plt
from mountain_car import MountainCar
from mountain_car_animation import animate
from ch13 import reinforce, reinforce_baseline, actor_critic_1step, actor_critic_et

# import h5py


NUM_RUNS = 1
ACTS = [1, -1, 0]
mc = MountainCar()

rewards_run_reinforce = []
steps_run_reinforce = []

rewards_run_reinforce_bl = []
steps_run_reinforce_bl = []

rewards_run_ac_1step = []
steps_run_ac_1step = []

rewards_run_ac_et = []
steps_run_ac_et = []
#
# for run in range(NUM_RUNS):
#     print('Run Number ' + str(run + 1) + ' of ' + str(NUM_RUNS))
#
#     # print('REINFORCE')
#     # trajectory_reinforce, rewards_reinforce, steps_reinforce = reinforce(alpha=2e-12)
#     # rewards_run_reinforce.append(rewards_reinforce)
#     # steps_run_reinforce.append(steps_reinforce)
#     #
#     print('REINFORCE with Baseline')
#     trajectory_reinforce_bl, rewards_reinforce_bl, steps_reinforce_bl = reinforce_baseline(alpha_theta=4e-6, alpha_w=4e-3)
#     rewards_run_reinforce_bl.append(rewards_reinforce_bl)
#     steps_run_reinforce_bl.append(steps_reinforce_bl)
    # print('One-step Actor Critic')
    # trajectory_ac_1step, rewards_ac_1step, steps_ac_1step = actor_critic_1step(alpha_theta=4e-6, alpha_w=4e-3)
    # rewards_run_ac_1step.append(rewards_ac_1step)
    # steps_run_ac_1step.append(steps_ac_1step)
    #
    # print('Actor Critic with Eligibility Traces')
    # trajectory_ac_et, rewards_ac_et, steps_ac_et = actor_critic_et(lambda_theta=.7, lambda_w=.7,
    #                                                                alpha_theta=4e-6, alpha_w=4e-3)
    # rewards_run_ac_et.append(rewards_ac_et)
    # steps_run_ac_et.append(steps_ac_et)
#
# f = h5py.File('REINFORCE_with_Baseline.hdf5', "w")
# f.create_dataset('rewards', data=rewards_run_reinforce_bl)
# f.create_dataset('steps', data=steps_run_reinforce_bl)
# #
# plt.figure('Rewards comparison')
# # plt.plot(np.mean(np.asarray(rewards_run_reinforce), axis=0), label='REINFORCE')
# plt.plot(np.mean(np.asarray(rewards_run_reinforce_bl), axis=0), label='REINFORCE with Baseline')
# # plt.plot(np.mean(np.asarray(rewards_run_ac_1step), axis=0), label='One-step Actor Critic')
# # plt.plot(np.mean(np.asarray(rewards_run_ac_et), axis=0), label='Actor Critic with Eligibility Traces')
# plt.legend()
# plt.ylabel('Average Rewards After ' + str(NUM_RUNS) + ' Runs')
# plt.xlabel('Episodes')
#
# plt.figure('Steps comparison')
# # plt.plot(np.mean(np.asarray(steps_run_reinforce), axis=0), label='REINFORCE')
# plt.plot(np.mean(np.asarray(steps_run_reinforce_bl), axis=0), label='REINFORCE with Baseline')
# # plt.plot(np.mean(np.asarray(steps_run_ac_1step), axis=0), label='One-step Actor Critic')
# # plt.plot(np.mean(np.asarray(steps_run_ac_et), axis=0), label='Actor Critic with Eligibility Traces')
# plt.legend()
# plt.ylabel('Average Number of Steps After ' + str(NUM_RUNS) + ' Runs')
# plt.xlabel('Episodes')
# plt.yscale('log')
# plt.show()

# Animating the algorithmsr

# traj_log_x = np.asarray(reinforce(alpha=4e-7)[0])
# animate(traj_log_x[:, 0])

# traj_log_x = np.asarray(reinforce_baseline(alpha_theta=4e-6, alpha_w=4e-3)[0])
# animate(traj_log_x[:, 0])

# traj_log_x = np.asarray(actor_critic_1step(alpha_theta=4e-6, alpha_w=4e-3)[0])
# animate(traj_log_x[:, 0])

traj_log_x = np.asarray(actor_critic_et(lambda_theta=.7, lambda_w=.7, alpha_theta=4e-6, alpha_w=4e-3)[0])
animate(traj_log_x[:, 0])
#
