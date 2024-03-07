import matplotlib.pyplot as plt
import numpy as np
from features import state_range, fourier, feature_st_act, polynomials
from mountain_car import MountainCar
from scipy.stats import truncnorm


NUM_OF_STATES = 2
APPROXIMATION_ORDER = 3
NUM_EPISODES = 1000
NUM_STEPS = 10000
ACTS = [1, -1, 0]
Gamma = 1
mc = MountainCar()


def policy(theta, x):
    return theta.T @ x


def policy_cont_acts(theta_mu, theta_sigma, x):
    return theta_mu.T @ x, np.exp(theta_sigma.T @ x)


def softmax(h_b):
    z = h_b - np.max(h_b)
    return np.exp(z) / np.sum(np.exp(z))


def state_value(weight, x):
    return weight.T @ x


def action_value(state, action, det_pol, grad_det_pol, w, V_weight):
    return (action-det_pol).T @ grad_det_pol.T @ w + state_value(V_weight, state)


def reinforce(alpha=4e-6):
    # weight initialization
    theta = np.zeros((len(ACTS) * (APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))
    rewards_log = []
    step_log = []
    for eps in range(NUM_EPISODES):

        print('Episode ' + str(eps + 1), 'of', str(NUM_EPISODES))

        x_t = np.random.uniform(low=-0.6, high=-0.4)
        x_t_dot = 0
        state = [x_t, x_t_dot]

        mc.goal_reached = False

        grads = []
        rewards = []
        G_hist = []
        actions = []

        trajectory = [(x_t, x_t_dot)]
        tl = 0

        while True:

            # Fourier features
            x_s = fourier(state_range(state, mc.state_ub, mc.state_lb), APPROXIMATION_ORDER)

            # action preference
            h_b = [policy(theta, feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, act, x_s)) for act in ACTS]

            # policies by softmax
            probs = softmax(h_b)

            # take action
            action = np.random.choice(ACTS, p=probs)
            x_s_a = feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, action, x_s)

            next_x_t, next_x_t_dot, reward, status = mc.move(x_t, x_t_dot, action)
            state_prime = [next_x_t, next_x_t_dot]

            # eligibility vector
            grad_sum = np.zeros_like(x_s_a)
            for act in ACTS:
                grad_sum += feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, act, x_s) * probs[ACTS.index(act)]
            grad = x_s_a - grad_sum

            # keeping track of rewards, eligibility vectors, and states trajectory
            rewards.append(reward)
            G_hist.append(reward * (Gamma ** tl))
            grads.append(grad)
            actions.append(action)
            trajectory.append((next_x_t, next_x_t_dot))

            if status:
                print('Episode finished in ' + str(tl + 1) + ' steps')
                rewards_log.append(sum(rewards))
                step_log.append(tl + 1)
                break

            tl += 1
            x_t, x_t_dot = next_x_t, next_x_t_dot
            state = state_prime

        G = np.cumsum(G_hist)
        G = G[::-1]
        for i in range(len(grads)):
            theta += alpha * (Gamma ** i) * G[i] * grads[i]

    return trajectory, rewards_log, step_log


def reinforce_baseline(alpha_theta=4e-6, alpha_w=4e-3):
    # weight initialization
    theta = np.zeros((len(ACTS) * (APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))
    w = np.zeros(((APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))
    rewards_log = []
    step_log = []
    for eps in range(NUM_EPISODES):

        print('Episode ' + str(eps + 1), 'of', str(NUM_EPISODES))

        x_t = np.random.uniform(low=-0.6, high=-0.4)
        x_t_dot = 0
        state = [x_t, x_t_dot]

        mc.goal_reached = False

        grads = []
        rewards = []
        G_hist = []
        actions = []
        all_features = []

        trajectory = [(x_t, x_t_dot)]

        tl = 0

        while True:

            # Fourier features
            x_s = fourier(state_range(state, mc.state_ub, mc.state_lb), APPROXIMATION_ORDER)

            # action preference
            h_b = [policy(theta, feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, act, x_s)) for act in ACTS]

            # policies by softmax
            probs = softmax(h_b)

            # take action
            action = np.random.choice(ACTS, p=probs)
            x_s_a = feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, action, x_s)

            next_x_t, next_x_t_dot, reward, status = mc.move(x_t, x_t_dot, action)
            state_prime = [next_x_t, next_x_t_dot]

            # eligibility vector
            grad_sum = np.zeros_like(x_s_a)
            for act in ACTS:
                grad_sum += feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, act, x_s) * probs[ACTS.index(act)]
            grad = x_s_a - grad_sum

            # keeping track of rewards, eligibility vectors, and states trajectory
            rewards.append(reward)
            G_hist.append(reward * (Gamma ** tl))
            grads.append(grad)
            actions.append(action)
            trajectory.append((x_t, x_t_dot))
            all_features.append(x_s)

            if status:
                print('Episode finished in ' + str(tl + 1) + ' steps')
                rewards_log.append(sum(rewards))
                step_log.append(tl + 1)
                break

            tl += 1
            x_t, x_t_dot = next_x_t, next_x_t_dot
            state = state_prime

        G = np.cumsum(G_hist)
        G = G[::-1]
        for i in range(len(grads)):
            delta = G[i] - state_value(w, all_features[i])
            w += alpha_w * delta * all_features[i]
            theta += alpha_theta * (Gamma ** i) * delta * grads[i]

    return trajectory, rewards_log, step_log


def actor_critic_1step(alpha_theta=5e-4, alpha_w=5e-4):
    theta = np.zeros((len(ACTS) * (APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))
    w = np.zeros(((APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))

    rewards_log = []
    step_log = []

    for eps in range(NUM_EPISODES):

        print('Episode ' + str(eps + 1), 'of', str(NUM_EPISODES))

        x_t = np.random.uniform(low=-0.6, high=-0.4)
        x_t_dot = 0
        state = [x_t, x_t_dot]

        mc.goal_reached = False

        trajectory = [(x_t, x_t_dot)]

        rewards = []

        I = 1
        tl = 0

        while True:
            # Fourier features for state
            x_s = fourier(state_range(state, mc.state_ub, mc.state_lb), APPROXIMATION_ORDER)

            # action preference
            h_b = [policy(theta, feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, act, x_s)) for act in ACTS]

            # policies by softmax
            probs = softmax(h_b)

            # take action
            action = np.random.choice(ACTS, p=probs)
            x_s_a = feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, action, x_s)

            next_x_t, next_x_t_dot, reward, status = mc.move(x_t, x_t_dot, action)

            state_prime = [next_x_t, next_x_t_dot]

            # Fourier features for state_prime
            x_s_prime = fourier(state_range(state_prime, mc.state_ub, mc.state_lb), APPROXIMATION_ORDER)

            # eligibility vector
            grad_sum = np.zeros_like(x_s_a)
            for act in ACTS:
                grad_sum += feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, act, x_s) * probs[ACTS.index(act)]
            grad = x_s_a - grad_sum

            vs_prime = state_value(w, x_s_prime)
            if status:
                vs_prime = 0
            delta = reward + Gamma * vs_prime - state_value(w, x_s)

            w += alpha_w * delta * x_s
            theta += alpha_theta * I * delta * grad

            I *= Gamma
            x_t, x_t_dot = next_x_t, next_x_t_dot

            state = state_prime
            rewards.append(reward)

            # keeping track of rewards, eligibility vectors, and states trajectory
            trajectory.append((next_x_t, next_x_t_dot))

            if status:
                print('Episode finished in ' + str(tl + 1) + ' steps')
                rewards_log.append(sum(rewards))
                step_log.append(tl + 1)
                break

            tl += 1
    return trajectory, rewards_log, step_log


def actor_critic_et(lambda_theta=.8, lambda_w=.8, alpha_theta=4e-6, alpha_w=4e-3):
    theta = np.zeros((len(ACTS) * (APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))
    w = np.zeros(((APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))

    rewards_log = []
    step_log = []

    for eps in range(NUM_EPISODES):

        print('Episode ' + str(eps + 1), 'of', str(NUM_EPISODES))

        x_t = np.random.uniform(low=-0.6, high=-0.4)
        x_t_dot = 0
        state = [x_t, x_t_dot]

        mc.goal_reached = False

        trajectory = [(x_t, x_t_dot)]

        z_theta = 0
        z_w = 0
        I = 1
        tl = 0

        rewards = []

        while True:
            # Fourier features for state
            x_s = fourier(state_range(state, mc.state_ub, mc.state_lb), APPROXIMATION_ORDER)

            # action preference
            h_b = [policy(theta, feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, act, x_s)) for act in ACTS]

            # policies by softmax
            probs = softmax(h_b)

            # take action
            action = np.random.choice(ACTS, p=probs)
            x_s_a = feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, action, x_s)

            next_x_t, next_x_t_dot, reward, status = mc.move(x_t, x_t_dot, action)

            state_prime = [next_x_t, next_x_t_dot]
            rewards.append(reward)

            # Fourier features for state_prime
            x_s_prime = fourier(state_range(state_prime, mc.state_ub, mc.state_lb), APPROXIMATION_ORDER)

            # eligibility vector
            grad_sum = np.zeros_like(x_s_a)
            for act in ACTS:
                grad_sum += feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, act, x_s) * probs[ACTS.index(act)]
            grad = x_s_a - grad_sum
            vs_prime = state_value(w, x_s_prime)
            if status:
                vs_prime = 0
            delta = reward + (Gamma * vs_prime) - state_value(w, x_s)

            z_w = (Gamma * lambda_w * z_w) + x_s
            z_theta = (Gamma * lambda_theta * z_theta) + (I * grad)

            w += alpha_w * delta * z_w
            theta += alpha_theta * delta * z_theta

            I *= Gamma
            x_t, x_t_dot = next_x_t, next_x_t_dot
            state = state_prime

            # keeping track of rewards, eligibility vectors, and states trajectory
            trajectory.append((next_x_t, next_x_t_dot))

            if status:
                print('Episode finished in ' + str(tl + 1) + ' steps')

                rewards_log.append(sum(rewards))
                step_log.append(tl + 1)
                break

            tl += 1

    return trajectory, rewards_log, step_log


def actor_critic_et_cont(lambda_theta=.8, lambda_w=.8, alpha_theta=4e-6, alpha_w=4e-3, alpha_r=4e-3):
    R_bar = 0
    theta = np.zeros((len(ACTS) * (APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))
    w = np.zeros(((APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))

    rewards_log = []
    step_log = []

    x_t = np.random.uniform(low=-0.6, high=-0.4)
    x_t_dot = 0
    state = [x_t, x_t_dot]

    mc.goal_reached = False

    trajectory = [(x_t, x_t_dot)]

    z_theta = 0
    z_w = 0
    tl = 0

    rewards = []

    while True:
        # Fourier features for state
        x_s = fourier(state_range(state, mc.state_ub, mc.state_lb), APPROXIMATION_ORDER)

        # action preference
        h_b = [policy(theta, feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, act, x_s)) for act in ACTS]

        # policies by softmax
        probs = softmax(h_b)

        # take action
        action = np.random.choice(ACTS, p=probs)
        x_s_a = feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, action, x_s)

        next_x_t, next_x_t_dot, reward, status = mc.move(x_t, x_t_dot, action)

        state_prime = [next_x_t, next_x_t_dot]
        rewards.append(reward)

        # Fourier features for state_prime
        x_s_prime = fourier(state_range(state_prime, mc.state_ub, mc.state_lb), APPROXIMATION_ORDER)

        # eligibility vector
        grad_sum = np.zeros_like(x_s_a)
        for act in ACTS:
            grad_sum += feature_st_act(ACTS, APPROXIMATION_ORDER, NUM_OF_STATES, act, x_s) * probs[ACTS.index(act)]
        grad = x_s_a - grad_sum
        vs_prime = state_value(w, x_s_prime)
        if status:
            vs_prime = 0
        delta = reward - R_bar + vs_prime - state_value(w, x_s)
        R_bar += alpha_r * delta
        z_w = (lambda_w * z_w) + x_s
        z_theta = (lambda_theta * z_theta) + grad

        w += alpha_w * delta * z_w
        theta += alpha_theta * delta * z_theta

        x_t, x_t_dot = next_x_t, next_x_t_dot
        state = state_prime

        # keeping track of rewards, eligibility vectors, and states trajectory
        trajectory.append((next_x_t, next_x_t_dot))
        # unnecessary for this algorithm
        if status:
            print('Episode finished in ' + str(tl + 1) + ' steps')

            rewards_log.append(sum(rewards))
            step_log.append(tl + 1)
            break

        tl += 1

    return trajectory, rewards_log, step_log


def actor_critic_et_cont_cont_acts(lambda_theta_mu=.8, lambda_theta_sigma=.8, lambda_w=.8,
                                   alpha_theta_mu=4e-6, alpha_theta_sigma=4e-6, alpha_w=4e-3, alpha_r=4e-3):
    R_bar = 0
    theta_mu = np.zeros(((APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))
    theta_sigma = np.zeros(((APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))
    w = np.zeros(((APPROXIMATION_ORDER + 1) ** NUM_OF_STATES,))

    rewards_log = []
    step_log = []

    x_t = np.random.uniform(low=-0.6, high=-0.4)
    x_t_dot = 0
    state = [x_t, x_t_dot]

    mc.goal_reached = False

    trajectory = [(x_t, x_t_dot)]

    z_theta_mu = 0
    z_theta_sigma = 0
    z_w = 0
    tl = 0

    rewards = []

    while True:
        # Fourier features for state
        x_s = fourier(state_range(state, mc.state_ub, mc.state_lb), APPROXIMATION_ORDER)
        mu_s_theta, sigma_s_theta = policy_cont_acts(theta_mu, theta_sigma, x_s)

        # take action
        action = truncnorm.rvs(-1, 1, mu_s_theta, sigma_s_theta)
        next_x_t, next_x_t_dot, reward, status = mc.move(x_t, x_t_dot, action)

        state_prime = [next_x_t, next_x_t_dot]
        rewards.append(reward)

        # Fourier features for state_prime
        x_s_prime = fourier(state_range(state_prime, mc.state_ub, mc.state_lb), APPROXIMATION_ORDER)

        # policy gradient params

        grad_mu = (1/(sigma_s_theta**2)) * (action - mu_s_theta) * x_s
        grad_theta = (((action - mu_s_theta)**2 / (sigma_s_theta**2)) - 1) * x_s

        vs_prime = state_value(w, x_s_prime)
        if status:
            vs_prime = 0
        delta = reward - R_bar + vs_prime - state_value(w, x_s)
        R_bar += alpha_r * delta
        z_w = (lambda_w * z_w) + x_s
        z_theta_mu = (lambda_theta_mu * z_theta_mu) + grad_mu
        z_theta_sigma = (lambda_theta_sigma * z_theta_sigma) + grad_theta

        w += alpha_w * delta * z_w
        theta_mu += alpha_theta_mu * delta * z_theta_mu
        theta_sigma += alpha_theta_sigma * delta * z_theta_sigma

        x_t, x_t_dot = next_x_t, next_x_t_dot
        state = state_prime

        # keeping track of rewards, eligibility vectors, and states trajectory
        trajectory.append((next_x_t, next_x_t_dot))
        # unnecessary for this algorithm
        if status:
            print('Episode finished in ' + str(tl + 1) + ' steps')

            rewards_log.append(sum(rewards))
            step_log.append(tl + 1)
            break

        tl += 1

    return trajectory, rewards_log, step_log
