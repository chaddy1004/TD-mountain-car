import argparse
import gym

from td_zero import TDZeroAgent
from mc_control import McControlAgent
from td_n import TDNAgent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", type=str, default="n_step_sarsa", help="which algorithm to run")
    args = vars(ap.parse_args())
    env = gym.make('MountainCar-v0')
    n_test_iterations = 100
    if args["algorithm"] == "sarsa":
        print(args["algorithm"])
        agent = TDZeroAgent(gamma=0.99, n_bins=30, n_iterations=5000, env=env)
        agent.sarsa()
        avg_score = agent.test(n_test_iterations=n_test_iterations, render=False)
        agent.generate_plot(name=args["algorithm"])
        print(
            f"Average Scores over {n_test_iterations} iterations: {avg_score}")

    elif args["algorithm"] == "q-learning":
        print(args["algorithm"])
        agent = TDZeroAgent(gamma=0.99, n_bins=30, n_iterations=5000, env=env)
        agent.q_learning()
        avg_score = agent.test(n_test_iterations=n_test_iterations, render=False)
        agent.generate_plot(name=args["algorithm"])
        print(
            f"Average Scores over {n_test_iterations} iterations: {avg_score}")

    elif args["algorithm"] == "expected-sarsa":
        print(args["algorithm"])
        agent = TDZeroAgent(gamma=0.99, n_bins=30, n_iterations=5000, env=env)
        agent.expected_sarsa()
        avg_score = agent.test(n_test_iterations=n_test_iterations, render=False)
        agent.generate_plot(name=args["algorithm"])
        print(
            f"Average Scores over {n_test_iterations} iterations: {avg_score}")

    elif args["algorithm"] == "mc-control":
        print(args["algorithm"])
        agent = McControlAgent(gamma=1, n_bins=10, n_iterations=2000, env=env)
        agent.mc_control()
        avg_score = agent.test(n_test_iterations=n_test_iterations, render=False)
        agent.generate_plot(name=args["algorithm"])
        print(
            f"Average Scores over {n_test_iterations} iterations: {avg_score}")
    elif args["algorithm"] == "n_step_sarsa":
        agent = TDNAgent(gamma=0.99, n_bins=30, n_iterations=5000, env=env, n=2)
        agent.sarsa()
        avg_score_2 = agent.test(n_test_iterations=100, render=False)
        agent.generate_plot(name=args["algorithm"])

        agent = TDNAgent(gamma=0.99, n_bins=30, n_iterations=5000, env=env, n=3)
        agent.sarsa()
        avg_score_3 = agent.test(n_test_iterations=100, render=False)
        agent.generate_plot(name=args["algorithm"])

        agent = TDNAgent(gamma=0.99, n_bins=30, n_iterations=5000, env=env, n=4)
        agent.sarsa()
        avg_score_4 = agent.test(n_test_iterations=100, render=False)
        agent.generate_plot(name=args["algorithm"])

        print(
            f"Average Scores over {n_test_iterations} iterations: 2-step: {avg_score_2}, 3-step: {avg_score_3}, 4-step: {avg_score_4}")
