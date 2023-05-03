METRIC = "QTM"
SCRAMBLE_LENGTH = {"QTM": 26, "HTM": 20}[METRIC]

import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import time
from scipy.special import softmax
from copy import deepcopy


from Cube3_class import *
from classes import *

plt.rcParams["axes.prop_cycle"] = cycler(color=["#212121", "#2180FE", "#EB4275"])

def plt_history(h):
    fig, axes = plt.subplots(1, 1, figsize=[4, 4])
    axes.plot(h)
    axes.set_xscale("log")
    axes.set_title("Loss")
    plt.show()

def regression_coef(x, y):
    coef = np.array(y) / np.array(x)
    coef = np.mean(np.squeeze(coef))
    return coef

def plot_result(solutions_all, num_nodes_all, times_all):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax = ax.ravel()
    for i, result in enumerate([solutions_all, num_nodes_all, times_all]):
        result = [e for e in result if e is not None]
        if i == 0:  # soltions
            result = [len(e) for e in result if e is not None]
            ax[i].axvline(
                np.mean(result),
                color="#00ffff",
                label=f"mean={round(np.mean(result),3)}",
            )
            result = {i: result.count(i)
                      for i in range(min(result), max(result) + 1)}
            ax[i].bar(
                result.keys(),
                result.values(),
                width=1.0,
                label=f"Success: {len([len(e) for e in solutions_all if e is not None])}/{len(solutions_all)}",
            )
            ax[i].legend()
            ax[i].set_xlabel("Solution length")
            ax[i].set_ylabel("Frequency")
        else:
            ax[i].hist(result)
            ax[i].axvline(
                np.mean(result),
                color="#00ffff",
                label=f"mean={round(np.mean(result),3)}",
            )
            ax[i].legend()
            if i == 1:
                ax[i].set_xlabel("No. of nodes")
            else:
                ax[i].set_xlabel("Calculation time (s)")

    solution_lengths, num_nodes, times = [
        [e for e in result if e is not None]
        for result in [solutions_all, num_nodes_all, times_all]
    ]
    solution_lengths = [len(e) for e in solution_lengths]

    for (xlabel, ylabel), (x, y) in [
        [("Solution lengths", "No. of nodes"), (solution_lengths, num_nodes)],
        [("No. of nodes", "Calculation time (s)"), (num_nodes, times)],
        [("Calculation time (s)", "Solution lengths"), (times, solution_lengths)],
    ]:
        i += 1
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        x_range = np.linspace(0, max(x), 100)
        ax[i].plot(
            x_range,
            x_range * regression_coef(x, y),
            label=f"slope={round(regression_coef(x, y), 5)}",
            color="#00ffff",
        )
        ax[i].scatter(x, y)
        ax[i].legend()

    plt.show()

def beam_search(
        env,
        model,
        max_depth,
        beam_width,
        __eval = SearchConfig.eval__,
        skip_redundant_moves=True,
    ):
    """
    Best-first search algorithm.
    Input:
        env: A scrambled instance of the given environment. 
        beam_width: Number of top solutions to return per depth.
        max_depth: Maximum depth of the search tree.
        __eval: Evaluation method for sorting nodes to expand, based on DNN outputs: 'softmax', 'logits', or 'cumprod'. 
        skip_redundant_moves: If True, skip redundant moves.
        ...
    Output: 
        if solved successfully:
            True, {'solutions':solution path, "num_nodes":number of nodes expanded, "times":time taken to solve}
        else:
            False, None
    """
    with torch.no_grad():
        # metrics
        num_nodes, time_0 = 0, time.time()
        candidates = [
            {"state":deepcopy(env.state), "path":[], "value":1.}
        ] # list of dictionaries

        for depth in range(max_depth+1):
            # TWO things at a time for every candidate: 1. check if solved & 2. add to batch_x
            batch_x = np.zeros((len(candidates), env.state.shape[-1]), dtype=np.int64)
            for i,c in enumerate(candidates):
                c_path, env.state = c["path"], c["state"]
                if c_path:
                    env.finger(c_path[-1])
                    num_nodes += 1
                    if env.is_solved():
                        return True, {'solutions':c_path, "num_nodes":num_nodes, "times":time.time()-time_0}
                batch_x[i, :] = env.state

            # after checking the nodes expanded at the deepest    
            if depth==max_depth:
                print("Solution not found.")
                return False, None

            # make predictions with the trained DNN
            batch_x = torch.from_numpy(batch_x).to(device)
            batch_p = model(batch_x).to("cpu").detach().numpy()

            # loop over candidates
            candidates_next_depth = []  # storage for the depth-level candidates storing (path, value, index).
            for i, c in enumerate(candidates):
                c_path = c["path"]
                value_distribution = batch_p[i, :] # output logits for the given state
                if __eval in ["softmax","cumprod"]:
                    value_distribution = softmax(value_distribution)
                    if __eval=="cumprod":
                        value_distribution *= c["value"] # multiply the cumulative probability so far of the expanded path

                for m, value in zip(env.moves_inference, value_distribution): # iterate over all possible moves.
                    # predicted value to expand the path with the given move.

                    if c_path and skip_redundant_moves:
                        if env.metric=='QTM':
                            if m not in env.moves_available_after[c_path[-1]]:
                                # Two mutually canceling moves
                                continue
                            elif len(c_path) > 1:
                                if c_path[-2] == c_path[-1] == m:
                                    # three subsequent same moves
                                    continue
                                elif (
                                    c_path[-2][0] == m[0]
                                    and len(c_path[-2] + m) == 3
                                    and c_path[-1][0] == env.pairing[m[0]]
                                ):
                                    # Two mutually canceling moves sandwiching an opposite face move
                                    continue
                        elif env.metric=='HTM':
                            if c_path:
                                if skip_redundant_moves:
                                    if m[0] == c_path[-1][0]:
                                        # Two mutually canceling moves
                                        continue
                                    elif len(c_path)>1:
                                        if c_path[-2][0] == m[0] and c_path[-1][0] == env.pairing[m[0]]:
                                            # Two mutually canceling moves sandwiching an opposite face move
                                            continue
                        else:
                            raise

                    # add to the next-depth candidates unless 'continue'd.
                    candidates_next_depth.append({
                        'state':deepcopy(c['state']),
                        "path": c_path+[m],
                        "value":value,
                    })

            # sort potential paths by expected values and renew as 'candidates'
            candidates = sorted(candidates_next_depth, key=lambda item: -item['value'])
            # if the number of candidates exceed that of beam width 'beam_width'
            candidates = candidates[:beam_width]

