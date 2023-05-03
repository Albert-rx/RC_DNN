METRIC = "QTM"
SCRAMBLE_LENGTH = {"QTM": 26, "HTM": 20}[METRIC]

from Cube3_class import *
import multiprocessing
import numpy as np
import multiprocessing
from tqdm import trange
from IPython.display import clear_output
import torch
from torch import nn

from Cube3_class import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = Cube3()

def plt_history(h,plt):
    fig, axes = plt.subplots(1, 1, figsize=[4, 4])
    axes.plot(h)
    axes.set_xscale("log")
    axes.set_title("Loss")
    plt.show()


global get_minibatch

def get_minibatch(i):
    __dtype = np.int64
    scramble_length=SCRAMBLE_LENGTH
    n_jobs=multiprocessing.cpu_count()
    envs = [Cube3()] * n_jobs
    generators = [c.scrambler(scramble_length=scramble_length) for c in envs]

    states = np.zeros((scramble_length, 9 * 6), dtype=__dtype)
    last_moves = np.zeros((scramble_length,), dtype=__dtype)
    g_local = generators[i % n_jobs]
    for j in range(scramble_length):
        _, last_move = next(g_local)
        states[j, :] = envs[i % n_jobs].state  # _to_numpy()
        last_moves[j] = env.moves.index(last_move)

    return states, last_moves
# create the Pool instance after defining `envs` and `get_minibatch`



def batch_generator(
        batch_size_per_depth,
        scramble_length=SCRAMBLE_LENGTH,
        n_jobs=multiprocessing.cpu_count(),
    ):
    # setup
    __dtype = np.int64
    batch_size = batch_size_per_depth * scramble_length
    # multiprocessing
    envs = [Cube3()] * n_jobs
    generators = [c.scrambler(scramble_length=scramble_length) for c in envs]

    
    # create the Pool instance after defining `envs` and `get_minibatch`
    p = multiprocessing.Pool(n_jobs)
    for _ in iter(int, 1):
        ret = p.map(get_minibatch, list(range(batch_size_per_depth)))
        batch_x = np.concatenate([e[0] for e in ret])
        batch_y = np.concatenate([e[1] for e in ret], axis=0)
        yield (batch_x, batch_y)


def train(config,model,plt):

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    BATCH_SIZE = config.batch_size_per_depth * config.scramble_length
    scramble_length = config.scramble_length
    g = batch_generator(
        batch_size_per_depth=config.batch_size_per_depth,
        scramble_length=config.scramble_length,
    )
    h = []

    for i in trange(1, config.num_train_steps + 1, smoothing=0):
        # prep
        batch_x, batch_y = next(g)
        batch_x, batch_y = torch.from_numpy(batch_x).to(device), torch.from_numpy(
            batch_y
        ).to(device)

        # update
        pred_y = model(batch_x)
        loss = loss_fn(pred_y, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        h.append(loss.item())
        if config.interval_steps_plot and i % config.interval_steps_plot == 0:
            clear_output()
            plt_history(h,plt)
        if config.interval_steps_save and i % config.interval_steps_save == 0:
            #torch.save(model.state_dict(), f"{i}steps.pth")  #original

            #torch.save(model, f"{i}steps.pth")  #尝试保存整个网络，查看是否成功

            torch.jit.save(torch.jit.script(model),f"{i}steps_script.pth")

            # x = torch.randn(1, 3, 256, 256, requires_grad=True).cuda()
            # script_module = torch.jit.trace(model,x,strict=False)
            # torch.jit.save(script_module, f"{i}steps_jitscript.pth")

            # x = torch.randn(1, 3, 256, 256, requires_grad=True).cuda()
            # script_module = torch.jit.trace(model.module,x,strict=False)
            # torch.jit.save(script_module, f"{i}steps_jitscript.pth")
            
            print("Model saved.")

