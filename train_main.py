import torch

from classes import *
from newfun import *
from Train import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)

print(f'device: {device}')
print(f'multiprocessing.cpu_count(): {multiprocessing.cpu_count()}')


METRIC = "QTM"
SCRAMBLE_LENGTH = {"QTM": 26, "HTM": 20}[METRIC]


if __name__ =="__main__":
    print(11111111111111111111111111111111)
    plt.rcParams["axes.prop_cycle"] = cycler(color=["#212121", "#2180FE", "#EB4275"])

    train(TrainConfig,model,plt)
    print(f"Trained on data equivalent to {TrainConfig.batch_size_per_depth * TrainConfig.num_train_steps} solves.")
