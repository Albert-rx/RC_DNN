from newfun import *
from classes import *

#scramble = """F U U L L B B F' U L L U R R D D L' B L L B' R R U U""" # Scramble for current *human* world record 
scramble = """R U L U' F U B U D R F U F L U R L B D F U F U'""" 
beam_width = 2**10
max_depth = 52 # arbitrary

# load the trained model
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model = torch.jit.load("./1000steps_no.pth").to(device)#original
#model = torch.load("./1000steps_no.pth").to(device)  
#model=torch.load("./100steps_nojit.pth").to(device)

model =torch.jit.load("./20steps_script.pth").to(device)

model.eval()

# set up Rubik's Cube environment and apply the scramble
#from utils import environments
#env = environments.Cube3()
env.apply_scramble(scramble)

# execute a beam search
#from utils import search
success, result = beam_search(env, model, max_depth, beam_width)


if success:
    print('Solution:', ' '.join(result['solutions']))
    print('Length:', len(result['solutions']))
    env.reset()
    env.apply_scramble(scramble)
    env.apply_scramble(result['solutions'])
    assert env.is_solved()
else:
    print('Failed') 