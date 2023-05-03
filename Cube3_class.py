import numpy as np
import random


class Cube3(object):
    """
    A class for 3x3x3 Rubik's Cube
    """
    def __init__(self, metric="QTM"):
        self.DTYPE = np.int64
        self.metric = metric

        # define state and goal
        self.reset()  # state
        self.goal = np.arange(0, 9 * 6, dtype=self.DTYPE) // 9

        # define moves
        ## faces and turns
        faces = ["U", "D", "L", "R", "B", "F"]
        ## [90 degrees clockwise, 90 degrees counter-clockwise]
        degrees = ["", "'"]
        degrees_inference = degrees[::-1]
        if self.metric == "HTM":
            # += [180 degrees]
            degrees += ["2"]
            degrees_inference += ["2"]
        else:
            assert self.metric == "QTM"
        self.moves = [f"{f}{n}" for f in faces for n in degrees]
        self.moves_inference = [f"{f}{n}" for f in faces for n in degrees_inference]

        # opposite faces
        self.pairing = {
            "R": "L",
            "L": "R",
            "F": "B",
            "B": "F",
            "U": "D",
            "D": "U",
        }
        # prohibit obviously reduntant moves. 
        if self.metric == "HTM":
            # two subsequent moves on the same face (cancelling or redundant).
            self.moves_available_after = {
                m: [v for v in self.moves if v[0] != m[0]] for m in self.moves
            }
        elif self.metric == "QTM":
            # self-cancelling moves on the same face
            self.moves_available_after = {
                m: [v for v in self.moves if v[0] != m[0]] + [m] for m in self.moves
            }
        else:
            raise

        # vectorize the sticker group replacement operations
        self.__vectorize_moves()

    def reset(self):
        self.state = np.arange(0, 9 * 6, dtype=self.DTYPE) // 9

    def is_solved(self):
        return np.all(self.state == self.goal)

    def state_to_batch(self):
        return np.expand_dims(self.state, axis=0)

    def finger(self, move):
        self.state[self.sticker_target[move]] = self.state[self.sticker_source[move]]

    def apply_scramble(self, scramble):
        if isinstance(scramble, str):
            scramble = scramble.split()
        for m in scramble:
            if m[-1]=='2':
                for _ in range(2):
                    self.finger(m[0])
            else:
                    self.finger(m)

    def scrambler(self, scramble_length):
        """
        A generator function yielding the state and scramble
        """
        while True:
            # reset the self.state, scramble, and retun self.state and scramble moves
            self.reset()
            scramble = []

            for i in range(scramble_length):
                if i:
                    last_move = scramble[-1]
                    if i > 1:  # N(>=3)th moves
                        while True:
                            move = random.choice(self.moves_available_after[last_move])
                            if self.metric == "QTM":
                                if scramble[-2] == last_move == move:
                                    # Two mutually canceling moves in a row
                                    continue
                                elif (
                                    scramble[-2][0] == move[0]
                                    and len(scramble[-2] + move) == 3
                                    and last_move[0] == self.pairing[move[0]]
                                ):
                                    # Two mutually canceling moves sandwiching an opposite face move
                                    continue
                                else:
                                    break
                            elif self.metric == "HTM":
                                if scramble[-2][0] == move[0] and last_move[0] == self.pairing[move[0]]:
                                    # Two mutually canceling moves sandwiching an opposite face move
                                    continue
                                else:
                                    break
                            else:
                                raise
                    else:  # 2nd move
                        move = random.choice(self.moves_available_after[last_move])
                else:  # 1st move
                    move = random.choice(self.moves)

                self.finger(move)
                scramble.append(move)

                yield self.state, move

    def __vectorize_moves(self):  #！！！！
        """
        This method defines ```self.sticker_target``` and ```self.sticker_source``` to manage sticker colors (target is replaced by source).
        They define indices of target and source stickers so that the moves can be vectorized.

        colors:
                0 0 0
                0 0 0
                0 0 0
            2 2 2 5 5 5 3 3 3 4 4 4
            2 2 2 5 5 5 3 3 3 4 4 4
            2 2 2 5 5 5 3 3 3 4 4 4
                1 1 1
                1 1 1
                1 1 1
        order of stickers on each face:
             2  5  8
             1  4  7
            [0] 3  6

        indices of state (each starting with 9*(n-1)):
                         2   5   8
                         1   4   7
                        [0]  3   6
             20  23 26  47  50  53  29  32 35  38  41 44
             19  22 25  46  49  52  28  31 34  37  40 43
            [18] 21 24 [45] 48  51 [27] 30 33 [36] 39 42
                        11   14 17
                        10   13 16
                        [9]  12 15
        """
        self.sticker_target, self.sticker_source = dict(), dict()

        self.sticker_replacement = {
            # Sticker A is replaced by another sticker at index B -> A:B
            "U": {
                0: 6,
                1: 3,
                2: 0,
                3: 7,
                5: 1,
                6: 8,
                7: 5,
                8: 2,
                20: 47,
                23: 50,
                26: 53,
                29: 38,
                32: 41,
                35: 44,
                38: 20,
                41: 23,
                44: 26,
                47: 29,
                50: 32,
                53: 35,
            },
            "D": {
                9: 15,
                10: 12,
                11: 9,
                12: 16,
                14: 10,
                15: 17,
                16: 14,
                17: 11,
                18: 36,
                21: 39,
                24: 42,
                27: 45,
                30: 48,
                33: 51,
                36: 27,
                39: 30,
                42: 33,
                45: 18,
                48: 21,
                51: 24,
            },
            "L": {
                0: 44,
                1: 43,
                2: 42,
                9: 45,
                10: 46,
                11: 47,
                18: 24,
                19: 21,
                20: 18,
                21: 25,
                23: 19,
                24: 26,
                25: 23,
                26: 20,
                42: 11,
                43: 10,
                44: 9,
                45: 0,
                46: 1,
                47: 2,
            },
            "R": {
                6: 51,
                7: 52,
                8: 53,
                15: 38,
                16: 37,
                17: 36,
                27: 33,
                28: 30,
                29: 27,
                30: 34,
                32: 28,
                33: 35,
                34: 32,
                35: 29,
                36: 8,
                37: 7,
                38: 6,
                51: 15,
                52: 16,
                53: 17,
            },
            "B": {
                2: 35,
                5: 34,
                8: 33,
                9: 20,
                12: 19,
                15: 18,
                18: 2,
                19: 5,
                20: 8,
                33: 9,
                34: 12,
                35: 15,
                36: 42,
                37: 39,
                38: 36,
                39: 43,
                41: 37,
                42: 44,
                43: 41,
                44: 38,
            },
            "F": {
                0: 24,
                3: 25,
                6: 26,
                11: 27,
                14: 28,
                17: 29,
                24: 17,
                25: 14,
                26: 11,
                27: 6,
                28: 3,
                29: 0,
                45: 51,
                46: 48,
                47: 45,
                48: 52,
                50: 46,
                51: 53,
                52: 50,
                53: 47,
            },
        }
        for m in self.moves:
            if len(m) == 1:
                assert m in self.sticker_replacement
            else:
                if "'" in m:
                    self.sticker_replacement[m] = {
                        v: k for k, v in self.sticker_replacement[m[0]].items()
                    }
                elif "2" in m:
                    self.sticker_replacement[m] = {
                        k: self.sticker_replacement[m[0]][v]
                        for k, v in self.sticker_replacement[m[0]].items()
                    }
                else:
                    raise

            self.sticker_target[m] = list(self.sticker_replacement[m].keys())
            self.sticker_source[m] = list(self.sticker_replacement[m].values())

            for i, idx in enumerate(self.sticker_target[m]):
                assert self.sticker_replacement[m][idx] == self.sticker_source[m][i]
