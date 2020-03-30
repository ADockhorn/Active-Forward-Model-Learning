import numpy as np
import random
import os
from numba import njit  # deactive this line and following "@njit" lines if numba is not compatible with your processor
import copy
import operator

level_tiles = {".": '0',    # empty floor
               "w": '1',    # wall
               "A": '2',    # avatar
               "*": '3',    # box
               "o": '4',    # hole
               "u": '5',    # avatar on hole
               "+": '6'}    # box on hole


level_tiles_reverse = {y: x for (x,y) in level_tiles.items()}


class Sokoban:
    """
        This class provides a numpy based implementation of the game Sokoban, which focusses on speed.
        The level-tile encoding is provided at the top of the file and maps the Sokoban tile set used in:
            http://sokobano.de/wiki/index.php?title=Level_format
        to the int-based notation. "level_tiles_reverse" lets the user map the level back to original notation.
    """
    def __init__(self, level=0, noise_rate=0):
        """

        :param level: (int, str) either an int for loading a user_generated level or a filename to load
        :param noise_rate: [0, 1]: values greater than 0 will add uniform noise to the level observation, see
            get_tile_map for details
        """
        if isinstance(level, int):
            if level == -1:
                self.level, self.base_level, self.pos = self.load_test_level()
            else:
                self.level, self.base_level, self.pos = self.load_level_by_number(level)
        elif isinstance(level, str):
            self.level, self.base_level, self.pos = self.load_level_from_file(level)

        if isinstance(level, np.ndarray):
            self.level, self.base_level, self.pos = self.generate_level_from_matrix(level)
        self.noise_rate = noise_rate

    def is_terminal(self):
        """

        :return: true if the level does not consist any boxes or uncovered holes
        """
        return np.sum(self.level == 3) == 0 and np.sum(self.level == 4) == 0

    def next(self, action):
        step(self.level, self.base_level, self.pos, action)

    def get_width(self):
        return self.level.shape[1]

    def get_height(self):
        return self.level.shape[0]

    def get_grid(self):
        return self.level

    def get_tile_map(self):
        obs = self.level.copy()
        if self.noise_rate > 0:
            tiles = list(level_tiles.values())
            for x in range(obs.shape[0]):
                for y in range(obs.shape[1]):
                    if random.random() < self.noise_rate:
                        obs[x, y] = random.choice(tiles)
        return obs

    def get_copy(self):
        return copy.deepcopy(self)

    @property
    def get_random_action(self):
        return random.choice([0, 1, 2, 3])

    @staticmethod
    def load_level_by_number(level_no):
        return Sokoban.load_level_from_file(os.sep.join(["sokoban", "user_generated_levels", f"level-{level_no}.txt"]))

    @staticmethod
    def load_level_from_file(filename):
        grid_data = []
        with open(filename, "r") as file:
            lines = file.readlines()
            dims = lines[1].split(",")
            w = int(dims[0])
            h = int(dims[1])

            for line in lines[2:]:
                grid_data += line[:-1]

        level_data = np.array([level_tiles[cell] for cell in grid_data], dtype=np.dtype("i")).reshape(h, w)
        level = level_data.copy()
        base_level = level_data.copy()

        base_level[level == 3] = 0
        base_level[level == 2] = 0
        pos = np.argwhere(level_data == 2)

        return level, base_level, pos.flatten()

    def save_level_to_file(self, filename, leveltitle):
        with open(filename, "w") as file:
            file.write(leveltitle)
            file.write("\n")
            file.write(f"{self.get_width()},{self.get_height()}")
            file.write("\n")

            for x in self.level:
                file.write("".join([level_tiles_reverse[str(y)] for y in x]))
                file.write("\n")

    @staticmethod
    def load_test_level():
        width = 10
        height = 10

        level = np.zeros((width, height))
        level[:, 0] = 1
        level[:, height - 1] = 1
        level[0, :] = 1
        #level[1, :] = 1
        #level[2, :] = 1
        #level[3, :] = 1
        #level[7, :] = 1
        #level[8, :] = 1
        #level[9, :] = 1
        level[width - 1, :] = 1

        #level[4, 5] = 4
        #level[5, 4] = 4
        #level[5, 6] = 4
        #level[6, 5] = 4
        base_level = level.copy()

        #level[5, 5] = 3

        pos = np.array([4, 4])
        level[pos[0], pos[1]] = 2

        return level, base_level, pos

    def generate_level_from_matrix(self, matrix):

        level = matrix.copy()
        base_level = level.copy()

        pos = np.argwhere(np.logical_or(level == 2, level == 5))[0]

        if level[pos[0], pos[1]] == 2:
            base_level[pos[0], pos[1]] = 0
        else:
            base_level[pos[0], pos[1]] = 4

        box_positions = np.argwhere(np.logical_or(level == 3, level == 6))
        for box_pos in box_positions:
            if level[box_pos[0], box_pos[1]] == 3:
                base_level[box_pos[0], box_pos[1]] = 0
            else:
                base_level[box_pos[0], box_pos[1]] = 4

        return level, base_level, pos


# noinspection DuplicatedCode
@njit
def step(level_matrix, base_level, pos, action):
    """ Determining the next game-state based on the players action. Results are calculated in-place, so be sure to
    copy the arrays in case you want to remember the previous state

    :param level_matrix:
    :param base_level:
    :param pos:
    :param action:
    :return:
    """
    if action == 0:
        new_pos_x, new_pos_y = pos[0], pos[1] + 1
    elif action == 1:
        new_pos_x, new_pos_y = pos[0] + 1, pos[1]
    elif action == 2:
        new_pos_x, new_pos_y = pos[0], pos[1] - 1
    elif action == 3:
        new_pos_x, new_pos_y = pos[0] - 1, pos[1]
    else:
        new_pos_x, new_pos_y = (0, 0)

    target = level_matrix[new_pos_x, new_pos_y]
    # wall
    if target == 1:
        return False
    # empty or hole
    elif target == 0 or target == 4:
        # avatar on hole?
        if base_level[new_pos_x, new_pos_y] == 4:
            level_matrix[new_pos_x, new_pos_y] = 5
        else:
            level_matrix[new_pos_x, new_pos_y] = 2
        level_matrix[pos[0], pos[1]] = base_level[pos[0], pos[1]]
        pos[0] = new_pos_x
        pos[1] = new_pos_y
    # box
    elif target == 3 or target == 6:
        if action == 0:
            next_tile_x, next_tile_y = new_pos_x, new_pos_y + 1
        elif action == 1:
            next_tile_x, next_tile_y = new_pos_x + 1, new_pos_y
        elif action == 2:
            next_tile_x, next_tile_y = new_pos_x, new_pos_y - 1
        elif action == 3:
            next_tile_x, next_tile_y = new_pos_x - 1, new_pos_y

        pushtarget = level_matrix[next_tile_x, next_tile_y]
        # push into wall or box
        if pushtarget == 1 or pushtarget == 3 or pushtarget == 6:
            return False
        # push into empty or hole
        elif pushtarget == 0 or pushtarget == 4:
            if base_level[next_tile_x, next_tile_y] == 4:
                level_matrix[next_tile_x, next_tile_y] = 6
            else:
                level_matrix[next_tile_x, next_tile_y] = 3

            # Avatar on hole?
            if base_level[new_pos_x, new_pos_y] == 4:
                level_matrix[new_pos_x, new_pos_y] = 5
            else:
                level_matrix[new_pos_x, new_pos_y] = 2

            level_matrix[pos[0], pos[1]] = base_level[pos[0], pos[1]]
            pos[0] = new_pos_x
            pos[1] = new_pos_y

        return False
    return False

@njit
def oracle(pattern, mask, action):
    """ This oracle function returns the future state of the patterns center-tile after applying the action.

    :param pattern: the pattern to be requested
    :param mask: the underlying mask of the pattern
    :param action: the latest action
    :return:
    """
    level = np.ones((mask.shape), dtype=np.dtype("i"))
    ind = 0
    for i in range(level.shape[0]):
        for j in range(level.shape[1]):
            if mask[i, j]:
                level[i, j] = pattern[ind]
                ind += 1
    #level[mask] = pattern
    cx, cy = [(x-1)//2 for x in mask.shape]
    center_tile = level[cx, cy]

    # a wall will never change to something else
    if center_tile == 1:
        return 1

    if action == 0:
        pos_to_check_x, pos_to_check_y = cx, cy - 1
    elif action == 1:
        pos_to_check_x, pos_to_check_y = cx - 1, cy
    elif action == 2:
        pos_to_check_x, pos_to_check_y = cx, cy + 1
    elif action == 3:
        pos_to_check_x, pos_to_check_y = cx + 1, cy
    else:
        return -1 # illegal action

    # does the player move to the tile?
    neighbor = level[pos_to_check_x, pos_to_check_y]
    if neighbor == 2 or neighbor == 5:
        if center_tile == 4 or center_tile == 0:
            if center_tile == 4: # if hole
                return 5    # player on hole
            else:
                return 2    # player

        if center_tile == 3 or center_tile == 6: # can the box be pushed?
            if action == 0:
                push_target_x, push_target_y = cx, cy + 1
            elif action == 1:
                push_target_x, push_target_y = cx + 1, cy
            elif action == 2:
                push_target_x, push_target_y = cx, cy - 1
            elif action == 3:
                push_target_x, push_target_y = cx - 1, cy
            else:
                return -1

            push_target = level[push_target_x, push_target_y]
            if push_target == 0 or push_target == 4:    # box can be pushed -> player can move
                if center_tile == 3:    # pushed box away from empty field
                    return 2
                else:   # pushed box away from hole
                    return 5

    # if the neighbor is a pushable box, check if the player is one field behind it
    if (neighbor == 3 or neighbor == 6) and (center_tile == 0 or center_tile == 4):
        if action == 0:
            push_start_x, push_start_y = pos_to_check_x, pos_to_check_y - 1
        elif action == 1:
            push_start_x, push_start_y = pos_to_check_x - 1, pos_to_check_y
        elif action == 2:
            push_start_x, push_start_y = pos_to_check_x, pos_to_check_y + 1
        elif action == 3:
            push_start_x, push_start_y = pos_to_check_x + 1, pos_to_check_y
        else:
            return -1

        if level[push_start_x, push_start_y] == 2 or level[push_start_x, push_start_y] == 5:
            if center_tile == 0:
                return 3
            else:
                return 6

    if center_tile == 2 or center_tile == 5:
        if action == 0:
            move_target_x, move_target_y = cx, cy + 1
        elif action == 1:
            move_target_x, move_target_y = cx + 1, cy
        elif action == 2:
            move_target_x, move_target_y = cx, cy - 1
        elif action == 3:
            move_target_x, move_target_y = cx - 1, cy
        else:
            return -1

        move_target = level[move_target_x, move_target_y]
        if move_target == 1:
            return center_tile

        if move_target == 0 or move_target == 4:    # position is free to move to
            if center_tile == 5:
                return 4
            else:
                return 0

        # check if neighboring box can be pushed
        if action == 0:
            push_target_x, push_target_y = move_target_x, move_target_y + 1
        elif action == 1:
            push_target_x, push_target_y = move_target_x + 1, move_target_y
        elif action == 2:
            push_target_x, push_target_y = move_target_x, move_target_y - 1
        elif action == 3:
            push_target_x, push_target_y = move_target_x - 1, move_target_y
        else:
            return -1

        push_target = level[push_target_x, push_target_y]
        if push_target == 0 or push_target == 4:
            if center_tile == 5:
                return 4
            else:
                return 0

    return center_tile


def add_patterns_to_data_set(prev_game_state, action, game_state, mask, span, data_set):
    """ Extracting the local forward model pattern for each cell of the grid's game-state.

    :param prev_game_state: game-state at time t
    :param action: players action at time t
    :param game_state: resulting game-state at time t+1
    :param mask: square pattern mask (boolean array to mark which tiles should be included.
    :param span: The span of the mask.
    :param data_set: the data_set dict to which the observation should be added to
        data_set[pattern,action] = {result: observation_count}
        use transform_data_set(data_set, "arg_max") to transform this into a training set
    """
    # only iterate over positions that were affected by the game state's changes
    positions = [(x, y) for x in range(game_state.shape[0]) for y in range(game_state.shape[1])]
    ext_game_state_grid = np.pad(prev_game_state, span, "constant", constant_values=1)

    for x, y in positions:
        el = ext_game_state_grid[span + x - span: span + x + span + 1, span + y - span: span + y + span + 1][mask].tolist()
        el.append(action)
        t = tuple(el)
        if t not in data_set:
            data_set[t] = {game_state[x, y]: 1}
        else:
            if game_state[x, y] in data_set[t]:
                data_set[t][game_state[x, y]] += 1
            else:
                data_set[t][game_state[x, y]] = 1


def extract_all_patterns(game_state, action, mask, span):
    """ Extracting the local forward model pattern for each cell of the grid's game-state and returning a numpy array

    :param prev_game_state: game-state at time t
    :param action: players action at time t
    :param game_state: resulting game-state at time t+1
    :param mask: square pattern mask (boolean array to mark which tiles should be included.
    :param span: The span of the mask.
    :return: np.ndarray of observed patterns
    """

    data_set = np.zeros((game_state.shape[0]*game_state.shape[1], np.sum(mask)+1))

    # only iterate over positions that were affected by the game state's changes
    positions = [(x, y) for x in range(game_state.shape[0]) for y in range(game_state.shape[1])]
    ext_game_state_grid = np.pad(game_state, span, "constant", constant_values=1)

    for i, (x, y) in enumerate(positions):
        el = ext_game_state_grid[span + x - span: span + x + span + 1, span + y - span: span + y + span + 1][mask].tolist()
        el.append(action)
        data_set[i, :] = el
    return data_set


def extract_all_patterns_without_action(game_state, mask, span):
    """ extract patterns without providing the agent's action. Used for copying the pattern for each possible action.

    :param prev_game_state: game-state at time t
    :param action: players action at time t
    :param game_state: resulting game-state at time t+1
    :param mask: square pattern mask (boolean array to mark which tiles should be included.
    :param span: The span of the mask.
    :return: np.ndarray of observed patterns
    """

    data_set = np.zeros((game_state.shape[0]*game_state.shape[1], np.sum(mask)), dtype=np.int)

    # only iterate over positions that were affected by the game state's changes
    positions = [(x, y) for x in range(game_state.shape[0]) for y in range(game_state.shape[1])]
    ext_game_state_grid = np.pad(game_state, span, "constant", constant_values=1)

    for i, (x, y) in enumerate(positions):
        el = ext_game_state_grid[span + x - span: span + x + span + 1, span + y - span: span + y + span + 1][mask].tolist()
        data_set[i, :] = el
    return data_set


def get_all_patterns_and_rotations(prev_game_state, action, game_state, mask, span, data_set):
    """

    :param prev_game_state:
    :param action:
    :param game_state:
    :param mask:
    :param span:
    :param data_set:
    :return:
    """
    prev_game_state = prev_game_state.copy()
    game_state = game_state.copy()
    mask = mask.copy()

    add_patterns_to_data_set(prev_game_state, action, game_state, mask, span, data_set)
    for i in range(3):
        prev_game_state = np.rot90(prev_game_state)
        game_state = np.rot90(game_state)
        mask = np.rot90(mask)
        action = (action - 1) if (action > 0) else 3
        add_patterns_to_data_set(prev_game_state, action, game_state, mask, span, data_set)


def transform_data_set(data_set, mode="all"):
    if mode == "all":
        nrows = sum([data_set[x][y] for x in data_set for y in data_set[x]])
        np_data_set = np.empty((nrows, len(next(iter(data_set)))+1), dtype=np.dtype("int"))
        row_idx = 0
        for x in data_set:
            for y in data_set[x]:
                for i in range(data_set[x][y]):
                    np_data_set[row_idx, :-1] = [cell for cell in x]
                    np_data_set[row_idx, -1] = y
                    row_idx += 1

    elif mode == "argmax":
        np_data_set = np.empty((len(data_set), len(next(iter(data_set)))+1), dtype=np.dtype("int"))
        for row_idx, x in enumerate(data_set):
            np_data_set[row_idx, :-1] = [cell for cell in x]
            np_data_set[row_idx, -1] = max(data_set[x].items(), key=operator.itemgetter(1))[0]
    elif mode == "simple":
        np_data_set = np.empty((len(data_set), len(next(iter(data_set)))+1), dtype=np.dtype("int"))
        for row_idx, x in enumerate(data_set):
            np_data_set[row_idx, :-1] = [cell for cell in x]
            np_data_set[row_idx, -1] = data_set[x]
    else:
        raise ValueError(f"mode {mode} unknown, choose one of (all, argmax, simple)")
    return np_data_set


if __name__ == "__main__":
    from abstractclasses.AbstractNeighborhoodPattern import CrossNeighborhoodPattern

    span = 2
    mask = CrossNeighborhoodPattern(span).get_mask()

    # load level and apply action to it
    game = Sokoban(0)
    action = 2
    prev_game_state = game.level.copy()
    game.next(action)

    # extract patterns and transform in training data set
    data_set = dict()
    add_patterns_to_data_set(prev_game_state, action, game.level, mask, span, data_set)
    data_set = transform_data_set(data_set, "argmax")

    # test oracle prediction
    oracle_prediction = [oracle(data_set[i, :-2], mask, data_set[i, -2])
                         for i in range(len(data_set))]
    print(np.all([x == y for (x, y) in zip(oracle_prediction, data_set[:, -1])]))

