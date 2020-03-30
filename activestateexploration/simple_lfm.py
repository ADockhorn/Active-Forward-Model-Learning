from sokoban.fast_sokoban import extract_all_patterns
import numpy as np


class MinimalLocalForwardModel:
    def __init__(self, model, mask, span, remember_predictions=False):
        self.model = model
        self.mask = mask
        self.span = span
        self.remember_predictions = remember_predictions
        if self.remember_predictions:
            self.known_predictions = dict()
        pass

    def extract_unknown_patterns(self, game_state, action, mask, span):
        """
        :param prev_game_state:
        :param action:
        :param game_state:
        :param mask:
        :param span:
        :param data_set:
        :return:
        """
        prediction_mask = np.zeros(game_state.shape, dtype=np.bool)
        result = np.zeros(game_state.shape, dtype=np.int)
        data_set = []

        # only iterate over positions that were affected by the game state's changes
        positions = [(x, y) for x in range(game_state.shape[0]) for y in range(game_state.shape[1])]
        ext_game_state_grid = np.pad(game_state, span, "constant", constant_values=1)

        for i, (x, y) in enumerate(positions):
            el = ext_game_state_grid[span + x - span: span + x + span + 1, span + y - span: span + y + span + 1][
                mask].tolist()
            el.append(action)
            el = tuple(el)
            if el in self.known_predictions:
                result[x, y] = self.known_predictions[el]
            else:
                prediction_mask[x, y] = 1
                data_set.append(el)
        return data_set, prediction_mask, result

    def predict(self, level, action):
        if self.remember_predictions:
            data, prediction_mask, result = self.extract_unknown_patterns(level, action, self.mask, self.span)

            if len(data) > 0:
                prediction = self.model.predict(data)
                result[prediction_mask] = prediction
                for el, pred in zip(data, prediction):
                    self.known_predictions[el] = pred
            return result
        else:
            data = extract_all_patterns(level, action, self.mask, self.span)
            return self.model.predict(data).reshape(level.shape)
