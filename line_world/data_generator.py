from line_world.utils import ParamsProc, Component
from skimage.transform import rotate
import numpy as np


class DataGenerator(Component):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.add(
            'grid_size', int,
            'The size of the grid on which the line world lives'
        )
        proc.add(
            'thickness', int,
            'The thickness of the lines in the line world'
        )
        proc.add(
            'length_range', tuple,
            'The range of the length for the lines in the line world'
        )
        proc.add(
            'n_rotations', int,
            'The number of angles of rotations we are going to look at'
        )
        return proc

    @staticmethod
    def params_proc(params):
        params['rotation_angles'] = np.linspace(0, 180, params['n_rotations'], endpoint=False)

    @staticmethod
    def params_test(params):
        assert params['length_range'][0] < params['length_range'][1]
        assert params['length_range'][0] >= 6
        assert params['length_range'][1] < params['grid_size'] - 3
        assert params['thickness'] >= 2

    def __init__(self, params):
        super().__init__(params)

    def draw_sample(self, n_lines):
        length_list = np.random.randint(
            self.params['length_range'][0], self.params['length_range'][1], (2 * n_lines,)
        )
        angle_list = np.random.choice(self.params['rotation_angles'], (2 * n_lines,))
        prototype_list = []
        ii = 0
        while len(prototype_list) < n_lines:
            prototype = get_rotated_prototype(self.params['thickness'], length_list[ii], angle_list[ii], 1)
            n_rows, n_cols = prototype.shape
            if n_rows <= self.params['grid_size'] and n_cols <= self.params['grid_size']:
                prototype_list.append(prototype)

            ii += 1

        if len(prototype_list) != n_lines:
            raise Exception('Not enough samples. Consider resetting length_range')

        prototype_shape_list = [
            prototype.shape for prototype in prototype_list
        ]
        location_range_list = [
            (self.params['grid_size'] - shape[0] + 1, self.params['grid_size'] - shape[1] + 1)
            for shape in prototype_shape_list
        ]
        location_list = [
            (np.random.randint(location_range[0]), np.random.randint(location_range[1]))
            for location_range in location_range_list
        ]
        sample = np.zeros((self.params['grid_size'], self.params['grid_size']), dtype=int)
        for ii in range(n_lines):
            row_top = location_list[ii][0]
            row_bottom = row_top + prototype_shape_list[ii][0]
            col_left = location_list[ii][1]
            col_right = col_left + prototype_shape_list[ii][1]
            sample[row_top:row_bottom, col_left:col_right][prototype_list[ii] == 1] = 1

        return sample


def remove_zero_padding(prototype):
    assert np.sum(prototype == 0) + np.sum(prototype == 1) == prototype.size
    row_sum = np.sum(prototype, axis=1)
    col_sum = np.sum(prototype, axis=0)
    prototype = prototype[row_sum > 0][:, col_sum > 0]
    return prototype


def get_rotated_prototype(thickness, length, angle, order):
    prototype = np.zeros((length, length), dtype=int)
    top_index = int(np.floor(length / 2) - np.floor(thickness / 2))
    bottom_index = top_index + thickness
    prototype[top_index:bottom_index] = 1
    prototype = rotate(prototype, angle=angle, order=order, preserve_range=True)
    prototype[prototype > 0] = 1
    prototype = remove_zero_padding(prototype)
    return prototype
