from line_world.data_generator import DataGenerator


params = {
    'grid_size': 64,
    'thickness': 3,
    'length_range': (6, 50),
    'n_rotations': 10
}

data_generator = DataGenerator(params)
sample = data_generator.draw_sample(4)
