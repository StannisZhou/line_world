from line_world.params import generate_image_templates

kernel_size = 4
thickness = 1
length = 3
n_rotations = 15
image_templates = generate_image_templates(kernel_size, thickness, length, n_rotations)
image_templates = image_templates.to_dense().numpy()
