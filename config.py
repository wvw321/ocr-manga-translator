from environs import Env

env = Env()
env.read_env()

CRAFT_PATH = env.str('CRAFT_PATH')
FROZEN_EAST_DETECTOR_PATH = env.str('FROZEN_EAST_DETECTOR_PATH')
IMAGE_DIRECTORY = env.str('IMAGE_DIRECTORY')
