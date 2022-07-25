from environs import Env

env = Env()
env.read_env()

# IMAGE_DIRECTORY = env.str('IMAGE_DIRECTORY')
IMG_PATH = env.str('IMG_PATH')