import os

import config


def create_dirs():
    # Creates model evaluation folders
    if os.path.isdir('./model_evaluation'):
        os.makedirs(f'./model_evaluation/{config.TIMESTAMP}/weights', exist_ok=True)
    else:
        os.makedirs(f'./model_evaluation/{config.TIMESTAMP}/weights', exist_ok=True)