import logging
import numpy as np
import torch
import torch.optim as optim
import tensorboard_logger

from tensorboard_logger import log_value
from utils import read_config
from utils.generators.mixed_len_generator import MixedGenerateData
#from Model.model import CSGmodel

#Config parameters, set logging
config = read_config.Config("2dsup_config.yml")

model_name = config.model_path.format(config.mode)
print(config.config, flush=True)

if config.debug == False:
    config.write_config("log/configs/{}_config.txt".format(model_name))
    tensorboard_logger.configure("log/tensorboard/{}".format(model_name), flush_secs=5)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    file_handler = logging.FileHandler('log/logger/{}.log'.format(model_name), mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(config.config)

#Data load, processing
data_labels_paths = {
    3: "data/synthetic/one_op/expressions.txt",
    5: "data/synthetic/two_ops/expressions.txt",
    7: "data/synthetic/three_ops/expressions.txt"
}

proportion = config.proportion
dataset_sizes = {
    3: [proportion * 250, proportion * 50],
    5: [proportion * 1000, proportion * 100],
    7: [proportion * 1500, proportion * 200]
}

generator = MixedGenerateData(
    data_labels_paths=data_labels_paths,
    batch_size=config.batch_size,
    canvas_shape=config.canvas_shape)

max_len = max(data_labels_paths.keys())
types_prog = len(dataset_sizes)
train_gen_objs = {}
test_gen_objs = {}
config.train_size = sum(dataset_sizes[k][0] for k in dataset_sizes.keys())
config.test_size = sum(dataset_sizes[k][1] for k in dataset_sizes.keys())
total_importance = sum(k for k in dataset_sizes.keys())
    
for k in data_labels_paths.keys():
    test_batch_size = int(config.batch_size * dataset_sizes[k][1] / config.test_size)
    train_batch_size = config.batch_size // types_prog
    train_gen_objs[k] = generator.get_train_data(
        train_batch_size, k,
        num_train_images=dataset_sizes[k][0],
        jitter_program=True)
    test_gen_objs[k] = generator.get_test_data(
        test_batch_size, k,
        num_train_images=dataset_sizes[k][0],
        num_test_images=dataset_sizes[k][1],
        jitter_program=True)
data, labels = next(train_gen_objs[5])
# data: length_of_program + 1, batch_size, stack_size, canvas(64, 64)
# lables: batch_size, length_of_program + 1


'''
net = CSGmodel().cuda()
if torch.cuda.is_available():
    print('GPU available')
    net.cuda()

if config.preload_model:
    print(config.pretrain_modelpath, "Loaded")
    net.load_state_dict(torch.load(config.pretrain_modelpath))
'''