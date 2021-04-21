import logging
import os
import numpy as np
import torch
import torch.optim as optim
import tensorboard_logger

from tensorboard_logger import log_value
from utils import read_config
from utils.generators.mixed_len_generator import MixedGenerateData
from utils.scheduler import LearningRate
from utils.loss import chamfer, sup_loss, cosine_similarity
from utils.train_utils import to_onehot
from utils.visualizer import CSGEngine
from models.model import CSGmodel
from tqdm import tqdm

#Config parameters, set logging
config = read_config.Config("2dsup_config.yml")

model_name = config.model_path.format(config.mode)
print(config.config, flush=True)

logger = logging.getLogger(__name__)
if config.debug == False:
    config.write_config("log/configs/{}_config.txt".format(model_name))
    tensorboard_logger.configure("log/tensorboard/{}".format(model_name), flush_secs=5)
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

net = CSGmodel(config.input_size, config.hidden_size, config.mode, config.encoder_drop, 
                        config.dropout, config.canvas_shape, len(generator.unique_draw), num_layers=1)
if torch.cuda.is_available():
    print('Using GPU')
    net.cuda()

if config.preload_model:
    print(config.pretrain_modelpath, "Loaded")
    net.load_state_dict(torch.load(config.pretrain_modelpath))

optimizer = optim.Adam([para for para in net.parameters() if para.requires_grad],
                            weight_decay=config.weight_decay, lr=config.lr)
reduce_plat = LearningRate(optimizer, init_lr=config.lr, lr_dacay_fact=0.2,
                            patience=config.patience, logger=logger)

# Create the output directory.
if not os.path.exists('trained_models'):
    os.makedirs('trained_models')

#Training, Testing of supervised learning
prev_test_loss = 1e20
prev_test_cd = 1e20
prev_test_iou = 0

for epoch in range(config.epochs):

    net.train()
    pbar = tqdm(total=config.train_size, leave=False)
    epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
            str(epoch).zfill(len(str(config.epochs))), config.epochs)

    train_loss = 0.0

    for batch_idx in range(config.train_size // config.batch_size):
        optimizer.zero_grad()
        batch_kloss = 0.0
        
        for k in data_labels_paths.keys():
            # data: length_of_program + 1, batch_size, stack_size, canvas(64, 64)
            # lables: batch_size, pg_len + 1
            # len(generator.unique_draw) = num_draws
            data, labels = next(train_gen_objs[k])
            data = data[:, :, 0:1, :, :]
            one_hot_labels = to_onehot(labels, len(generator.unique_draw))

            one_hot_labels = torch.from_numpy(one_hot_labels).cuda()
            data = torch.from_numpy(data).cuda()
            labels = torch.from_numpy(labels).cuda()

            outputs = net([data, one_hot_labels, k])

            loss_k = (sup_loss(outputs, labels, time_steps=k + 1) / (k + 1)) / types_prog
            loss_k.backward()

            batch_kloss += loss_k.item()

        optimizer.step()
        train_loss += batch_kloss

        pbar.set_description('{} {} Loss: {:f}'.format(epoch_str, 'Train', batch_kloss))
        pbar.update(config.batch_size)
        
        log_value('train_loss_batch', batch_kloss, epoch * (config.train_size // config.batch_size) + batch_idx)

    pbar.close()
    mean_train_loss = train_loss / (config.train_size // config.batch_size)
    print("Epoch {}/{} => train_loss: {}".format(epoch, config.epochs, mean_train_loss))
    log_value('train_loss', mean_train_loss, epoch)

    net.eval()
    pbar = tqdm(total=config.test_size, leave=False)

    test_loss = 0.0
    metrics = {"cos": 0, "iou": 0, "cd": 0}
    IOU = 0
    COS = 0
    CD = 0

    for batch_idx in range(config.test_size // config.batch_size):
        parser = CSGEngine(generator.unique_draw, max_len, config.canvas_shape)
        batch_kloss = 0.0
        for k in data_labels_paths.keys():
            data, labels = next(test_gen_objs[k])
            data = data[:, :, 0:1, :, :]
            one_hot_labels = to_onehot(labels, len(generator.unique_draw))

            with torch.no_grad():
                gt_image = data[-1, :, 0, :, :].astype(dtype=bool)

                one_hot_labels = torch.from_numpy(one_hot_labels).cuda()
                data = torch.from_numpy(data).cuda()
                labels = torch.from_numpy(labels).cuda()

                outputs = net([data, one_hot_labels, k])
                batch_kloss += (sup_loss(outputs, labels, time_steps=k + 1).item() / (k + 1)) / types_prog 
                
                pred_op = net.test([data, one_hot_labels, max_len])
                pred_images, correct_prog, pred_prog = parser.get_final_canvas(pred_op, False, True)
            
            iou = np.sum(np.logical_and(pred_images, gt_image), (1, 2)) / np.sum(np.logical_or(pred_images, gt_image), (1, 2))
            cosine = cosine_similarity(pred_images, gt_image)
            chamfer_dis = chamfer(pred_images, gt_image)
            
            IOU += np.sum(iou)
            COS += np.sum(cosine)
            CD += np.sum(chamfer_dis)

        test_loss += batch_kloss

        pbar.set_description('{} {} Loss: {:f}'.format(epoch_str, 'Test', batch_kloss))
        pbar.update(config.batch_size)

    pbar.close()

    metrics["iou"] = IOU / config.test_size
    metrics["cos"] = COS / config.test_size
    metrics["cd"] = CD / config.test_size

    mean_test_loss = test_loss / (config.test_size // (config.batch_size))

    log_value('test_IOU', metrics["iou"], epoch)
    log_value('test_cosine', metrics["cos"], epoch)
    log_value('test_CD', metrics["cd"], epoch)
    log_value('test_loss', mean_test_loss, epoch)

    reduce_plat.reduce_on_plateu(metrics["cd"])

    logger.info("Epoch {}/{} => test_loss: {:f}, IOU: {:f}%, CD: {:f}, cosine: {:f}".format(epoch, config.epochs, 
                    mean_test_loss, 100.0*metrics["iou"], metrics["cd"], metrics["cos"]))
    print("Epoch {}/{} => test_loss: {:f}, IOU: {:f}%, CD: {:f}, cosine: {:f}".format(epoch, config.epochs, 
                    mean_test_loss, 100.0*metrics["iou"], metrics["cd"], metrics["cos"]))

    if prev_test_cd > metrics["cd"]:
        logger.info("Saving the Model weights based on CD: %f"%(metrics["cd"]))
        print("Saving the Model weights based on CD: %f"%(metrics["cd"]), flush=True)
        torch.save(net.state_dict(), "trained_models/{}.pth".format(model_name))
        prev_test_cd = metrics["cd"]