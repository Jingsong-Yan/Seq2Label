import argparse
import logging
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from data_processing.dataset import MyDataSet, MyDataloader
# from evaluate import *
from model.bart_label_generation import BartForLabelGeneration
from model.t5_label_generation import T5ForLabelGeneration


def run(config):
    # get dataset
    dataset = MyDataSet(os.path.join(config['data_path'], config['dataset']))

    train_data = dataset.get_data(config['mode'], 'train')

    valid_data = dataset.get_data(config['mode'], 'valid')
    # get dataloader
    dataloader = MyDataloader(
        config, batch_size=config['train_batch_size'], dataset=dataset)
    train_dataloader = dataloader.get_dataloader(train_data)
    dataloader.batch_size = config['valid_batch_size']
    valid_dataloader = dataloader.get_dataloader(valid_data)

    # model
    config['label_size'] = dataset.label_nums + 3

    if config['model'] == 'bart':
        model = BartForLabelGeneration.from_pretrained(config['pretrained_path'],
                                                       my_config=config,
                                                       ignore_mismatched_sizes=True)
    elif config['model'] == 't5':
        model = T5ForLabelGeneration.from_pretrained(config['pretrained_path'],
                                                     my_config=config,
                                                     ignore_mismatched_sizes=True)
    else:
        raise RuntimeError("No model was selected!")

    model.cuda()

    logging.info(config)
    optimizer = AdamW(model.parameters(), config['lr'])
    total_steps = len(train_dataloader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    max_micro = 0
    max_macro = 0
    # early_stop = 0
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            input_ids, attention_mask, dec, dec_mask, labels, target, negative_sample = batch
            if negative_sample is not None:
                negative_sample = negative_sample.cuda()
            out = model(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                decoder_input_ids=dec.cuda(),
                decoder_attention_mask=dec_mask.cuda(),
                labels=labels.cuda(),
                negative_sample=negative_sample
            )
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        total_loss /= len(train_dataloader)

        logging.info("epoch:" + str(epoch) + ", loss:" + str(total_loss))
        print("epoch:" + str(epoch) + ", loss:" + str(total_loss))
        with torch.no_grad():
            result = validation(valid_dataloader, model, dataset.label_nums,
                                config[config['dataset']]['label_max_length'],
                                dataset.id2label, dataset.parent, epoch)

        if result['macro'] > max_macro:
            max_macro = result['macro']
            torch.save(model, os.path.join(config['save_model_path'], config['name'] + '.pth'))
            logging.info("epoch:" + str(epoch) +
                         ', save model with best macro_f1:' + str(max_macro))
        if result['micro'] > max_micro:
            max_micro = result['micro']
            torch.save(model, os.path.join(config['save_model_path'], config['name'] + '.pth'))
            logging.info("epoch:" + str(epoch) +
                         ', save model with best micro_f1:' + str(max_micro))


def validation(dataloader, model, num_labels, maxlength, id2label, parent, epoch=None):
    model.eval()

    pred_positive = np.zeros(num_labels)
    target_positive = np.zeros(num_labels)
    right_positive = np.zeros(num_labels)

    candidate = None
    all_target = None
    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids, attention_mask, dec, dec_mask, labels, target, negative_sample = batch
        target = target.cpu().numpy()
        pred_id = model.generate(input_ids.cuda(), decoder_start_token_id=0, bos_token_id=2, max_length=maxlength,
                                 num_beams=1)
        pred_id = pred_id.cpu().numpy()[:, 1:]
        pred_label = create_labels(pred_id, num_labels)

        if candidate is None:
            candidate = pred_label
            all_target = target
        else:
            candidate = np.vstack((candidate, pred_label))
            all_target = np.vstack((all_target, target))
        # 我的实现
        pred_positive = pred_positive + np.sum(pred_label, axis=0)
        target_positive = target_positive + np.sum(target, axis=0)
        right_positive = right_positive + np.sum(np.array((pred_label == 1) & (target == 1), dtype=int), axis=0)
    metrics = get_metrics(pred_positive, target_positive, right_positive, id2label)
    sk_micro_f1 = f1_score(all_target, candidate, average='micro')
    sk_macro_f1 = f1_score(all_target, candidate, average='macro')
    logging.info("epoch:" + str(epoch) + ", metrics:" + str(metrics))
    logging.info("epoch:" + str(epoch) + ", sk_metrics:" +
                 str({"sk_micro_f1": sk_micro_f1, "sk_macro_f1:": sk_macro_f1}))
    return {
        'micro': sk_micro_f1,
        'macro': sk_macro_f1
    }


def get_metrics(pred_positive, target_positive, right_positive, id2label):
    precision = np.divide(right_positive, pred_positive, where=(right_positive != 0) & (pred_positive != 0))
    recall = np.divide(right_positive, target_positive, where=(right_positive != 0) & (target_positive != 0))
    micro_precision = np.sum(right_positive) / np.sum(pred_positive)
    micro_recall = np.sum(right_positive) / np.sum(target_positive)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0.0 else 0.0
    macro_all_f1 = 2 * np.divide(precision * recall, precision + recall, where=(precision + recall != 0))
    for i in range(len(macro_all_f1)):
        if macro_all_f1[i] > 1:
            logging.info('right_positive: ' + str(right_positive[i]) + ' target_positive: ' + str(
                right_positive[i]) + ' pred_positive: ' + str(pred_positive[i]) + ' precision: ' + str(
                precision[i]) + ' recall: ' + str(recall[i]))
            macro_all_f1[i] = 0.0
    macro_f1 = np.average(macro_all_f1)
    return {
        'micro': micro_f1,
        'macro': macro_f1
    }


def test(config):
    config['negative_sample'] = False
    dataset = MyDataSet(os.path.join(config['data_path'], config['dataset']))
    test_data = dataset.get_data(config['mode'], 'test')
    dataloader = MyDataloader(
        config, batch_size=config['valid_batch_size'], dataset=dataset)
    test_dataloader = dataloader.get_dataloader(test_data)
    model = torch.load(os.path.join(config['save_model_path'], config['name'] + '.pth')).cuda()
    logging.info(20 * '*' + "test" + 20 * '*')
    validation(test_dataloader, model, dataset.label_nums, config[config['dataset']]['label_max_length'],
               dataset.id2label, dataset.parent)


def create_labels(inputs, class_nums):
    outputs = []
    for example in inputs:
        temp = np.zeros(class_nums)
        for label in example:
            # end
            if label == 2:
                break
            if label < 2 or label >= class_nums + 3:
                continue
            temp[label - 3] = 1
        outputs.append(temp)
    return np.array(outputs)


def get_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    config = get_config('config.yaml')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='bart', type=str)
    parser.add_argument("--gpu_id", default=1, type=int)
    parser.add_argument("--dataset", default='wos', type=str)
    parser.add_argument("--mode", default='bfs', type=str,
                        help='[random, dfs, bfs, random_each_epoch]')
    parser.add_argument(
        "--data_path", default='', type=str)
    parser.add_argument("--log_path", default='./log', type=str)
    parser.add_argument("--name", default='exam', type=str)
    parser.add_argument("--reversed", default=0, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--save_model_path", default='./outputs/', type=str)
    parser.add_argument("--train", default=1, type=int)
    parser.add_argument("--test", default=1, type=int)
    parser.add_argument("--negative_sample", default=0, type=int)
    parser.add_argument("--random_negative_sample", default=0, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--beam", default=1, type=int)
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--valid_batch_size", default=64, type=int)
    # parser.add_argument("--epsilon", default=0, type=float)

    args = vars(parser.parse_args())
    config = {**config, **args}

    if not os.path.exists(config['log_path']):
        os.makedirs(config['log_path'])
    if not os.path.exists(config['save_model_path']):
        os.makedirs(config['save_model_path'])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(filename)s : %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %A %H:%M:%S",
        filename=os.path.join(config['log_path'], config['name'] + '.log'),
        filemode='a'
    )
    config['epsilon'] = config[config['dataset']]['epsilon']

    if config['train']:
        run(config)
    if config['test']:
        test(config)


if __name__ == '__main__':
    main()
