import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
import argparse
import time
import datetime
import random
from deap import gp
from pathlib import Path
from apex.parallel import DistributedDataParallel as DDP
from datasets import glue_train_tasks
from models import select_config, select_model
from tokenizers import select_tokenizer
from metrics import compute_glue_metrics, all_glue_select_metrics
from utils import AverageMeter, register_custom_ops, register_custom_ops2,  \
    set_seeds, setup_logger, reduce_tensor, calc_params, soft_cross_entropy, load_pretrain_state_dict, \
    load_multi_task_state_dict, create_optimizer, create_scheduler, create_split_dataset, create_dataset, \
    create_pretrain_dataset, create_multi_task_dataset, save_checkpoint

from flops import get_cand_flops

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

parser = argparse.ArgumentParser()
parser.add_argument('--distributed', action='store_true', help='distributed mode')
parser.add_argument('--gpu_devices', default='0,1,2,3', type=str, help='available gpu devices')
parser.add_argument('--seed', default=42, type=int, help='seed')

parser.add_argument('--lowercase', action='store_true', help='whether to do lowercase')
parser.add_argument('--sce_temp', default=1, type=float, help='temperature for soft cross entropy loss')
parser.add_argument('--hidden_ratio', default=1, type=float, help='ratio for hidden loss')
parser.add_argument('--pred_ratio', default=1, type=float, help='ratio for prediction loss')

parser.add_argument('--task', default='mnli', type=str, help='task name')
parser.add_argument('--data_dir', default='', type=str, help='task dataset directory')
parser.add_argument('--vocab_path', default='', type=str, help='path to pretrained vocabulary file')
parser.add_argument('--merge_path', default='', type=str, help='path to pretrained merge file (for roberta)')
parser.add_argument('--max_seq_len', default=128, type=int, help='max length of input sequences')
parser.add_argument('--max_query_len', default=64, type=int, help='max length of input questions (for squad) or question-answer pairs (for multi-choice tasks)')
parser.add_argument('--trunc_stride', default=32, type=int, help='context truncate stride (for squad)')
parser.add_argument('--n_best_size', default=20, type=int, help='total number of top-n best predictions to generate (for squad)')
parser.add_argument('--max_answer_len', default=30, type=int, help='maximum length of an answer that can be generated (for squad)')
parser.add_argument('--null_score_diff_threshold', default=0, type=float, help='if null_score - best_non_null is greater than the threshold predict null (for squad)')

parser.add_argument('--min_height', default=3, type=int, help='min height of GP tree')
parser.add_argument('--max_height', default=7, type=int, help='max height of GP tree')
parser.add_argument('--min_params', default=13700000, type=float, help='min params to search')
parser.add_argument('--max_params', default=15700000, type=float, help='max params to search')
parser.add_argument('--n_init_samples', default=80, type=int, help='num of initial samples to train action space')
parser.add_argument('--train_interval', default=10, type=int, help='num sample interval to train action space')
parser.add_argument('--n_total_samples', default=1000000, type=int, help='num of total samples to search')

parser.add_argument('--train_ratio4', default=0.9, type=float, help='downstream train ratio for stage 3')
parser.add_argument('--val_ratio3', default=0, type=float, help='pretrain val ratio for stage 3')
parser.add_argument('--val_ratio4', default=0.1, type=float, help='downstream val ratio for stage 3')
parser.add_argument('--batch_size4', default=64, type=int, help='downstream batch size for stage 3')
parser.add_argument('--lr3', default=1e-4, type=float, help='initial pretrain learning rate for stage 3')
parser.add_argument('--lr4', default=4e-4, type=float, help='initial downstream learning rate for stage 3')
parser.add_argument('--optim_type', default='adamw', type=str, help='optimizer type')
parser.add_argument('--sched_type', default='step', type=str, help='lr scheduler type')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='proportion of warmup steps')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max gradient norm')
parser.add_argument('--loss_disp_freq', default=50, type=int, help='loss display frequency')
parser.add_argument('--ind_disp_freq', default=10, type=int, help='top individuals display frequency')
parser.add_argument('--val_freq1', default=1000000, type=int, help='validate frequency for stage 1 and 2')
parser.add_argument('--val_freq2', default=500, type=int, help='validate frequency for stage 3')
parser.add_argument('--save_freq', default=1, type=int, help='checkpoint save frequency for stage 3')
parser.add_argument('--ckpt_keep_num', default=1, type=int, help='max number of checkpoint files to keep for stage 3')

parser.add_argument('--base_model_name', default='mt_mysupernet', type=str, help='base model name')
parser.add_argument('--base_supernet_path', default='', type=str, help='path to stole base supernet dict')
parser.add_argument('--teacher_pretrain_path', default='', type=str, help='path to pretrained teacher state dict for all stages')
parser.add_argument('--teacher_downstream_path2', default='', type=str, help='path to finetuned downstream teacher state dict for stage 3')
parser.add_argument('--student_pretrain_path2', default='', type=str, help='path to pretrained student state dict for stage 3 (entire)')
parser.add_argument('--student_pretrain_path3', default='', type=str, help='path to pretrained student state dict for stage 3 (slice)')
parser.add_argument('--student_downstream_path', default='', type=str, help='path to finetuned downstream student state dict for stage 3')
parser.add_argument('--pretrain_dir2', default='', type=Path, help='directory to train dataset for stage 3')
parser.add_argument('--wiki_dir', default='', type=Path, help='directory to wikipedia dataset')
parser.add_argument('--book_dir', default='', type=Path, help='directory to bookcorpus dataset')
parser.add_argument('--concate_data_dir', default='', type=Path, help='directory to concatenated dataset')
parser.add_argument('--cache_dir', default='./cache', type=str, help='cache directory to save processed dataset')
parser.add_argument('--exp_dir', default='./exp/tmp/', type=str, help='experiment directory')
parser.add_argument('--local_rank', default=0, type=int, help='DDP local rank')
parser.add_argument('--local-rank', default=0, type=int, help='DDP local rank')
parser.add_argument('--world_size', default=1, type=int, help='DDP world size')

parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--max-epochs', type=int, default=40)
parser.add_argument('--select-num', type=int, default=1)
parser.add_argument('--population-num', type=int, default=5)
parser.add_argument('--m_prob', type=float, default=0.1)
parser.add_argument('--crossover-num', type=int, default=25)
parser.add_argument('--mutation-num', type=int, default=25)
parser.add_argument('--flops-limit', type=float, default=330 * 1e10)
parser.add_argument('--max-train-iters', type=int, default=200)
parser.add_argument('--max-test-iters', type=int, default=40)
parser.add_argument('--train-batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--type_blocks', default=3, type=int, help='number of the type')
parser.add_argument('--num_layers', default=12, type=int, help='number layers')
parser.add_argument('--range1', default=0, type=int, help='')
parser.add_argument('--range2', default=0, type=int, help='')
parser.add_argument('--ffn_expr', default='[]', type=str, help='feed-forward network expression')
parser.add_argument('--type_each_block', default=1, type=int, help='')
parser.add_argument('--fixed_dimension', action='store_true', help='')
parser.add_argument('--only2', action='store_true', help='if only 2 blocks work')
parser.add_argument('--proxy_type', default=0, type=int, help='') # 0: PCA  1:para 2:PCA*para
parser.add_argument('--total_epochs', default=10, type=int, help='total epochs')
parser.add_argument('--train_ratio', default=1, type=float, help='ratio of train dataset')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--new_loss', default=0, type=int, help='')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


if not args.teacher_downstream_path2:
    args.teacher_downstream_path2 = [
        './exp/train/bert_base/mnli/best_model.bin',
        './exp/train/bert_base/qqp/best_model.bin',
        './exp/train/bert_base/qnli/best_model.bin',
        './exp/train/bert_base/sst-2/best_model.bin',
        './exp/train/bert_base/cola/best_model.bin',
        './exp/train/bert_base/sts-b/best_model.bin',
        './exp/train/bert_base/mrpc/best_model.bin',
        './exp/train/bert_base/rte/best_model.bin',
    ]

def covariance(jacobs):
    jacob = torch.transpose(jacobs, 0, 1).reshape(jacobs.size(1), -1).cpu().numpy()
    correlations = np.corrcoef(jacob)
    v, _ = np.linalg.eig(correlations)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))

def calc_distil_losses(teacher_outputs, student_outputs, labels, task, is_pretrain, is_calc_pca_cka):
    teacher_pred_logits, teacher_attn_outputs, teacher_ffn_outputs = teacher_outputs
    #logging.info("len(student_outputs)={}".format(len(student_outputs)))
    #student_pred_logits, student_attn_outputs, student_ffn_outputs, _, _ = student_outputs
    student_pred_logits, student_attn_outputs, student_ffn_outputs = student_outputs

    def _replace_attn_mask(attn_output):
        replace_values = torch.zeros_like(attn_output)
        if use_gpu:
            replace_values = replace_values.cuda()
        attn_output = torch.where(attn_output <= -1e2, replace_values, attn_output)
        return attn_output

    attn_loss, ffn_loss = 0, 0
    mse_loss = nn.MSELoss()
    ffn_loss += mse_loss(teacher_ffn_outputs[0], student_ffn_outputs[0])
    cos_sim = []
    for layer_id in range(num_student_layers):
        teacher_layer_id = layer_id * teacher_interval
        #logging.info("teacher_attn_outputs[teacher_layer_id].shape={}, student_attn_outputs[layer_id].shape={}",teacher_attn_outputs[teacher_layer_id].shape, student_attn_outputs[layer_id].shape)
        attn_loss += mse_loss(teacher_attn_outputs[teacher_layer_id], student_attn_outputs[layer_id])
        ffn_loss += mse_loss(teacher_ffn_outputs[teacher_layer_id + 1], student_ffn_outputs[layer_id + 1])


    hidden_loss = attn_loss + ffn_loss
    if is_pretrain:
        return hidden_loss, attn_loss, ffn_loss, None

    #logging.info("task={}, student_pred_logits.shape={}, teacher_pred_logits.shape={}", task, student_pred_logits.shape, teacher_pred_logits.shape)
    if task == 'sts-b':
        pred_loss = mse_loss(student_pred_logits, labels)
    else:
        pred_loss = soft_cross_entropy(student_pred_logits, teacher_pred_logits, args.sce_temp)
    total_loss = args.hidden_ratio * hidden_loss + args.pred_ratio * pred_loss
    return total_loss, attn_loss, ffn_loss, pred_loss, cos_sim

class EvolutionSearcher(object):
    def __init__(self, args):
        self.args = args

        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.flops_limit = args.flops_limit
        self.max_params = args.max_params
        self.min_params = args.min_params

        #load_pretrain_state_dict(args.base_model_name, base_model, args.base_supernet_path, use_gpu, is_finetune=True, is_spos=True)
        #supernet_state_dict = torch.load(
        #    '../Supernet/models/checkpoint-latest.pth.tar')['state_dict']
        #self.model.load_state_dict(supernet_state_dict)

        self.log_dir = args.log_dir
        self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        self.nr_layer = args.num_layers
        self.nr_state = args.type_blocks

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        #info['params'] = self.params
        torch.save(info, self.checkpoint_name)
        logging.info('save checkpoint to {}'.format(self.checkpoint_name))

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']
        #self.params = info['params']

        logging.info('load checkpoint from {}'.format(self.checkpoint_name))
        return True


    def is_legal(self, cand):
        assert isinstance(cand, tuple) and len(cand) == self.nr_layer
      #  for i in range(len(cand)):
       #     cand[i] = 0
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        if 'flops' not in info:
            info['flops'] = base_model.module.flops(list(cand), args.max_seq_len)

        if 'params' not in info:
            info['params'] = base_model.module.params(list(cand))

        if args.local_rank == 0:
            logging.info("{}-{}-flops: {}-params: {}".format(list(cand), base_model.module.get_out_size_list(list(cand)),
                                                             info['flops'], info['params']))

        logging.info("{}-flops: {}-params: {}".format(list(cand), info['flops'], info['params']))


        if info['params'] > self.max_params:
            logging.info('params limit exceed')
            return False

        #if info['params'] < self.min_params:
        #    logging.info('params too low')
        #    return False

        #info['err'] = get_cand_err(self.model, cand, self.args)
        if args.proxy_type == 1:
            info['err'] = info['params']
        else:
            args.params = info['params']
            cand = eval(args.ffn_expr)
            info['err'] = validate2(cand, is_search=True)

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        logging.info('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                if args.only2:
                    modified_cand = tuple(args.type_each_block if x >= args.type_each_block else 0 for x in cand)
                else:
                    modified_cand = cand
                yield modified_cand

    def get_random(self, num):
        logging.info('random select ........')
        cand_iter = self.stack_random_cand(
            lambda: tuple(np.random.randint(args.range1, args.range2+1) for i in range(self.nr_layer)))
        while len(self.candidates) < num:
            cand = next(cand_iter)
            logging.info('cand={}'.format(cand))
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            #logging.info('self.nr_state={}'.format(self.nr_state))
            for i in range(self.nr_layer):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.nr_state)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        logging.info('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('crossover {}/{}'.format(len(res), crossover_num))

        logging.info('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            logging.info('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])

            logging.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                logging.info('No.{} {} Top-1 err = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['err']))
                ops = [i for i in cand]
                logging.info(ops)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

        #self.save_checkpoint()


# Validate function for stage 3
def validate2(ffn_arch, use_dev=False, is_search=False):
    base_model.eval()
    all_fitness = AverageMeter()
    all_task_fitness = {}
    if use_dev:
        data_loader = downstream_dev_loader
    else:
        data_loader = downstream_val_loader

    with torch.no_grad():
        for task, data_loader in zip(glue_train_tasks, data_loader):
            if is_search and task != 'mnli':  # Only use MNLI to search
                continue
            if use_dev and task == 'qqp':  # Do not evaluate QQP when training the supernet with multiple tasks
                continue
            all_task_fitness[task] = AverageMeter()

            for batch_idx, data in enumerate(data_loader):
                if use_gpu:
                    data = [data_.cuda() for data_ in data]
                task_ids, token_ids, segment_ids, position_ids, attn_mask, labels = data
                logging.info("list(ffn_arch)={}".format(list(ffn_arch)))
                outputs, fitness = base_model(task_ids[0].item(), token_ids, segment_ids, position_ids, attn_mask, args.type_blocks, list(ffn_arch), is_spos=True)
                fitness /= args.num_layers
                if args.proxy_type == 2:
                    fitness *= args.params
                if args.distributed:
                    fitness = reduce_tensor(torch.tensor(fitness, dtype=torch.float64).cuda(), args.world_size)
                all_task_fitness[task].update(fitness.item(), data[0].size(0))
                all_fitness.update(fitness.item(), data[0].size(0))
                if batch_idx > 1:
                    break
            all_task_fitness[task] = all_task_fitness[task].avg

        if use_dev:
            return all_fitness.avg, all_task_fitness
        return all_fitness.avg

def create_downstream_dataset_loaders():
    _, _, _, downstream_train_loader = create_multi_task_dataset(
        args.base_model_name, glue_train_tasks, args.data_dir, tokenizer, args.max_seq_len, args.max_query_len,
        args.trunc_stride, args.train_batch_size, args.train_ratio4, args.val_ratio4, use_gpu, args.distributed,
        'train', args.local_rank, args.cache_dir)
    _, _, _, downstream_val_loader = create_multi_task_dataset(
        args.base_model_name, glue_train_tasks, args.data_dir, tokenizer, args.max_seq_len, args.max_query_len,
        args.trunc_stride, args.test_batch_size, args.train_ratio4, args.val_ratio4, use_gpu, args.distributed,
        'val', args.local_rank, args.cache_dir)
    _, _, _, downstream_dev_loader = create_multi_task_dataset(
        args.base_model_name, glue_train_tasks, args.data_dir, tokenizer, args.max_seq_len, args.max_query_len,
        args.trunc_stride, args.test_batch_size, args.train_ratio4, args.val_ratio4, use_gpu, args.distributed,
        'dev', args.local_rank, args.cache_dir)
    return downstream_train_loader, downstream_val_loader, downstream_dev_loader



if __name__ == '__main__':
    use_gpu = False
    if args.gpu_devices and torch.cuda.is_available():
        use_gpu = True
    args.exp_dir = os.path.join(args.exp_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    setup_logger(args.exp_dir)

    if args.local_rank == 0:
        logging.info(args)
        if use_gpu:
            logging.info('Currently using GPU: {}'.format(args.gpu_devices))
        else:
            logging.info('Currently using CPU')
    set_seeds(args.seed, use_gpu)
    expr = eval(args.ffn_expr)
    # logging.info("expr={}".format(expr))
    if expr == []:
        issingle = args.range1 == args.range2
        args.type_blocks = (args.range2 - args.range1 + 1) * args.type_each_block
        args.range1 *= args.type_each_block
        args.range2 = (args.range2 + 1) * args.type_each_block - 1
    else:
        issingle = True
        args.range2 = 0
        args.range1 = 10000
        for e in expr:
            if e != expr[0]:
                issingle = False
            if args.range2 < e:
                args.range2 = e
            if args.range1 > e:
                args.range1 = e

    if use_gpu and args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        logging.info('Training in distributed mode (process {}/{})'.format(args.local_rank + 1, args.world_size))

    # Load model and tokenizer
    base_model_name = args.base_model_name

    if args.local_rank == 0:
        logging.info('args:{}'.format(args))
    issingle = args.range1 == args.range2
    #issingle = True
    base_config = select_config(base_model_name, args.lowercase, issingle)
    #fixed_dimension = True
    base_model = select_model(base_model_name, args.lowercase, args.task, return_hid=args.proxy_type==3, issingle=False, fixed_dimension=args.fixed_dimension)
    tokenizer = select_tokenizer(
        base_model_name, args.lowercase, args.task, args.vocab_path, args.max_seq_len, args.max_query_len, args.merge_path)
    teacher_model_name = 'mt_bert_base'
    teacher_config = select_config(teacher_model_name, args.lowercase, issingle)
    teacher_model = select_model(teacher_model_name, args.lowercase, args.task, return_hid=True,
                                 issingle=issingle)
    teacher_interval = teacher_config.num_layers // base_config.num_layers
    num_student_layers = base_config.num_layers
    if use_gpu:
        teacher_model, base_model= teacher_model.cuda(), base_model.cuda()
        if args.distributed:
            teacher_model = DDP(teacher_model, delay_allreduce=True)
            base_model = DDP(base_model, delay_allreduce=True)
        else:
            teacher_model = nn.DataParallel(teacher_model)
            base_model = nn.DataParallel(base_model)


    # Create pretrain and downstream datasets
    from pathlib import PosixPath

    if args.book_dir != PosixPath('.') and args.concate_data_dir != PosixPath('.'):
        args.all_train_dir = [args.wiki_dir, args.book_dir]
    else:
        args.all_train_dir = [args.wiki_dir]

    downstream_train_loader, downstream_val_loader, downstream_dev_loader = create_downstream_dataset_loaders()
    downstream_num_sched_steps = len(downstream_train_loader) * args.max_epochs
    downstream_num_warmup_steps = int(downstream_num_sched_steps * args.warmup_proportion)


    # Count total training examples
    all_num_data_epochs = []
    total_examples = 0
    for train_dir in args.all_train_dir:
        num_epoch_examples = []
        num_data_epochs = len([file for file in os.listdir(train_dir) if re.match(r'epoch_\d+_metrics.json', file) is not None])
        all_num_data_epochs.append(num_data_epochs)
        for i in range(num_data_epochs):
            metrics_file = train_dir / 'epoch_{}_metrics.json'.format(i)
            if metrics_file.is_file():
                metrics = json.loads(metrics_file.read_text())
                num_epoch_examples.append(metrics['num_training_examples'])
        for epoch in range(args.total_epochs):
            total_examples += int(num_epoch_examples[epoch % len(num_epoch_examples)] * args.train_ratio)

    # Create optimizer and scheduler
    num_sched_steps = total_examples // (args.batch_size * args.world_size)
    num_warmup_steps = int(num_sched_steps * args.warmup_proportion)
    optimizer = create_optimizer(base_model, args.optim_type, args.lr, args.weight_decay, args.momentum)
    scheduler = create_scheduler(optimizer, args.sched_type, num_sched_steps, num_warmup_steps)

    pset = register_custom_ops()

    st_time = time.time()

    searcher = EvolutionSearcher(args)

    searcher.search()

    '''
    search_phase = SearchPhase(
        init_expr, pset, param_list, num_student_layers, args.min_height, args.max_height,
        args.min_params, args.max_params, args.fixed_mat_dir, args.n_init_samples, args.train_interval,
        args.n_total_samples, is_stage2=args.stage2, is_stage3=args.stage3)
    search_phase.run(train_val_function, args.exp_dir, args.ind_disp_freq, local_rank=args.local_rank)
    '''
    if args.local_rank == 0:
        elapsed = round(time.time() - st_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logging.info('Finished, total training time (h:m:s): {}'.format(elapsed))
