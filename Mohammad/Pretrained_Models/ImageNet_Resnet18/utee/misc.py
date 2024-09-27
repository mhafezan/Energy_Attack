import cv2
import os
import shutil
import pickle as pkl
import time
import numpy as np
import hashlib
import torch

from IPython import embed

class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        self.init('/tmp', 'tmp.log')
        self._logger.info(str_info)
logger = Logger()

print = logger.info
def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)

def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v

def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def auto_select_gpu(mem_bound=500, utility_bound=0, gpus=(0, 1, 2, 3, 4, 5, 6, 7), num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVCIES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5): # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert(len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

        if len(ideal_gpus) < num_gpu:
            print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')

    print("Setting GPU: {}".format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))


def load_lmdb(lmdb_file, n_records=None):
    import lmdb
    import numpy as np
    lmdb_file = expand_user(lmdb_file)
    if os.path.exists(lmdb_file):
        data = []
        env = lmdb.open(lmdb_file, readonly=True, max_readers=512)
        with env.begin() as txn:
            cursor = txn.cursor()
            begin_st = time.time()
            print("Loading lmdb file {} into memory".format(lmdb_file))
            for key, value in cursor:
                _, target, _ = key.decode('ascii').split(':')
                target = int(target)
                img = cv2.imdecode(np.fromstring(value, np.uint8), cv2.IMREAD_COLOR)
                data.append((img, target))
                if n_records is not None and len(data) >= n_records:
                    break
        env.close()
        print("=> Done ({:.4f} s)".format(time.time() - begin_st))
        return data
    else:
        print("Not found lmdb file".format(lmdb_file))

def str2img(str_b):
    return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_COLOR)

def img2str(img):
    return cv2.imencode('.jpg', img)[1].tostring()

def md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()

def eval_model_resnet(model, ds, n_sample=None, ngpu=1, is_imagenet=False):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        def forward(self, input):
            input.data.div_(255.)
            input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
            input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
            input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
            return self.model(input)

    correct1, correct5 = 0, 0
    n_passed = 0
    L1_L22_zeros_total = [0] * 22
    L1_L22_sizes_total = [0] * 22
    Net_zeros_total, Net_sizes_total = [0] * 2 # mohammad_sparsity_calculation
    
    if is_imagenet:
        model = ModelWrapper(model)
    
    model = model.eval()
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda()

    n_sample = len(ds)

    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        n_passed += len(data)

        if torch.cuda.is_available():
            data =  Variable(torch.FloatTensor(data)).cuda()
        else:
            data =  Variable(torch.FloatTensor(data)).cpu()

        indx_target = torch.LongTensor(target)
        
        output, L1_L22_zeros, L1_L22_sizes = model(data) # mohammad_sparsity_calculation
        
        # Sparsity calculations zone begin (mohammad_sparsity_calculation)
        L1_L22_zeros_total = [xx + yy for xx, yy in zip(L1_L22_zeros_total, L1_L22_zeros)]
            
        L1_L22_sizes_total = [xx + yy for xx, yy in zip(L1_L22_sizes_total, L1_L22_sizes)]
        
        Net_zeros_total += sum(L1_L22_zeros)
            
        Net_sizes_total += sum(L1_L22_sizes)
        # Sparsity calculations zone end (mohammad_sparsity_calculation)

        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum()

        if idx >= n_sample - 1:
            break

    # Calculating Sparsity Statistics for each layer (mohammad_sparsity_calculation)
    SR_L1   = (L1_L22_zeros_total[0]/L1_L22_sizes_total[0])
    SR_L2   = (L1_L22_zeros_total[1]/L1_L22_sizes_total[1])
    SR_L3   = (L1_L22_zeros_total[2]/L1_L22_sizes_total[2])
    SR_L4   = (L1_L22_zeros_total[3]/L1_L22_sizes_total[3])
    SR_L5   = (L1_L22_zeros_total[4]/L1_L22_sizes_total[4])
    SR_L6   = (L1_L22_zeros_total[5]/L1_L22_sizes_total[5])
    SR_L7   = (L1_L22_zeros_total[6]/L1_L22_sizes_total[6])
    SR_L8   = (L1_L22_zeros_total[7]/L1_L22_sizes_total[7])
    SR_L9   = (L1_L22_zeros_total[8]/L1_L22_sizes_total[8])
    SR_L10  = (L1_L22_zeros_total[9]/L1_L22_sizes_total[9])
    SR_L11  = (L1_L22_zeros_total[10]/L1_L22_sizes_total[10])
    SR_L12  = (L1_L22_zeros_total[11]/L1_L22_sizes_total[11])
    SR_L13  = (L1_L22_zeros_total[12]/L1_L22_sizes_total[12])
    SR_L14  = (L1_L22_zeros_total[13]/L1_L22_sizes_total[13])
    SR_L15  = (L1_L22_zeros_total[14]/L1_L22_sizes_total[14])
    SR_L16  = (L1_L22_zeros_total[15]/L1_L22_sizes_total[15])
    SR_L17  = (L1_L22_zeros_total[16]/L1_L22_sizes_total[16])
    SR_L18  = (L1_L22_zeros_total[17]/L1_L22_sizes_total[17])
    SR_L19  = (L1_L22_zeros_total[18]/L1_L22_sizes_total[18])
    SR_L20  = (L1_L22_zeros_total[19]/L1_L22_sizes_total[19])
    SR_L21  = (L1_L22_zeros_total[20]/L1_L22_sizes_total[20])
    SR_L22  = (L1_L22_zeros_total[21]/L1_L22_sizes_total[21])
    SR_Net  = (Net_zeros_total/Net_sizes_total)

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5, SR_L1, SR_L2, SR_L3, SR_L4, SR_L5, SR_L6, SR_L7, SR_L8, SR_L9, SR_L10, SR_L11, SR_L12, SR_L13, SR_L14, SR_L15, SR_L16, SR_L17, SR_L18,\
                       SR_L19, SR_L20, SR_L21, SR_L22, SR_Net

def eval_model_vgg(model, ds, n_sample=None, ngpu=1, is_imagenet=False):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        def forward(self, input):
            input.data.div_(255.)
            input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
            input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
            input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
            return self.model(input)

    correct1, correct5 = 0, 0
    n_passed = 0
    L1_zeros_total, L1_size_total, L2_zeros_total, L2_size_total, L3_zeros_total, L3_size_total, L4_zeros_total, L4_size_total, L5_zeros_total, L5_size_total,\
    L6_zeros_total, L6_size_total, L7_zeros_total, L7_size_total, L8_zeros_total, L8_size_total, L9_zeros_total, L9_size_total, L10_zeros_total, L10_size_total,\
    L11_zeros_total, L11_size_total, L12_zeros_total, L12_size_total, L13_zeros_total, L13_size_total, L14_zeros_total, L14_size_total,\
    L15_zeros_total, L15_size_total, L16_zeros_total, L16_size_total, L17_zeros_total, L17_size_total, L18_zeros_total, L18_size_total,\
    Net_zeros_total, Net_size_total = [0]*38 # mohammad_sparsity
    
    if is_imagenet:
        model = ModelWrapper(model)
    
    model = model.eval()
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda()

    n_sample = len(ds) if n_sample is None else n_sample

    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        n_passed += len(data)

        if torch.cuda.is_available():
            data =  Variable(torch.FloatTensor(data)).cuda()
        else:
            data =  Variable(torch.FloatTensor(data)).cpu()

        indx_target = torch.LongTensor(target)
        
        output, L1_zeros, L1_size, L2_zeros, L2_size, L3_zeros, L3_size, L4_zeros, L4_size, L5_zeros, L5_size, L6_zeros, L6_size, L7_zeros, L7_size,\
        L8_zeros, L8_size, L9_zeros, L9_size, L10_zeros, L10_size, L11_zeros, L11_size, L12_zeros, L12_size, L13_zeros, L13_size, L14_zeros, L14_size,\
        L15_zeros, L15_size, L16_zeros, L16_size, L17_zeros, L17_size, L18_zeros, L18_size = model(data)
        
        # Sparsity calculations zone begin (mohammad_sparsity_calculation)
        L1_zeros_total  += L1_zeros
        L1_size_total   += L1_size
        L2_zeros_total  += L2_zeros
        L2_size_total   += L2_size
        L3_zeros_total  += L3_zeros
        L3_size_total   += L3_size
        L4_zeros_total  += L4_zeros
        L4_size_total   += L4_size
        L5_zeros_total  += L5_zeros
        L5_size_total   += L5_size
        L6_zeros_total  += L6_zeros
        L6_size_total   += L6_size
        L7_zeros_total  += L7_zeros
        L7_size_total   += L7_size
        L8_zeros_total  += L8_zeros
        L8_size_total   += L8_size
        L9_zeros_total  += L9_zeros
        L9_size_total   += L9_size
        L10_zeros_total += L10_zeros
        L10_size_total  += L10_size
        L11_zeros_total += L11_zeros
        L11_size_total  += L11_size
        L12_zeros_total += L12_zeros
        L12_size_total  += L12_size
        L13_zeros_total += L13_zeros
        L13_size_total  += L13_size
        L14_zeros_total += L14_zeros
        L14_size_total  += L14_size
        L15_zeros_total += L15_zeros
        L15_size_total  += L15_size
        L16_zeros_total += L16_zeros
        L16_size_total  += L16_size
        L17_zeros_total += L17_zeros
        L17_size_total  += L17_size
        L18_zeros_total += L18_zeros
        L18_size_total  += L18_size
        Net_zeros_total += (L1_zeros + L2_zeros + L3_zeros + L4_zeros + L5_zeros + L6_zeros + L7_zeros + L8_zeros + L9_zeros + L10_zeros +\
                            L11_zeros + L12_zeros + L13_zeros + L14_zeros + L15_zeros + L16_zeros + L17_zeros + L18_zeros)
        Net_size_total  += (L1_size + L2_size + L3_size + L4_size + L5_size + L6_size + L7_size + L8_size + L9_size +\
                            L10_size + L11_size + L12_size + L13_size + L14_size + L15_size + L16_size + L17_size + L18_size)
        # Sparsity calculations zone end (mohammad_sparsity_calculation)

        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum()

        if idx >= n_sample - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed

    # Calculating Sparsity Statistics for each layer (mohammad_sparsity_calculation)
    SR_L1   = (L1_zeros_total/L1_size_total)
    SR_L2   = (L2_zeros_total/L2_size_total)
    SR_L3   = (L3_zeros_total/L3_size_total)
    SR_L4   = (L4_zeros_total/L4_size_total)
    SR_L5   = (L5_zeros_total/L5_size_total)
    SR_L6   = (L6_zeros_total/L6_size_total)
    SR_L7   = (L7_zeros_total/L7_size_total)
    SR_L8   = (L8_zeros_total/L8_size_total)
    SR_L9   = (L9_zeros_total/L9_size_total)
    SR_L10  = (L10_zeros_total/L10_size_total)
    SR_L11  = (L11_zeros_total/L11_size_total)
    SR_L12  = (L12_zeros_total/L12_size_total)
    SR_L13  = (L13_zeros_total/L13_size_total)
    SR_L14  = (L14_zeros_total/L14_size_total)
    SR_L15  = (L15_zeros_total/L15_size_total)
    SR_L16  = (L16_zeros_total/L16_size_total)
    SR_L17  = (L17_zeros_total/L17_size_total)
    SR_L18  = (L18_zeros_total/L18_size_total)
    SR_Net  = (Net_zeros_total/Net_size_total)

    return acc1, acc5, SR_L1, SR_L2, SR_L3, SR_L4, SR_L5, SR_L6, SR_L7, SR_L8, SR_L9, SR_L10,\
           SR_L11, SR_L12, SR_L13, SR_L14, SR_L15, SR_L16, SR_L17, SR_L18, SR_Net

def load_state_dict(model, model_urls, model_root):
    from torch.utils import model_zoo
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict() # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = model_zoo.load_url(model_urls, model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            print(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    missing = set(own_state.keys()) - set(state_dict.keys())
    no_use = set(state_dict.keys()) - set(own_state.keys())
    if len(no_use) > 0:
        raise KeyError('some keys are not used: "{}"'.format(no_use))

