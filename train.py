import argparse
import datetime
import math
import os
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from shepardmetzler import ShepardMetzler, Scene, transform_viewpoint
from model import GQN

def arrange_data(x_data, v_data, seed=None):
    random.seed(seed)
    batch_size, m, *_ = x_data.size()

    # Sample random number of views
    n_views = random.randint(2, m-1)

    indices = torch.randperm(m)
    representation_idx, query_idx = indices[:n_views], indices[n_views]

    x, v = x_data[:, representation_idx], v_data[:, representation_idx]
    x_q, v_q = x_data[:, query_idx], v_data[:, query_idx]
    
    return x, v, x_q, v_q


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
    parser.add_argument('--gradient_steps', type=int, default=2*(10**6), help='number of gradient steps to run (default: 2 million)')
    parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
    parser.add_argument('--train_data_dir', type=str, help='location of training data', \
                        default="/workspace/dataset/shepard_metzler_7_parts-torch/train")
    parser.add_argument('--test_data_dir', type=str, help='location of test data', \
                        default="/workspace/dataset/shepard_metzler_7_parts-torch/test")
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='/workspace/logs')
    parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=100)
    parser.add_argument('--save_interval', type=int, help='interval number of steps for saveing models', default=100000)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # data directory
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    
    # number of workers to load data
    num_workers = args.workers

    # for logging
    log_interval_num = args.log_interval_num
    save_interval_num = args.save_interval
    dir_name = str(datetime.datetime.now())
    log_dir = os.path.join(args.root_dir, dir_name)
    os.mkdir(log_dir)
    os.mkdir(log_dir+'/models')
    os.mkdir(log_dir+'/runs')

    # tensorboardX
    writer = SummaryWriter(log_dir=log_dir+'/runs')

    batch_size = args.batch_size
    gradient_steps = args.gradient_steps

    train_dataset = ShepardMetzler(root_dir=train_data_dir, target_transform=transform_viewpoint)
    test_dataset = ShepardMetzler(root_dir=test_data_dir, target_transform=transform_viewpoint)


    # model settings
    xDim=3
    vDim=7
    rDim=256
    hDim=128
    zDim=64
    L=12
    SCALE = 4 # Scale of image generation process

    # model
    gqn=GQN(xDim,vDim,rDim,hDim,zDim, L, SCALE).to(device)
    gqn = nn.DataParallel(gqn) if args.data_parallel

    # learning rate
    mu_i, mu_f = 5*10**(-4), 5*10**(-5)
    # pixel variance
    sigma_i, sigma_f = 2.0, 0.7
    # initial values
    mu = mu_i
    sigma = sigma_i

    optimizer = torch.optim.Adam(gqn.parameters(), lr=mu, betas=(0.9, 0.999))
    kwargs = {'num_workers':num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=36, shuffle=True, **kwargs)

    x_data_test, v_data_test = next(iter(test_loader))

    # number of gradient steps
    s = 0
    while True:
        for x_data, v_data in tqdm(train_loader):
            x_data = x_data.to(device)
            v_data = v_data.to(device)
            x, v, x_q, v_q = arrange_data(x_data, v_data)
            nll, kl, x_q_rec = gqn(x, v, v_q, x_q, sigma)
            nll = nll.mean()
            kl = kl.mean()
            loss = nll + kl
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('train_nll', nll, s)
            writer.add_scalar('train_kl', kl, s)
            writer.add_scalar('train_loss', loss, s)

            s += 1

            with torch.no_grad():
                # Keep a checkpoint every n steps
                if s % log_interval_num == 0 or s == 1:
                    writer.add_image('train_ground_truth', x_q[:8], s)
                    writer.add_image('train_reconstruction', x_q_rec[:8], s)

                    x_data_test = x_data_test.to(device)
                    v_data_test = v_data_test.to(device)

                    x_test, v_test, x_q_test, v_q_test = arrange_data(x_data_test, v_data_test, seed=0)
                    nll_test, kl_test, x_q_rec_test = gqn(x_test, v_test, v_q_test, x_q_test, sigma)
                    x_q_hat_test = gqn.module.generate(x_test, v_test, v_q_test)

                    nll_test = nll_test.mean()
                    kl_test = kl_test.mean()
                    loss_test = nll_test + kl_test

                    writer.add_scalar('test_nll', nll_test, s)
                    writer.add_scalar('test_kl', kl_test, s)
                    writer.add_scalar('test_loss', loss_test, s)
                    writer.add_image('test_ground_truth', x_q_test[:8], s)
                    writer.add_image('test_reconstruction', x_q_rec_test[:8], s)
                    writer.add_image('test_generation', x_q_hat_test[:8], s)

                if s % save_interval_num == 0:
                    torch.save(gqn.state_dict(), log_dir + "/models/model-{}.pt".format(s))

                if s >= gradient_steps:
                    break

                # Anneal learning rate
                mu = max(mu_f + (mu_i - mu_f)*(1 - s/(1.6 * 10**6)), mu_f)
                for group in optimizer.param_groups:
                    group["lr"] = mu
                # Anneal pixel variance
                sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - s/(2 * 10**5)), sigma_f)

        if s >= gradient_steps:
            torch.save(gqn.state_dict(), log_dir + "/models/model-final.pt")
            break
    writer.close()