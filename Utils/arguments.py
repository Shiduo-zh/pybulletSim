import argparse
from xml.etree.ElementInclude import default_loader
from click import help_option

def get_args():
    parser=argparse.ArgumentParser(description='quadruped robot rl')
    parser.add_argument(
        '--policy-lr',
        default=1e-4,
        help='policy learning rate in ppo algorithm')
    parser.add_argument(
        '--value-lr',
        default=1e-4,
        help='value learning rate'
    )
    parser.add_argument(
        '--horizen',
        default=1000,
        help='horizen in one episode'
    )
    parser.add_argument(
        '--op-epochs',
        default=3,
        help='optimazition epochs'
    )
    parser.add_argument(
        '--discount',
        default=0.99,
        help='discount factor'
    )
    parser.add_argument(
        '--batch-size',
        default=256,
        help='batch size for training'
    )
    parser.add_argument(
        '--num-env-steps',
        default=10e6,
        help='total number of environment steps to train'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1, 
        help='random seed (default: 1)'
    )
    parser.add_argument(
        '--env-type',
        default='thin_obstacle',
        help='the tpye of training environment'
    )
    parser.add_argument(
        '--connection-mode',
        default='visual',
        help='choose whether traning or testing process visible'
    )
    #about trajectory
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in PPO (default: 5)')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=8,
        help='how many training CPU/GPU processes to use')
    
    #about ppo parameters,refers to acktr
    parser.add_argument(
        '--clip',
        type=float,
        default=0.2,
        help='ppo clip paramete'
    )
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs'
    )
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)'
    )
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)'
    )
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='Adam optimizer epsilon (default: 1e-5)'
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)'
    )
    parser.add_argument(
        '--local-rank',
        default=1,
        help='gpu numbers in training'
    )

    #about log and save
    parser.add_argument(
        '--log-interval',
        default=10,
        help='log interval, one log per n steps'
    )
    parser.add_argument(
        '--save-interval',
        default=100,
        type=int,
        help='save interval, one save per n steps'
    )
    parser.add_argument(
        '--log-dir',
        default='./logs',
        help='dir to save agent logs'
    )
    parser.add_argument(
        '--save-dir',
        default='./trained_models'
    )

    args=parser.parse_args()
    return args