import os
import tqdm
import argparse
import numpy as np


from options.options import options_initialization


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment of EMLB benchmark')
    parser.add_argument('-i', '--input_path', type=str, default='datasets', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str, default='results', help='path to output denoising result')
    parser.add_argument('-d', '--denoisor', type=list, default=['knoise', 'ynoise'], help='choose denoisors')
    parser.add_argument("-p", "--params", type=float, default=[[], []], nargs='+', help="specified parameters")
    args = options_initialization(parser)

    breakpoint()
