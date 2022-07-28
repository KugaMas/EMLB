import os
import time
import argparse
import numpy as np


from options.options import set_inference_options
from scripts.denoisors import Denoisor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment of EMLB benchmark')
    parser.add_argument('-i', '--input_path', type=str, default='datasets', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str, default='results', help='path to output denoising result')
    parser.add_argument('-d', '--denoisors', type=list, default=['mlpf', ], help='choose denoisors')
    parser.add_argument("-p", "--params", type=float, default=[[True, True], ], nargs='+', help="specified parameters")
    args = set_inference_options(parser)
    assert len(args.denoisors) == len(args.params), "The number of denoisors must match parameters"
    
    for i in range(len(args.denoisors)):
        model = Denoisor(args.denoisors[i], args.params[i]) # load model
        for fdata in args.database:
            for fpath in fdata.file_paths:
                fdata.file_paths.set_description("Now implementing %10s to inferenc on %20s" % (model.name, os.path.basename(fpath).split('.')[0]))
                time.sleep(2)
