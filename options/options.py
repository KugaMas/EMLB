from tqdm import tqdm
from distutils.command.config import config
import os
import glob
import yaml

class Dataset():
    def __init__(self, path, size, fclass, use_aps):
        self.idx     = 0
        self.size    = size
        self.fclass  = fclass
        self.use_aps = use_aps
        
        # load all file paths
        self.file_paths = [file for folder in fclass for file in glob.glob(f"{path}/{folder}/*")]
        self.file_paths.sort()
        self.file_paths = tqdm(self.file_paths)
        

class Database():
    def __init__(self, args, save_path='options/dataset_info.yaml'):
        self.path, self.loader = args.input_path, dict()
        for fname in os.listdir(self.path):
            self.loader[fname] = os.listdir(f"{self.path}/{fname}")
        self.iterator = iter(self.loader.keys())

        # read config data
        with open(save_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # add information
        for fname, flist in self.loader.items():
            if fname in self.config.keys(): continue
            self.config[fname] = dict()
            self.config[fname]['size']  = (-1, -1)
            self.config[fname]['frame'] = False
            self.config[fname]['class'] = flist

        # write config data
        with open(save_path, "w") as f:
            yaml.dump(self.config, f, encoding='utf-8', allow_unicode=True)

    def __iter__(self):
        return self

    def __next__(self):
        _fname = next(self.iterator)
        _fpath = f"{self.path}/{_fname}"

        _size  = self.config[_fname]['size']
        _frame = self.config[_fname]['frame']
        _class = self.config[_fname]['class']
        
        return Dataset(_fpath, _size, _class, _frame)


def set_inference_options(parser):
    """ Fundamental Information Settings """
    parser.add_argument('--output_file_type', type=str, default='aedat4', help='output file type')

    """ Data Preprocess Settings """
    parser.add_argument('--use_polarity', type=bool, default=True)
    parser.add_argument('--excl_hotpixel', type=bool, default=True)

    """ Parser """
    args = parser.parse_args()

    """ Load Parameters Preparation """
    args.abs_path    = os.getcwd()
    args.input_path  = os.path.join(args.abs_path, args.input_path)
    args.output_path = os.path.join(args.abs_path, args.output_path)

    """ Iteratable Dataset Reader """
    args.database = Database(args)

    return args
