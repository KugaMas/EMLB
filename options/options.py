import os
import yaml


def dataset_information(file_path='options/dataset_info.yaml'):
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def options_initialization(parser):
    """
    Other customized settings
    """

    """ Fundamental information settings """
    parser.add_argument('--output_file_type', type=str, default='aedat4', help='output file type')

    """ Data preprocess settings """
    parser.add_argument('--use_polarity', type=bool, default=True)
    parser.add_argument('--remove_hotpixel', type=bool, default=True)

    args = parser.parse_args()
    assert len(args.denoisor) == len(args.params), "The number of denoisors must match parameters"

    """ Load parameters preparation """
    args.abs_path    = os.getcwd()
    args.data_info   = dataset_information()
    args.input_path  = os.path.join(args.abs_path, args.input_path)
    args.output_path = os.path.join(args.abs_path, args.output_path)

    return args
