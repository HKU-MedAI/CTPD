from dataset.mimic3.mimic3benchmark.readers import InHospitalMortalityReader, DecompensationReader, LengthOfStayReader, PhenotypingReader, ReadmissionReader, MultitaskReader
from dataset.mimic3.mimic3_model.preprocessing import Discretizer, Normalizer

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Script for creating a normalizer state - a file which stores the '
                                                 'means and standard deviations of columns of the output of a '
                                                 'discretizer, which are later used to standardize the input of '
                                                 'neural models.')
    parser.add_argument('--task', type=str, default='pheno',
                        choices=['ihm', 'readm', 'pheno', 'decomp', 'los', 'multi'])
    parser.add_argument('--timestep', type=float, default=1.0,
                        help="Rate of the re-sampling to discretize time-series.")
    parser.add_argument('--impute_strategy', type=str, default='zero',
                        choices=['zero', 'next', 'previous', 'normal_value'],
                        help='Strategy for imputing missing values.')
    parser.add_argument('--start_time', type=str, default='zero', choices=['zero', 'relative'],
                        help='Specifies the start time of discretization. Zero means to use the beginning of '
                             'the ICU stay. Relative means to use the time of the first ICU event')
    parser.add_argument('--store_masks', dest='store_masks', action='store_true',
                        help='Store masks that specify observed/imputed values.')
    parser.add_argument('--no-masks', dest='store_masks', action='store_false',
                        help='Do not store that specify specifying observed/imputed values.')
    parser.add_argument('--n_samples', type=int, default=-1, help='How many samples to use to estimates means and '
                        'standard deviations. Set -1 to use all training samples.')
    parser.add_argument('--output_dir', type=str, help='Directory where the output file will be saved.',
                        default=os.path.dirname(__file__))
    parser.add_argument('--data', type=str, 
                        default=os.path.join(os.path.dirname(__file__), '../data/'),
                        help='Path to the task data.')
    parser.set_defaults(store_masks=True)

    args = parser.parse_args()
    print(args)

    # create the reader
    reader = None
    args.data = os.path.join(args.data, args.task)
    dataset_dir = os.path.join(args.data, 'train')
    if args.task == 'ihm':
        reader = InHospitalMortalityReader(dataset_dir=dataset_dir, period_length=48.0)
    if args.task == 'decomp':
        reader = DecompensationReader(dataset_dir=dataset_dir)
    if args.task == 'los':
        reader = LengthOfStayReader(dataset_dir=dataset_dir)
    if args.task == 'pheno':
        reader = PhenotypingReader(dataset_dir=dataset_dir)
    if args.task == 'multi':
        reader = MultitaskReader(dataset_dir=dataset_dir)
    if args.task == 'readm':
        reader = ReadmissionReader(dataset_dir=dataset_dir, period_length=48.0)

    # create the discretizer
    discretizer = Discretizer(timestep=args.timestep,
                              store_masks=args.store_masks,
                              impute_strategy=args.impute_strategy,
                              start_time=args.start_time)
    discretizer_header = reader.read_example(0)['header']
    continuous_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    # create the normalizer
    normalizer = Normalizer(fields=continuous_channels)

    # read all examples and store the state of the normalizer
    n_samples = args.n_samples
    if n_samples == -1:
        n_samples = reader.get_number_of_examples()

    for i in range(n_samples):
        if i % 1000 == 0:
            print('Processed {} / {} samples'.format(i, n_samples), end='\r')
        ret = reader.read_example(i)
        data, new_header = discretizer.transform(ret['X'], end=ret['t'])
        normalizer._feed_data(data)
    print('\n')

    # all dashes (-) were colons(:)
    output_dir = os.path.join(args.output_dir, args.task)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = '{}_ts-{:.2f}_impute-{}_start-{}_masks-{}_n-{}.normalizer'.format(
        args.task, args.timestep, args.impute_strategy, args.start_time, args.store_masks, n_samples)
    file_name = os.path.join(output_dir, file_name)
    print('Saving the state in {} ...'.format(file_name))
    normalizer._save_params(file_name)


if __name__ == '__main__':
    main()
