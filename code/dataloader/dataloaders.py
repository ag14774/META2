import logging

import torch
from base.base_dataloader import BaseDataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from utils import transforms as mytransforms
from utils.util import get_global_rank, get_world_size

from . import datasets


def nothing_collate_fn(batch):
    return batch


class RefSeqProkaryotaDataLoader(BaseDataLoader):
    """
    RefSeq DataLoader

    Any encoding or transformation should be done here
    and passed to the Dataset
    """
    def __init__(self,
                 genome_dir,
                 taxonomy_dir,
                 total_samples,
                 batch_size,
                 fixed_dataset=False,
                 drop_last=False,
                 training_distribution='uniform',
                 validation_distribution='lognormal',
                 accessions_file=None,
                 taxids_list=None,
                 error_model=None,
                 rmin=None,
                 rmax=None,
                 download=True,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 noise=None,
                 filter_by_level=None,
                 num_to_keep=1,
                 genome_cache_size=1000,
                 sample_cache_size=0):

        self.logger = logging.getLogger(self.__class__.__name__)

        ncbi_email = 'your-email@domain.com'
        ncbi_api = None

        g2read = mytransforms.GenomeToNoisyRead(error_model,
                                                rmin,
                                                rmax,
                                                p=noise)

        self.rmin = rmin
        self.rmax = rmax
        self.noise = noise
        self.error_model = error_model
        self.fixed_dataset = fixed_dataset
        self.training_distribution = training_distribution
        self.validation_distribution = validation_distribution

        if self.error_model:
            self.rmin = g2read.rmin
            self.rmax = g2read.rmax

        trsfm_x = mytransforms.Compose([
            g2read,
            mytransforms.ToTensorWithView(dtype=torch.long, view=[1, -1])
        ])
        trsfm_y = mytransforms.Compose([mytransforms.ToTensorWithView()])

        self.valid_loader = False

        self.dataset = datasets.RefSeqProkaryota(
            genome_dir,
            taxonomy_dir,
            total_samples,
            accessions_file,
            taxids_list,
            download=download,
            ncbi_email=ncbi_email,
            ncbi_api=ncbi_api,
            transform_x=trsfm_x,
            transform_y=trsfm_y,
            filter_by_level=filter_by_level,
            num_to_keep=num_to_keep,
            genome_cache_size=genome_cache_size)
        if self.valid_loader:
            self.dataset.set_simulator(validation_distribution)
        else:
            self.dataset.set_simulator(training_distribution)

        super(RefSeqProkaryotaDataLoader,
              self).__init__(self.dataset, batch_size, shuffle,
                             validation_split, num_workers, drop_last)

    def enable_multithreading_if_possible(self):
        if self.dataset.genome_cache_is_full():
            try:
                if self.num_workers != self._num_workers:
                    self.num_workers = self._num_workers
                    self.logger.info(f'Enabling {self.num_workers} '
                                     'workers for data loading...')
            except AttributeError:
                pass
        else:
            self._num_workers = self.num_workers
            self.num_workers = 0

    def step(self, epoch):
        super().step(epoch)
        self.enable_multithreading_if_possible()
        if not self.fixed_dataset:
            self.dataset.idx_offset = epoch * len(self.dataset)
            seed = epoch
            seed = seed * get_world_size() + get_global_rank()
            if self.valid_loader:
                seed = 2**32 - seed
                self.dataset.set_simulator(self.validation_distribution)
            else:
                self.dataset.set_simulator(self.training_distribution)
            self.dataset.simulator.set_seed(seed)

    def init_validation(self, other):
        super().init_validation(other)
        self.fixed_dataset = other.fixed_dataset
        self.valid_loader = True
        self.training_distribution = other.training_distribution
        self.validation_distribution = other.validation_distribution


class RefSeqProkaryotaBagsDataLoader(BaseDataLoader):
    """
    RefSeq DataLoader

    Any encoding or transformation should be done here
    and passed to the Dataset
    """
    def __init__(self,
                 target_format,
                 genome_dir,
                 taxonomy_dir,
                 total_bags,
                 bag_size,
                 batch_size,
                 fixed_dataset=False,
                 drop_last=False,
                 training_distribution='lognormal',
                 validation_distribution='lognormal',
                 reseed_every_n_bags=1,
                 accessions_file=None,
                 taxids_list=None,
                 error_model=None,
                 rmin=None,
                 rmax=None,
                 download=True,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 noise=None,
                 filter_by_level=None,
                 num_to_keep=1,
                 genome_cache_size=1000,
                 single_read_target_vectors=False):

        self.logger = logging.getLogger(self.__class__.__name__)

        ncbi_email = 'your-email@domain.com'
        ncbi_api = None

        g2read = mytransforms.GenomeToNoisyRead(error_model,
                                                rmin,
                                                rmax,
                                                p=noise)

        self.rmin = rmin
        self.rmax = rmax
        self.noise = noise
        self.error_model = error_model
        self.fixed_dataset = fixed_dataset
        self.training_distribution = training_distribution
        self.validation_distribution = validation_distribution

        if self.error_model:
            self.rmin = g2read.rmin
            self.rmax = g2read.rmax

        trsfm_x = mytransforms.Compose([
            g2read,
            mytransforms.ToTensorWithView(dtype=torch.long, view=[1, -1])
        ])
        trsfm_y = mytransforms.Compose([mytransforms.ToTensorWithView()])

        self.valid_loader = False

        self.dataset = datasets.RefSeqProkaryotaBags(
            bag_size=bag_size,
            total_bags=total_bags,
            target_format=target_format,
            genome_dir=genome_dir,
            taxonomy_dir=taxonomy_dir,
            accessions_file=accessions_file,
            taxids_list=taxids_list,
            download=download,
            ncbi_email=ncbi_email,
            ncbi_api=ncbi_api,
            transform_x=trsfm_x,
            transform_y=trsfm_y,
            filter_by_level=filter_by_level,
            num_to_keep=num_to_keep,
            reseed_every_n_bags=reseed_every_n_bags,
            genome_cache_size=genome_cache_size,
            num_workers=num_workers,
            single_read_target_vectors=single_read_target_vectors)
        if self.valid_loader:
            self.dataset.set_simulator(validation_distribution)
        else:
            self.dataset.set_simulator(training_distribution)

        super(RefSeqProkaryotaBagsDataLoader,
              self).__init__(self.dataset, batch_size, shuffle,
                             validation_split, 0, drop_last)

    def step(self, epoch):
        super().step(epoch)
        if not self.fixed_dataset:
            self.dataset.idx_offset = epoch * len(self.dataset)
            if self.valid_loader:
                self.dataset.set_simulator(self.validation_distribution)
            else:
                self.dataset.set_simulator(self.training_distribution)

    def init_validation(self, other):
        super().init_validation(other)
        self.fixed_dataset = other.fixed_dataset
        self.valid_loader = True
        self.training_distribution = other.training_distribution
        self.validation_distribution = other.validation_distribution


class RefSeqProkaryotaLargeBagsDataLoader(BaseDataLoader):
    """
    RefSeq DataLoader

    Any encoding or transformation should be done here
    and passed to the Dataset
    """
    def __init__(
        self,
        # target_format,
        genome_dir,
        taxonomy_dir,
        total_bags,
        bag_size,
        mini_bag_size,
        batch_size,
        fixed_dataset=False,
        drop_last=False,
        training_distribution='lognormal',
        validation_distribution='lognormal',
        reseed_every_n_bags=1,
        accessions_file=None,
        taxids_list=None,
        error_model=None,
        rmin=None,
        rmax=None,
        download=True,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        noise=None,
        filter_by_level=None,
        num_to_keep=1,
        genome_cache_size=1000,
        single_read_target_vectors=False):

        self.logger = logging.getLogger(self.__class__.__name__)

        ncbi_email = 'your-email@domain.com'
        ncbi_api = None

        g2read = mytransforms.GenomeToNoisyRead(error_model,
                                                rmin,
                                                rmax,
                                                p=noise)

        self.rmin = rmin
        self.rmax = rmax
        self.noise = noise
        self.error_model = error_model
        self.fixed_dataset = fixed_dataset
        self.training_distribution = training_distribution
        self.validation_distribution = validation_distribution

        if self.error_model:
            self.rmin = g2read.rmin
            self.rmax = g2read.rmax

        trsfm_x = mytransforms.Compose([
            g2read,
            mytransforms.ToTensorWithView(dtype=torch.long, view=[1, -1])
        ])
        trsfm_y = mytransforms.Compose([mytransforms.ToTensorWithView()])

        self.valid_loader = False

        self.dataset = datasets.RefSeqProkaryotaBags2(
            bag_size=bag_size,
            mini_bag_size=mini_bag_size,
            total_bags=total_bags,
            genome_dir=genome_dir,
            taxonomy_dir=taxonomy_dir,
            accessions_file=accessions_file,
            taxids_list=taxids_list,
            download=download,
            ncbi_email=ncbi_email,
            ncbi_api=ncbi_api,
            transform_x=trsfm_x,
            transform_y=trsfm_y,
            filter_by_level=filter_by_level,
            num_to_keep=num_to_keep,
            reseed_every_n_bags=reseed_every_n_bags,
            genome_cache_size=genome_cache_size,
            num_workers=num_workers)
        if self.valid_loader:
            self.dataset.set_simulator(validation_distribution)
        else:
            self.dataset.set_simulator(training_distribution)

        super().__init__(self.dataset,
                         batch_size,
                         shuffle,
                         validation_split,
                         0,
                         drop_last,
                         collate_fn=nothing_collate_fn)

    def step(self, epoch):
        super().step(epoch)
        if not self.fixed_dataset:
            self.dataset.idx_offset = epoch * len(self.dataset)
            if self.valid_loader:
                self.dataset.set_simulator(self.validation_distribution)
            else:
                self.dataset.set_simulator(self.training_distribution)

    def init_validation(self, other):
        super().init_validation(other)
        self.fixed_dataset = other.fixed_dataset
        self.valid_loader = True
        self.training_distribution = other.training_distribution
        self.validation_distribution = other.validation_distribution


class RefSeqProkaryotaKmerDataLoader(BaseDataLoader):
    """
    RefSeq DataLoader

    Any encoding or transformation should be done here
    and passed to the Dataset
    """
    def __init__(self,
                 genome_dir,
                 taxonomy_dir,
                 total_samples,
                 batch_size,
                 kmer_vocab_file,
                 fixed_dataset=False,
                 drop_last=False,
                 training_distribution='uniform',
                 validation_distribution='lognormal',
                 accessions_file=None,
                 taxids_list=None,
                 error_model=None,
                 rmin=None,
                 rmax=None,
                 kmer_processing_method='hash',
                 hash_bits=16,
                 lsh=True,
                 lsh_k=11,
                 alternative_lsh=False,
                 download=True,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 noise=None,
                 forward_reads_only=False,
                 filter_by_level=None,
                 num_to_keep=1,
                 genome_cache_size=1000):

        # if kmer_vocab_file is provided:
        #    kmers are mapped to numbers according to numbers
        # else:
        #    if kmer_processing_method is 'hash':
        #       use normal hash or lsh according to hash_bits, lsh, lsh_k, alternative_lsh
        #    if kmer_processing_method is 'count':
        #       use kmer counter to count subkmers of size lsh_k

        assert kmer_processing_method in ['hash', 'count']

        self.logger = logging.getLogger(self.__class__.__name__)

        ncbi_email = 'your-email@domain.com'
        ncbi_api = None

        g2read = mytransforms.GenomeToNoisyKmerRead(
            kmer_vocab_file,
            error_model,
            rmin,
            rmax,
            p=noise,
            forward_reads_only=forward_reads_only,
            hash_bits=hash_bits,
            lsh=lsh,
            lsh_k=lsh_k,
            alternative_lsh=alternative_lsh,
            kmer_processing_method=kmer_processing_method)
        self.lsh_k = lsh_k

        if kmer_processing_method == 'hash':
            totensor = mytransforms.ToTensorWithView(dtype=torch.long,
                                                     view=[-1])
        else:
            totensor = mytransforms.ToTensorWithView(dtype=torch.float)

        self.rmin = rmin
        self.rmax = rmax
        self.noise = noise
        self.error_model = error_model
        self.fixed_dataset = fixed_dataset
        self.training_distribution = training_distribution
        self.validation_distribution = validation_distribution

        try:
            self.vocab_size = g2read.vocab_size
        except AttributeError:
            self.vocab_size = None
        if self.error_model:
            self.rmin = g2read.rmin
            self.rmax = g2read.rmax

        trsfm_x = mytransforms.Compose([g2read, totensor])
        trsfm_y = mytransforms.Compose([mytransforms.ToTensorWithView()])

        self.valid_loader = False

        self.dataset = datasets.RefSeqProkaryota(
            genome_dir,
            taxonomy_dir,
            total_samples,
            accessions_file,
            taxids_list,
            download=download,
            ncbi_email=ncbi_email,
            ncbi_api=ncbi_api,
            transform_x=trsfm_x,
            transform_y=trsfm_y,
            filter_by_level=filter_by_level,
            num_to_keep=num_to_keep,
            genome_cache_size=genome_cache_size)
        if self.valid_loader:
            self.dataset.set_simulator(validation_distribution)
        else:
            self.dataset.set_simulator(training_distribution)

        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers, drop_last)

    def enable_multithreading_if_possible(self):
        if self.dataset.genome_cache_is_full():
            try:
                if self.num_workers != self._num_workers:
                    self.num_workers = self._num_workers
                    self.logger.info(f'Enabling {self.num_workers} '
                                     'workers for data loading...')
            except AttributeError:
                pass
        else:
            self._num_workers = self.num_workers
            self.num_workers = 0

    def step(self, epoch):
        super().step(epoch)
        self.enable_multithreading_if_possible()
        if not self.fixed_dataset:
            self.dataset.idx_offset = epoch * len(self.dataset)
            seed = epoch
            seed = seed * get_world_size() + get_global_rank()
            if self.valid_loader:
                seed = 2**32 - seed
                self.dataset.set_simulator(self.validation_distribution)
            else:
                self.dataset.set_simulator(self.training_distribution)
            self.dataset.simulator.set_seed(seed)

    def init_validation(self, other):
        super().init_validation(other)
        self.fixed_dataset = other.fixed_dataset
        self.valid_loader = True
        self.training_distribution = other.training_distribution
        self.validation_distribution = other.validation_distribution


class RefSeqProkaryotaKmerBagsDataLoader(BaseDataLoader):
    """
    RefSeq DataLoader

    Any encoding or transformation should be done here
    and passed to the Dataset
    """
    def __init__(self,
                 target_format,
                 genome_dir,
                 taxonomy_dir,
                 total_bags,
                 bag_size,
                 batch_size,
                 kmer_vocab_file,
                 fixed_dataset=False,
                 drop_last=False,
                 training_distribution='lognormal',
                 validation_distribution='lognormal',
                 reseed_every_n_bags=1,
                 accessions_file=None,
                 taxids_list=None,
                 error_model=None,
                 rmin=None,
                 rmax=None,
                 kmer_processing_method='hash',
                 hash_bits=16,
                 lsh=True,
                 lsh_k=11,
                 alternative_lsh=False,
                 download=True,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 noise=None,
                 forward_reads_only=False,
                 filter_by_level=None,
                 num_to_keep=1,
                 genome_cache_size=1000,
                 single_read_target_vectors=False):

        self.logger = logging.getLogger(self.__class__.__name__)

        ncbi_email = 'your-email@domain.com'
        ncbi_api = None

        g2read = mytransforms.GenomeToNoisyKmerRead(
            kmer_vocab_file,
            error_model,
            rmin,
            rmax,
            p=noise,
            forward_reads_only=forward_reads_only,
            hash_bits=hash_bits,
            lsh=lsh,
            lsh_k=lsh_k,
            alternative_lsh=alternative_lsh,
            kmer_processing_method=kmer_processing_method)
        self.lsh_k = lsh_k

        if kmer_processing_method == 'hash':
            totensor = mytransforms.ToTensorWithView(dtype=torch.long,
                                                     view=[-1])
        else:
            totensor = mytransforms.ToTensorWithView(dtype=torch.float)

        self.rmin = rmin
        self.rmax = rmax
        self.noise = noise
        self.error_model = error_model
        self.fixed_dataset = fixed_dataset
        self.training_distribution = training_distribution
        self.validation_distribution = validation_distribution

        try:
            self.vocab_size = g2read.vocab_size
        except AttributeError:
            self.vocab_size = None
        if self.error_model:
            self.rmin = g2read.rmin
            self.rmax = g2read.rmax

        trsfm_x = mytransforms.Compose([g2read, totensor])
        trsfm_y = mytransforms.Compose([mytransforms.ToTensorWithView()])

        self.valid_loader = False

        self.dataset = datasets.RefSeqProkaryotaBags(
            bag_size=bag_size,
            total_bags=total_bags,
            target_format=target_format,
            genome_dir=genome_dir,
            taxonomy_dir=taxonomy_dir,
            accessions_file=accessions_file,
            taxids_list=taxids_list,
            download=download,
            ncbi_email=ncbi_email,
            ncbi_api=ncbi_api,
            transform_x=trsfm_x,
            transform_y=trsfm_y,
            filter_by_level=filter_by_level,
            num_to_keep=num_to_keep,
            reseed_every_n_bags=reseed_every_n_bags,
            genome_cache_size=genome_cache_size,
            num_workers=num_workers,
            single_read_target_vectors=single_read_target_vectors)
        if self.valid_loader:
            self.dataset.set_simulator(validation_distribution)
        else:
            self.dataset.set_simulator(training_distribution)

        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         0, drop_last)

    def step(self, epoch):
        super().step(epoch)
        if not self.fixed_dataset:
            self.dataset.idx_offset = epoch * len(self.dataset)
            if self.valid_loader:
                self.dataset.set_simulator(self.validation_distribution)
            else:
                self.dataset.set_simulator(self.training_distribution)

    def init_validation(self, other):
        super().init_validation(other)
        self.fixed_dataset = other.fixed_dataset
        self.valid_loader = True
        self.training_distribution = other.training_distribution
        self.validation_distribution = other.validation_distribution


if __name__ == '__main__':
    # Testing

    # rspdl = RefSeqProkaryotaDataLoader(
    #     ("/home/ageorgiou/eth/spring2019/"
    #      "thesis/data/refseq_prokaryota/genomes"),
    #     ("/home/ageorgiou/eth/spring2019/"
    #      "thesis/data/refseq_prokaryota/taxonomy"),
    #     100,
    #     2, ('/home/ageorgiou/eth/spring2019/'
    #         'thesis/data/refseq_prokaryota/ncbi_id_training.txt'),
    #     10,
    #     20,
    #     True,
    #     'your-email@domain.com',
    #     None,
    #     True,
    #     0.1,
    #     1,
    #     0.1,
    #     genome_cache_size=0)
    #
    # print(rspdl.n_samples)
    #
    # val_rspdl = rspdl.split_validation()
    # print(type(rspdl), type(val_rspdl))
    # for i, batch in enumerate(rspdl):
    #     if i > 0:
    #         break
    #     print(batch)
    # print("testing")
    # for i, batch in enumerate(val_rspdl):
    #     if i > 0:
    #         break
    #     print(batch)
    # print("testing")
    #
    # rspdl.step(5)
    # val_rspdl.eval()
    # print(rspdl.dataset.eval_mode)
    # print(val_rspdl.dataset.eval_mode)
    # for i, batch in enumerate(val_rspdl):
    #     if i > 0:
    #         break
    #     print(batch)

    # dl = RefSeqProkaryotaKmerBagsDataLoader(
    #     target_format='counts',
    #     genome_dir=("/home/ageorgiou/eth/spring2019/"
    #                 "thesis/data/refseq_prokaryota/genomes"),
    #     taxonomy_dir=("/home/ageorgiou/eth/spring2019/"
    #                   "thesis/data/refseq_prokaryota/taxonomy"),
    #     kmer_vocab_file=("/home/ageorgiou/eth/spring2019/"
    #                      "thesis/data/refseq_prokaryota/token_8mer"),
    #     total_bags=16,
    #     bag_size=4,
    #     batch_size=2,
    #     fixed_dataset=False,
    #     drop_last=False,
    #     dataset_distribution='lognormal',
    #     accessions_file=(
    #         '/home/ageorgiou/eth/spring2019/'
    #         'thesis/data/refseq_prokaryota/ncbi_id_training_filtered.txt'),
    #     taxids_list=None,
    #     error_model='perfect_16',
    #     rmin=None,
    #     rmax=None,
    #     download=False,
    #     shuffle=False,
    #     validation_split=0.0,
    #     num_workers=2,
    #     noise=None,
    #     filter_by_level=None,
    #     num_to_keep=1,
    #     genome_cache_size=1000)
    #
    # for i, (x, y) in enumerate(dl):
    #     if i > 0:
    #         break
    #     print(x.size(), x)
    #     print(y)

    dl = RefSeqProkaryotaLargeBagsDataLoader(
        genome_dir=("/home/ageorgiou/eth/spring2019/"
                    "thesis/data/refseq_prokaryota/genomes"),
        taxonomy_dir=("/home/ageorgiou/eth/spring2019/"
                      "thesis/data/refseq_prokaryota/taxonomy"),
        total_bags=16,
        bag_size=8,
        mini_bag_size=2,
        batch_size=1,
        fixed_dataset=False,
        drop_last=False,
        training_distribution='lognormal',
        validation_distribution='lognormal',
        accessions_file=(
            '/home/ageorgiou/eth/spring2019/'
            'thesis/data/refseq_prokaryota/ncbi_id_training_filtered.txt'),
        taxids_list=None,
        error_model='perfect_16',
        rmin=None,
        rmax=None,
        download=False,
        shuffle=False,
        validation_split=0.0,
        num_workers=2,
        noise=None,
        filter_by_level=None,
        num_to_keep=1,
        genome_cache_size=1000)

    for i, it in enumerate(dl):
        print(it)
        if i > 0:
            break
        for sub_it in it:
            for x, y in sub_it:
                print(x.size())
                print(y)
                print('--------------')
            print('===========================')
