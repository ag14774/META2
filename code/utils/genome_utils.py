import gzip
import logging
import urllib
from pathlib import Path

import iss
import numpy as np
import pysam
from Bio import Entrez, SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from iss.error_models import kde, ErrorModel
from iss.util import rev_comp
from utils.util import check_file_integrity, create_md5_file
from Bio.SeqIO.QualityIO import FastqGeneralIterator


class PerfectErrorModel(ErrorModel):
    """Perfect error model class
    Perfect Error Model. This is a model without errors.
    All Phred score are 40. No errors are introduced at all.
    """
    def __init__(self, read_length=151):
        super().__init__()
        self.read_length = read_length
        self.insert_size = 200
        self.quality_forward = self.quality_reverse = 40

        self.subst_choices_for = self.subst_choices_rev = [{
            'A': (['A', 'T', 'C', 'G'], [1, 0, 0, 0]),
            'T': (['A', 'T', 'C', 'G'], [0, 1, 0, 0]),
            'C': (['A', 'T', 'C', 'G'], [0, 0, 1, 0]),
            'G': (['A', 'T', 'C', 'G'], [0, 0, 0, 1])
        } for _ in range(self.read_length)]

        self.ins_for = self.ins_rev = self.del_for = self.del_rev = [{
            'A': 0.0,
            'T': 0.0,
            'C': 0.0,
            'G': 0.0
        } for _ in range(self.read_length)]

    def gen_phred_scores(self, mean_quality, orientation):
        """Fake randorm function returning the distribution of Phred
        scores. Score will be 40 for all positions
        Returns:
            list: list of phred scores (40 along the whole read)
        """
        return [40 for _ in range(self.read_length)]

    def random_insert_size(self):
        """Fake random function returning the default insert size of the
        basic arror model
        Returns:
            int: insert size
        """
        return self.insert_size


class GenomeTools(object):
    """
    Provides access to samtools and other useful utilities
    for handling genome files
    """
    def __init__(self, ncbi_email='your-email@domain.com', ncbi_api=None):
        self.ncbi_email = ncbi_email
        self.ncbi_api = ncbi_api
        Entrez.email = self.ncbi_email
        if self.ncbi_api:
            Entrez.api_key = self.ncbi_api
        self.logger = logging.getLogger(self.__class__.__name__)

    def download_genome(self, ncbi_id, genome_dir):

        genome_dir = Path(genome_dir)
        microbe_id = ncbi_id.split('.')[0]
        genome_file = genome_dir / (microbe_id + '.fasta')
        genome_md5 = genome_dir / (microbe_id + '.fasta.md5')

        if check_file_integrity(genome_file, genome_md5):
            return True

        try:
            handle = Entrez.efetch(db="nucleotide",
                                   id=microbe_id,
                                   rettype="fasta",
                                   retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            SeqIO.write(record, str(genome_file), "fasta")
            create_md5_file(genome_file, genome_md5)
            return True
        except urllib.error.HTTPError:
            self.logger.warning(
                f"Could not download genome for {microbe_id}...")
            return False
        except urllib.error.URLError:
            self.logger.warning(
                f"Could not download genome for {microbe_id}...")
            return False

    @classmethod
    def gunzip_genomes(cls, genome_dir, outfile):
        logger = logging.getLogger(cls.__name__)

        genome_dir = Path(genome_dir)
        outfile = genome_dir / outfile
        logger.info("Extracting all genomes...")
        if outfile.is_file():
            logger.info('{} already exists. Skipping gunzip_genomes...'.format(
                outfile.name))
            return

        filenames = genome_dir.glob('*.gz')
        outf = outfile.open('wt')
        for filename in filenames:
            logger.debug("Processing file: {}...".format(filename))
            try:
                with gzip.open(str(genome_dir / filename), 'rt') as f:
                    for line in f:
                        outf.write(line)
            except OSError:
                logger.warning("{} is not a gzipped file".format(filename))
        outf.close()

    @classmethod
    def create_index(cls, fasta_file, force_overwrite=False):
        logger = logging.getLogger(cls.__name__)

        fasta_file = Path(fasta_file)
        if not fasta_file.is_file():
            logger.error("File {} not found".format(fasta_file))
            exit(1)

        if not fasta_file.with_name(fasta_file.name +
                                    '.fai').is_file() or force_overwrite:
            pysam.faidx(str(fasta_file))

    @classmethod
    def read_fasta(cls, fasta_file, ncbi_id=None):
        '''
        If ncbi_id is provided then an index needs to be created first.
        Otherwise, the whole fasta_file is loaded
        '''
        logger = logging.getLogger(cls.__name__)

        fasta_file = Path(fasta_file)
        if not fasta_file.is_file():
            logger.error("File {} not found".format(fasta_file))
            exit(1)

        if ncbi_id:
            if not fasta_file.with_name(fasta_file.name + '.fai').is_file():
                logger.error("Index not found. Run create_index first")
                exit(1)

            try:
                res = "".join(
                    pysam.faidx(str(fasta_file), ncbi_id).split()[1:])
                res = res.strip()
                return res
            except pysam.utils.SamtoolsError:
                return None
        else:
            res = ""
            with fasta_file.open('rt') as f:
                res = [line.strip() for line in f][1:]
            res = "".join(res)
            res = res.strip()
            return res

    @classmethod
    def read_fastq(cls, fastq_file):
        '''
        If ncbi_id is provided then an index needs to be created first.
        Otherwise, the whole fasta_file is loaded
        '''
        fastq_file = Path(fastq_file)
        with fastq_file.open() as handle:
            res = [(title.rsplit('|', 1)[1], seq)
                   for title, seq, _ in FastqGeneralIterator(handle)]

        return res

    @classmethod
    def num_of_ids(cls, fasta_file):
        logger = logging.getLogger(cls.__name__)

        fasta_file = Path(fasta_file)
        if not fasta_file.with_name(fasta_file.name + '.fai').is_file():
            logger.error("Index not found. Run create_index first")
            exit(1)

        with fasta_file.with_name(fasta_file.name + '.fai').open('r') as f:
            num_lines = sum(1 for line in f)
        return num_lines

    @classmethod
    def get_all_ids(cls, fasta_file):
        logger = logging.getLogger(cls.__name__)

        fasta_file = Path(fasta_file)
        if not fasta_file.with_name(fasta_file.name + '.fai').is_file():
            logger.error("Index not found. Run create_index first")
            exit(1)

        ids = []
        with fasta_file.with_name(fasta_file.name + '.fai').open('r') as f:
            for line in f:
                ids.append(line.split()[0])
        return ids

    @classmethod
    def create_error_model(cls, error_model):
        def _get_model_path(model_type):
            p = Path(iss.__file__).parent
            if model_type.lower() == 'hiseq':
                npz = p / 'profiles/HiSeq'
            elif model_type.lower() == 'novaseq':
                npz = p / 'profiles/NovaSeq'
            elif model_type.lower() == 'miseq':
                npz = p / 'profiles/MiSeq'
            return str(npz)

        if error_model.startswith('perfect'):
            try:
                error_model = PerfectErrorModel(
                    read_length=int(error_model.split('_')[1]))
            except IndexError:
                error_model = PerfectErrorModel()
        else:
            error_model_path = _get_model_path(error_model)
            error_model = kde.KDErrorModel(error_model_path)
        return error_model

    @classmethod
    def simulate_read_with_error_model(cls,
                                       genome,
                                       ErrorModel,
                                       i,
                                       always_forward=True,
                                       return_string=True,
                                       name=None):
        """Form a read from one genome (or sequence) according to an
        ErrorModel
        returns a string
        Args:
            genome (string): sequence or genome of reference
            ErrorModel (ErrorModel): an ErrorModel class
            i (int): a number identifying the read
        Returns:
            string: a string representing a single read
        """
        # ErrorModel.read_length = ErrorModel.read_length - 1
        np_random = np.random.RandomState(seed=i)

        read_length = ErrorModel.read_length

        if len(genome) <= read_length:
            genome = "".join([genome, "N" * (read_length - len(genome) + 1)])

        if name:
            header = f'genome_{i}|{name}'
        else:
            header = f'genome_{i}'
        record = SeqRecord(Seq(genome, IUPAC.unambiguous_dna),
                           id=header,
                           description='')

        sequence = record.seq
        header = record.id

        # generate the forward read
        forward_start = np_random.randint(
            low=0, high=max(len(record.seq) - read_length + 1, 1))

        forward_end = forward_start + read_length

        generate_forward = np_random.randint(low=0, high=2)

        if generate_forward or always_forward:

            bounds = (forward_start, forward_end)
            # create a perfect read
            forward = SeqRecord(Seq(str(sequence[forward_start:forward_end]),
                                    IUPAC.unambiguous_dna),
                                id=header,
                                description='')
            # add the indels, the qual scores and modify the record accordingly
            forward.seq = ErrorModel.introduce_indels(forward, 'forward',
                                                      sequence, bounds)
            forward = ErrorModel.introduce_error_scores(forward, 'forward')
            forward.seq = ErrorModel.mut_sequence(forward, 'forward')

            if return_string:
                return str(forward.seq)
            else:
                return forward

        else:
            insert_size = ErrorModel.random_insert_size()
            try:
                reverse_start = forward_end + insert_size
                reverse_end = reverse_start + read_length
                assert reverse_end < len(record.seq)
            except AssertionError:
                reverse_end = np_random.randint(low=read_length,
                                                high=len(record.seq))
                reverse_start = reverse_end - read_length

            bounds = (reverse_start, reverse_end)
            reverse = SeqRecord(Seq(
                rev_comp(str(sequence[reverse_start:reverse_end])),
                IUPAC.unambiguous_dna),
                                id=header,
                                description='')
            reverse.seq = ErrorModel.introduce_indels(reverse, 'reverse',
                                                      sequence, bounds)
            reverse = ErrorModel.introduce_error_scores(reverse, 'reverse')
            reverse.seq = ErrorModel.mut_sequence(reverse, 'reverse')

            if return_string:
                return str(reverse.seq)
            else:
                return reverse

    @classmethod
    def simulate_read_with_uniform_noise(cls, genome, idx, rmin, rmax, p,
                                         nuc2int):

        np_random = np.random.RandomState(seed=idx)
        loc = np_random.randint(low=0, high=max(len(genome) - rmax + 1, 1))
        length = np_random.randint(low=rmin, high=rmax + 1)

        xold = genome[loc:loc + length].strip()

        x = np.zeros(rmax)
        x[:len(xold)] = [nuc2int[c] for c in xold]

        if p > 0:
            where_to_flip = (np_random.uniform(size=len(xold)) < p)
            noise = np_random.randint(size=len(xold), low=1, high=5)
            x[:len(xold
                   )] = ~where_to_flip * x[:len(xold)] + where_to_flip * noise

        return x


if __name__ == '__main__':
    gt = GenomeTools()
    gt.gunzip_genomes(
        '/home/ageorgiou/eth/spring2019/thesis/data/refseq_prokaryota/genomes',
        'prok.fa')
    gt.create_index(('/home/ageorgiou/eth/spring2019'
                     '/thesis/data/refseq_prokaryota/genomes/prok.fa'))
    print(
        gt.query_ncbi(('/home/ageorgiou/eth/spring2019'
                       '/thesis/data/refseq_prokaryota/genomes/prok.fa'),
                      'NZ_AUAU01000030.1'))
    print(
        gt.num_of_ids(('/home/ageorgiou/eth/spring2019'
                       '/thesis/data/refseq_prokaryota/genomes/prok.fa')))
    print(
        gt.query_ncbi(('/home/ageorgiou/eth/spring2019'
                       '/thesis/data/refseq_prokaryota/genomes/prok.fa'),
                      'NZ'))
    print(
        gt.get_all_ids(('/home/ageorgiou/eth/spring2019'
                        '/thesis/data/refseq_prokaryota/genomes/prok.fa'))[:5])
