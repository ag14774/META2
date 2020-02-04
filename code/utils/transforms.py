import numpy as np
import torch
from torchvision import transforms
import logging
from utils.genome_utils import GenomeTools
from pathlib import Path
import xxhash
import sourmash
import marshal
from itertools import product
import re


class Compose(transforms.Compose):
    def __call__(self, sample, *args, **kwargs):
        for t in self.transforms:
            sample = t(sample, *args, **kwargs)
        return sample


class GenomeToSeqRecord(object):
    """Convert string to numpy array of integers."""
    def __init__(self, error_model=None, rmin=None, rmax=None, p=None):
        logger = logging.getLogger(self.__class__.__name__)

        if error_model:
            error_model = GenomeTools.create_error_model(error_model)
            if rmin or rmax:
                logger.warning("Ignoring provided rmin and rmax values.. "
                               "Using error_model.read_length.")
            rmin = int(error_model.read_length)
            rmax = int(error_model.read_length)

        self.error_model = error_model
        self.rmin = rmin
        self.rmax = rmax
        self.p = p

    def __call__(self, genome, idx, name, *args, **kwargs):

        if self.error_model:
            x = GenomeTools.simulate_read_with_error_model(genome,
                                                           self.error_model,
                                                           idx,
                                                           return_string=False,
                                                           name=name)

        return x


class GenomeToNoisyRead(object):
    """Convert string to numpy array of integers."""
    def __init__(self, error_model=None, rmin=None, rmax=None, p=None):
        logger = logging.getLogger(self.__class__.__name__)

        if error_model:
            error_model = GenomeTools.create_error_model(error_model)
            if rmin or rmax:
                logger.warning("Ignoring provided rmin and rmax values.. "
                               "Using error_model.read_length.")
            rmin = int(error_model.read_length)
            rmax = int(error_model.read_length)
        else:
            if not rmin or not rmax or not p:
                logger.error("Set error_model or (rmin, rmax, p)...")
                exit(1)

        self.nuc2int = {
            'A': 1,
            'C': 2,
            'T': 3,
            'G': 4,
            'N': 5,
            'R': 5,
            'Y': 5,
            'S': 5,
            'W': 5,
            'K': 5,
            'M': 5,
            'B': 5,
            'D': 5,
            'H': 5,
            'V': 5,
            '.': 5,
            '-': 5
        }

        self.error_model = error_model
        self.rmin = rmin
        self.rmax = rmax
        self.p = p

    def __call__(self, genome, idx, *args, **kwargs):

        if self.error_model:
            xold = GenomeTools.simulate_read_with_error_model(
                genome, self.error_model, idx)
            x = np.array([self.nuc2int[c] for c in xold])
        else:
            x = GenomeTools.simulate_read_with_uniform_noise(
                genome, idx, self.rmin, self.rmax, self.p, self.nuc2int)

        return x


class GenomeToNoisyKmerRead(object):
    """Convert string to numpy array of integers."""
    def __init__(self,
                 vocab_path,
                 error_model=None,
                 rmin=None,
                 rmax=None,
                 p=None,
                 forward_reads_only=False,
                 hash_bits=16,
                 lsh=True,
                 lsh_k=11,
                 alternative_lsh=False,
                 kmer_processing_method='hash'):
        logger = logging.getLogger(self.__class__.__name__)
        self.forward_reads_only = forward_reads_only
        if error_model:
            error_model = GenomeTools.create_error_model(error_model)
            if rmin or rmax:
                logger.warning("Ignoring provided rmin and rmax values.. "
                               "Using error_model.read_length.")
            rmin = int(error_model.read_length)
            rmax = int(error_model.read_length)
        else:
            if not rmin or not rmax or not p:
                logger.error("Set error_model or (rmin, rmax, p)...")
                exit(1)

        assert kmer_processing_method in ['hash', 'count']
        self.kmer_processing_method = kmer_processing_method
        if isinstance(vocab_path, int):
            self.k = vocab_path
            if kmer_processing_method == 'hash':
                if lsh:
                    if not alternative_lsh:
                        self.kmer2index = self.kmer2index_lsh
                    else:
                        self.kmer2index = self.kmer2index_lsh2
                else:
                    self.kmer2index = self.kmer2index_hash
                self.max_hash = 2**hash_bits
                self.max_hash_bits = hash_bits
                self.vocab_size = self.max_hash
                self.lsh_k = lsh_k
            elif kmer_processing_method == 'count':
                self.kmer2index = self.kmer2counts
                self.lsh_k = lsh_k
                self.lshmer2id = {
                    ''.join(c): i
                    for i, c in enumerate(product('ACTG', repeat=self.lsh_k))
                }
        else:
            self.kmer2id, self.k = self.load_vocab(vocab_path)
            self.vocab_size = len(self.kmer2id)

        self.nuc2int = {
            'A': 1,
            'C': 2,
            'T': 3,
            'G': 4,
            'N': 5,
            'R': 5,
            'Y': 5,
            'S': 5,
            'W': 5,
            'K': 5,
            'M': 5,
            'B': 5,
            'D': 5,
            'H': 5,
            'V': 5,
            '.': 5,
            '-': 5
        }

        self.int2nuc = ['N', 'A', 'C', 'T', 'G', 'N']

        self.error_model = error_model
        self.rmin = rmin
        self.rmax = rmax
        self.p = p

    def load_vocab(self, vocab_path):
        vocab_path = Path(vocab_path)
        kmer2id = {}
        idx = 0
        k = 0
        with vocab_path.open('rt') as handle:
            for line in handle:
                word = line.rstrip()
                if word != '<unk>':
                    k = len(word)
                kmer2id[word] = idx
                idx += 1
        return kmer2id, k

    def forward2reverse(self, dna):
        """Converts an oligonucleotide(k-mer) to its reverse complement sequence.
        All ambiguous bases are treated as Ns.
        """
        translation_dict = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
            "N": "N",
            "K": "N",
            "M": "N",
            "R": "N",
            "Y": "N",
            "S": "N",
            "W": "N",
            "B": "N",
            "V": "N",
            "H": "N",
            "D": "N",
            "X": "N"
        }
        letters = list(dna)
        letters = [translation_dict[base] for base in letters]
        return ''.join(letters)[::-1]

    def kmer2index(self, k_mer):
        """Converts k-mer to index to the embedding layer"""
        kmer2id = self.kmer2id
        if k_mer in kmer2id:
            idx = kmer2id[k_mer]
        elif self.forward2reverse(k_mer) in kmer2id:
            idx = kmer2id[self.forward2reverse(k_mer)]
        else:
            idx = kmer2id['<unk>']
        return idx

    def kmer2index_hash(self, k_mer):
        """Converts k-mer to index to the embedding layer"""
        if 'N' in k_mer:
            k_mer = '<unk>'
        mask = (0x1 << self.max_hash_bits) - 1
        idx = xxhash.xxh32(k_mer).intdigest() & mask
        return idx

    def kmer2index_lsh2(self, k_mer):
        hash = xxhash.xxh32()
        minhash = sourmash.MinHash(n=0, ksize=self.lsh_k, scaled=1)
        minhash.add_sequence(k_mer, True)
        hashes = sorted(minhash.get_hashes())
        hash.update(marshal.dumps([hashes[0], hashes[-1]]))
        mask = (0x1 << self.max_hash_bits) - 1
        idx = hash.intdigest() & mask
        return idx

    def kmer2index_lsh(self, k_mer):
        hash = xxhash.xxh32()
        minhash = sourmash.MinHash(n=0, ksize=self.lsh_k, scaled=1)
        minhash.add_sequence(k_mer, True)
        hashes = sorted(minhash.get_hashes())
        hashes = hashes[:max(1, len(hashes) // 2)]
        hash.update(marshal.dumps(hashes))
        mask = (0x1 << self.max_hash_bits) - 1
        idx = hash.intdigest() & mask
        return idx

    def kmer2counts(self, k_mer):
        lsh_k = self.lsh_k
        lshmer2id = self.lshmer2id
        res = [0] * (len(lshmer2id))
        for i in range(0, len(k_mer) - lsh_k + 1):
            smallmer = k_mer[i:i + lsh_k]
            if 'N' in smallmer:
                continue
            res[lshmer2id[smallmer]] += 1
        return res

    def seq2kmer(self, seq):
        """Converts a DNA sequence split into a list of k-mers.
        The sequences in one data set do not have to share the same length.
        Returns:
             kmer_array: a numpy array of corresponding k-mer indexes.
        """
        if self.kmer_processing_method == 'count':
            seq = re.sub('[NKMRYSWBVHDX]', 'N', seq)
        k = self.k
        l = len(seq)
        kmer_list = []
        for i in range(0, l):
            if i + k >= l + 1:
                break
            k_mer = seq[i:i + k]
            idx = self.kmer2index(k_mer)
            kmer_list.append(idx)
        kmer_array = np.array(kmer_list)
        return kmer_array

    #
    # def seq2kmer_counts(self, seq):
    #     if self.kmer_processing_method == 'count':
    #         seq = re.sub('[NKMRYSWBVHDX]', 'N', seq)
    #     k = self.k
    #     l = len(seq)
    #     lsh_k = self.lsh_k
    #     lshmer2id = self.lshmer2id
    #     res = np.zeros((l - k + 2, len(lshmer2id)))
    #     for

    def __call__(self, genome, idx, *args, **kwargs):

        if self.error_model:
            read = GenomeTools.simulate_read_with_error_model(
                genome,
                self.error_model,
                idx,
                always_forward=self.forward_reads_only)
        else:
            read = GenomeTools.simulate_read_with_uniform_noise(
                genome, idx, self.rmin, self.rmax, self.p, self.nuc2int)
            read = ''.join([self.int2nuc[i] for i in read])

        read = self.seq2kmer(read)

        return read


class ReadToKmerRead(object):
    """Convert string to numpy array of integers."""
    def __init__(self,
                 vocab_path,
                 hash_bits=16,
                 lsh=True,
                 lsh_k=11,
                 alternative_lsh=False,
                 kmer_processing_method='hash'):
        logger = logging.getLogger(self.__class__.__name__)

        assert kmer_processing_method in ['hash', 'count']
        self.kmer_processing_method = kmer_processing_method
        if isinstance(vocab_path, int):
            self.k = vocab_path
            if kmer_processing_method == 'hash':
                if lsh:
                    if not alternative_lsh:
                        self.kmer2index = self.kmer2index_lsh
                    else:
                        self.kmer2index = self.kmer2index_lsh2
                else:
                    self.kmer2index = self.kmer2index_hash
                self.max_hash = 2**hash_bits
                self.max_hash_bits = hash_bits
                self.vocab_size = self.max_hash
                self.lsh_k = lsh_k
            elif kmer_processing_method == 'count':
                self.kmer2index = self.kmer2counts
                self.lsh_k = lsh_k
                self.lshmer2id = {
                    ''.join(c): i
                    for i, c in enumerate(product('ACTG', repeat=self.lsh_k))
                }
        else:
            self.kmer2id, self.k = self.load_vocab(vocab_path)
            self.vocab_size = len(self.kmer2id)

    def load_vocab(self, vocab_path):
        vocab_path = Path(vocab_path)
        kmer2id = {}
        idx = 0
        k = 0
        with vocab_path.open('rt') as handle:
            for line in handle:
                word = line.rstrip()
                if word != '<unk>':
                    k = len(word)
                kmer2id[word] = idx
                idx += 1
        return kmer2id, k

    def forward2reverse(self, dna):
        """Converts an oligonucleotide(k-mer) to its reverse complement sequence.
        All ambiguous bases are treated as Ns.
        """
        translation_dict = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
            "N": "N",
            "K": "N",
            "M": "N",
            "R": "N",
            "Y": "N",
            "S": "N",
            "W": "N",
            "B": "N",
            "V": "N",
            "H": "N",
            "D": "N",
            "X": "N"
        }
        letters = list(dna)
        letters = [translation_dict[base] for base in letters]
        return ''.join(letters)[::-1]

    def kmer2index(self, k_mer):
        """Converts k-mer to index to the embedding layer"""
        kmer2id = self.kmer2id
        if k_mer in kmer2id:
            idx = kmer2id[k_mer]
        elif self.forward2reverse(k_mer) in kmer2id:
            idx = kmer2id[self.forward2reverse(k_mer)]
        else:
            idx = kmer2id['<unk>']
        return idx

    def kmer2index_hash(self, k_mer):
        """Converts k-mer to index to the embedding layer"""
        if 'N' in k_mer:
            k_mer = '<unk>'
        mask = (0x1 << self.max_hash_bits) - 1
        idx = xxhash.xxh32(k_mer).intdigest() & mask
        return idx

    def kmer2index_lsh2(self, k_mer):
        hash = xxhash.xxh32()
        minhash = sourmash.MinHash(n=0, ksize=self.lsh_k, scaled=1)
        minhash.add_sequence(k_mer, True)
        hashes = sorted(minhash.get_hashes())
        hash.update(marshal.dumps([hashes[0], hashes[-1]]))
        mask = (0x1 << self.max_hash_bits) - 1
        idx = hash.intdigest() & mask
        return idx

    def kmer2index_lsh(self, k_mer):
        hash = xxhash.xxh32()
        minhash = sourmash.MinHash(n=0, ksize=self.lsh_k, scaled=1)
        minhash.add_sequence(k_mer, True)
        hashes = sorted(minhash.get_hashes())
        hashes = hashes[:max(1, len(hashes) // 2)]
        hash.update(marshal.dumps(hashes))
        mask = (0x1 << self.max_hash_bits) - 1
        idx = hash.intdigest() & mask
        return idx

    def kmer2counts(self, k_mer):
        lsh_k = self.lsh_k
        lshmer2id = self.lshmer2id
        res = [0] * (len(lshmer2id))
        for i in range(0, len(k_mer) - lsh_k + 1):
            smallmer = k_mer[i:i + lsh_k]
            if 'N' in smallmer:
                continue
            res[lshmer2id[smallmer]] += 1
        return res

    def seq2kmer(self, seq):
        """Converts a DNA sequence split into a list of k-mers.
        The sequences in one data set do not have to share the same length.
        Returns:
             kmer_array: a numpy array of corresponding k-mer indexes.
        """
        if self.kmer_processing_method == 'count':
            seq = re.sub('[NKMRYSWBVHDX]', 'N', seq)
        k = self.k
        l = len(seq)
        kmer_list = []
        for i in range(0, l):
            if i + k >= l + 1:
                break
            k_mer = seq[i:i + k]
            idx = self.kmer2index(k_mer)
            kmer_list.append(idx)
        kmer_array = np.array(kmer_list)
        return kmer_array

    #
    # def seq2kmer_counts(self, seq):
    #     if self.kmer_processing_method == 'count':
    #         seq = re.sub('[NKMRYSWBVHDX]', 'N', seq)
    #     k = self.k
    #     l = len(seq)
    #     lsh_k = self.lsh_k
    #     lshmer2id = self.lshmer2id
    #     res = np.zeros((l - k + 2, len(lshmer2id)))
    #     for

    def __call__(self, read, idx, *args, **kwargs):

        read = self.seq2kmer(read)

        return read


class ReadToOneHot(object):
    """Convert string to numpy array of integers."""
    def __init__(self):
        logger = logging.getLogger(self.__class__.__name__)

        self.nuc2int = {
            'A': 1,
            'C': 2,
            'T': 3,
            'G': 4,
            'N': 5,
            'R': 5,
            'Y': 5,
            'S': 5,
            'W': 5,
            'K': 5,
            'M': 5,
            'B': 5,
            'D': 5,
            'H': 5,
            'V': 5,
            '.': 5,
            '-': 5
        }

    def __call__(self, read, idx, *args, **kwargs):
        nuc2int = self.nuc2int
        x = np.array([nuc2int[c] for c in read])

        return x


class ToTensorWithView(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, dtype=None, view=None):
        self.dtype = dtype
        self.view = view

    def __call__(self, x, *args, **kwargs):

        x = torch.as_tensor(x, dtype=self.dtype)

        # C X L
        if self.view:
            x = x.view(self.view)

        return x


class SummarizeTargets(object):
    def __init__(self, list_num_classes, target_format='probs', eps=0.001):
        self.list_num_classes = list_num_classes
        self.total_num_of_classes = sum(list_num_classes)
        class_offset = torch.as_tensor([0] + list_num_classes[:-1])
        self.class_offset = torch.cumsum(class_offset, dim=0).view(1, -1)
        self.eps = eps
        assert target_format in ["probs", "logprobs", "counts"]
        conv_dict = {
            "probs": self.convert_batch_to_prob,
            "logprobs": self.convert_batch_to_log_prob,
            "counts": self.convert_batch_to_counts
        }
        self.conv_func = conv_dict[target_format]

    def convert_batch_to_counts(self, batch_labels):
        batch_labels = batch_labels + self.class_offset
        res_vector = torch.bincount(
            batch_labels.view(-1),
            minlength=self.total_num_of_classes).float()
        res_vector = torch.split(res_vector, self.list_num_classes)
        return list(res_vector)

    def convert_batch_to_prob(self, batch_labels):  # TODO: Add self.eps
        counts = self.convert_batch_to_counts(batch_labels)
        res = [(c + self.eps) / torch.sum(c + self.eps) for c in counts]

        return res

    def convert_batch_to_log_prob(self, batch_labels):
        counts = self.convert_batch_to_counts(batch_labels)
        res = [
            torch.log(c + self.eps) - torch.log(torch.sum(c + self.eps))
            for c in counts
        ]

        return res

    def __call__(self, y, *args, **kwargs):
        return self.conv_func(y)


class OneHot(object):
    """
    Converts array of size B x C x 1 x W to
    one-hot array of size B x C x V x W
    """
    def __init__(self, voc_size=6):
        self.voc_size = voc_size

    def __call__(self, x, *args, **kwargs):
        # Expects shape (batch_size, C, 1, W)
        shape = list(x.size())
        shape[2] = self.voc_size
        out = torch.zeros(shape, device=x.device)
        out.scatter_(2, x, 1)
        return out


class ToClassNumbers(object):
    def __init__(self):
        self.unique = []
        self.counts = []
        self.offset = 0

    def __call__(self, y, *args, **kwargs):
        '''
        Each column is processed separately.
        y is an array of size N x L where N is the sample
        size
        '''
        self.unique = []
        self.counts = []
        inverse = []
        for i in range(0, y.shape[1]):
            u, i, c = np.unique(y[:, i],
                                return_inverse=True,
                                return_counts=True)
            self.unique.append(u)
            self.counts.append(c)
            inverse.append(i)
        inverse = np.array(inverse).transpose()
        return inverse

    def revert(self, y):
        assert len(self.unique) != 0
        assert len(self.counts) != 0
        out = []
        for i in range(0, y.shape[1]):
            out.append(np.take(self.unique[self.offset + i], y[:, i]))
        return np.array(out).transpose()

    def get_tax(self, level, class_num):
        return self.unique[self.offset + level][class_num]

    def get_tax_all_levels(self, class_nums):
        assert len(class_nums) == len(self.unique)
        out = [self.get_tax(i, n) for i, n in enumerate(class_nums)]
        return np.array(out)

    def set_offset(self, offset):
        self.offset = offset


if __name__ == '__main__':
    # import time
    # input = torch.randint(6, (256, 1000))
    # input = input.view(256, 1, 1, -1)
    # oh = OneHot(6)
    #
    # start = time.time()
    # for i in range(1000):
    #     oh(input)
    # print(time.time() - start)

    tcn = ToClassNumbers()
    input = np.array([['25', '0'], ['9', '0'], ['25', '3'], ['30', '0']])
    print(input)
    x = tcn(input)
    print(x)
    print(len(tcn.counts[0]))
    print(tcn.revert(x))
    print(tcn.unique)
    print(tcn.get_tax(1, np.arange(2)))
    print(tcn.get_tax_all_levels([0, 0]))

    # c2n = GenomeToNoisyRead(rmin=5, rmax=10, p=0.1)
    # print(c2n('ACTGAAAAAAGG', 1))
