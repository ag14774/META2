from itertools import combinations
from pathlib import Path

import sourmash
from tqdm import tqdm
from utils.genome_utils import GenomeTools


def get_average_dist(file_list):
    file_list = [str(f) + '.sig' for f in file_list]
    signatures = [sourmash.load_one_signature(f) for f in file_list]
    dist = 0
    counter = 0
    for s1, s2 in combinations(signatures, 2):
        dist += s1.jaccard(s2)
        counter += 1
    return dist / counter


def create_signatures(file_list, ksize=21, verbose=False):
    file_list = [Path(str(f) + '.sig') for f in file_list]
    gt = GenomeTools()
    if verbose:
        file_list = tqdm(file_list, total=len(file_list))
    for f in file_list:
        if f.is_file():
            sig = sourmash.load_one_signature(str(f))
            if sig.minhash.ksize == ksize:
                continue
        minhash = sourmash.MinHash(n=1000, ksize=ksize)
        genome = gt.read_fasta(f.with_suffix(''))
        minhash.add_sequence(genome, True)
        sig = sourmash.SourmashSignature(minhash, name=f.stem)
        with f.open('wt') as handle:
            sourmash.save_signatures([sig], handle)


def create_signatures_all(dir, ksize=21, verbose=False):
    dir = Path(dir)
    file_list = dir.glob('*.fasta')
    create_signatures(file_list, ksize, verbose)


if __name__ == '__main__':
    genomes = ('/home/ageorgiou/eth/spring2019/thesis/data/'
               'refseq_prokaryota/genomes')

    create_signatures_all(genomes)
    res = get_average_dist([
        f"{genomes}/NC_018631.fasta", f"{genomes}/NC_012483.fasta",
        f"{genomes}/NZ_CP006777.fasta"
    ])
    print(res)
