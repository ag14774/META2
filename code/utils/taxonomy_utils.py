import json
import logging
import tarfile
import urllib

import numpy as np
from Bio import Entrez


class TaxTree(object):
    def __init__(self, taxfile, use_merged=False):
        self.id2parent = {}
        self.id2tax = {}
        self.id2children = {}
        self.tax2ids = {}
        self.id2dist = {}

        tf = tarfile.open(taxfile)

        # Load merged.dmp
        if use_merged:
            merged = tf.extractfile('merged.dmp').read().decode(
                'utf-8').strip()
            merged = merged.split('\n')
            self.merged = {}
            for line in merged:
                line = line.split('|')
                self.merged[line[0].strip()] = line[1].strip()

        # Load nodes.dmp
        nodes = tf.extractfile('nodes.dmp').read().decode('utf-8').strip()
        nodes = nodes.split('\n')

        for line in nodes:
            line = line.split('|')
            childid = line[0].strip()
            parentid = line[1].strip()
            tax = line[2].strip()

            self.id2tax[childid] = tax

            if use_merged:
                try:
                    childid = self.merged[childid]
                except KeyError:
                    pass
                try:
                    parentid = self.merged[parentid]
                except KeyError:
                    pass

            self.id2parent[childid] = parentid

        for child, parent in self.id2parent.items():
            try:
                self.id2children[parent].append(child)
            except KeyError:
                self.id2children[parent] = [child]

        for id, tax in self.id2tax.items():
            try:
                self.tax2ids[tax].append(id)
            except KeyError:
                self.tax2ids[tax] = [id]

        # Load names.dmp
        names = tf.extractfile('names.dmp').read().decode('utf-8').strip()
        names = names.split('\n')

        for line in names:
            line = line.split('|')

    def is_leaf(self, id):
        if id in self.id2children:
            return False
        else:
            return True

    def get_children(self, id):
        try:
            return self.id2children[id]
        except KeyError:
            return []

    def get_parent(self, id):
        return self.id2parent[id]

    def get_ids_at_level(self, level):
        return self.tax2ids[level]

    def get_all_leaves(self, id):
        stack = [id]
        leafs = []
        while stack:
            curr = stack.pop()
            if self.is_leaf(curr):
                leafs.append(curr)
            else:
                for child in self.get_children(curr):
                    stack.append(child)
        return leafs

    def ensure_levels_exist(self,
                            ordered_level_names=[
                                'phylum', 'class', 'order', 'family', 'genus',
                                'species'
                            ]):
        ordered_level_names_dict = {
            g: i
            for i, g in enumerate(ordered_level_names)
        }
        stack = ['1']
        while stack:
            parent = stack.pop()
            if not self.is_leaf(parent):
                children = self.get_children(parent).copy()
                for child in children:
                    stack.append(child)

                    if parent == '1' or self.id2tax[
                            parent] == ordered_level_names[-1]:
                        continue

                    curr = child
                    curr_pos = ordered_level_names_dict[self.id2tax[curr]]
                    parent_pos = ordered_level_names_dict[self.id2tax[parent]]

                    while parent_pos < (curr_pos - 1):
                        # Remove current connetion
                        self.id2parent.pop(curr)
                        self.id2children[parent].remove(curr)

                        # Add new dummy node and connections
                        new_curr = f'-{curr}'
                        new_pos = curr_pos - 1
                        if new_curr not in self.id2children[parent]:
                            self.id2children[parent].append(new_curr)
                        self.id2parent[new_curr] = parent

                        self.id2parent[curr] = new_curr
                        try:
                            if curr not in self.id2children[new_curr]:
                                self.id2children[new_curr].append(curr)
                        except KeyError:
                            self.id2children[new_curr] = [curr]

                        self.id2tax[new_curr] = ordered_level_names[new_pos]
                        if new_curr not in self.tax2ids[
                                ordered_level_names[new_pos]]:
                            self.tax2ids[ordered_level_names[new_pos]].append(
                                new_curr)

                        curr = new_curr
                        curr_pos = ordered_level_names_dict[self.id2tax[curr]]

    def trim_tree(self, leaves_to_keep, levels_to_keep=None):
        ids_to_keep = set()
        ids_to_keep.add('1')  # Always keep the root
        for id in leaves_to_keep:
            curr_id = id
            while curr_id != '1':
                ids_to_keep.add(curr_id)
                curr_id = self.get_parent(curr_id)

        new_tax2ids = {}
        for id in list(self.id2parent):
            if id not in ids_to_keep:
                parent_id = self.id2parent.pop(id)
                try:
                    self.id2children[parent_id].remove(id)
                    if not self.id2children[parent_id]:
                        self.id2children.pop(parent_id)
                except KeyError:
                    pass
                self.id2tax.pop(id)
                try:
                    self.id2children.pop(id)
                except KeyError:
                    pass
            else:
                tax = self.id2tax[id]
                try:
                    new_tax2ids[tax].append(id)
                except KeyError:
                    new_tax2ids[tax] = [id]
        self.tax2ids = new_tax2ids
        self.id2dist = {}

        if levels_to_keep:
            new_id2parent = {}
            new_id2children = {}
            new_id2tax = {}
            new_tax2ids = {}

            new_id2tax['1'] = self.id2tax['1']
            new_tax2ids[self.id2tax['1']] = ['1']
            for id in leaves_to_keep:
                parent_id = '1'
                ids = self.collect_path_ordered(id, categories=levels_to_keep)

                for child_id in ids:
                    new_id2parent[child_id] = parent_id
                    try:
                        if child_id not in new_id2children[parent_id]:
                            new_id2children[parent_id].append(child_id)
                    except KeyError:
                        new_id2children[parent_id] = [child_id]

                    new_id2tax[child_id] = self.id2tax[child_id]
                    try:
                        if child_id not in new_tax2ids[self.id2tax[child_id]]:
                            new_tax2ids[self.id2tax[child_id]].append(child_id)
                    except KeyError:
                        new_tax2ids[self.id2tax[child_id]] = [child_id]

                    parent_id = child_id

            self.id2parent = new_id2parent
            self.id2children = new_id2children
            self.id2tax = new_id2tax
            self.tax2ids = new_tax2ids

    def collect_path(
        self,
        start_id,
        categories=['phylum', 'class', 'order', 'family', 'genus', 'species']):
        positions = {}
        for i, cat in enumerate(categories):
            positions[cat] = i

        result = np.zeros(len(categories) + 1)
        curr_id = start_id
        while curr_id != '1':
            parent_id = self.id2parent[curr_id]
            curr_cat = self.id2tax[curr_id]
            if curr_cat in positions:
                curr_id = curr_id.strip('-')
                result[positions[curr_cat]] = curr_id
            curr_id = parent_id

        result[-1] = start_id
        return result

    def collect_path_ordered(
        self,
        start_id,
        categories=['phylum', 'class', 'order', 'family', 'genus', 'species']):
        categories = set(categories)
        collected = set()

        result = []
        curr_id = start_id
        while curr_id != '1':
            parent_id = self.id2parent[curr_id]
            curr_cat = self.id2tax[curr_id]
            if curr_cat in categories and curr_cat not in collected:
                result.append(curr_id)
                collected.add(curr_cat)
            curr_id = parent_id

        if result[0] != start_id:
            result = [start_id] + result
        return result[::-1]


class TaxonomyTools(object):
    """
    Tools for parsing taxonomy information from NCBI taxdump.tar.gz file
    """
    def __init__(self, ncbi_email='your-email@domain.com', ncbi_api=None):
        self.ncbi_email = ncbi_email
        self.ncbi_api = ncbi_api
        Entrez.email = self.ncbi_email
        if self.ncbi_api:
            Entrez.api_key = self.ncbi_api
        self.logger = logging.getLogger(self.__class__.__name__)

        self._tree = None

    def _get_tax_id_method1(self, ncbi_id):
        try:
            h = Entrez.elink(dbfrom="nucleotide",
                             db="taxonomy",
                             linkname="nuccore_taxonomy",
                             id=ncbi_id,
                             retmode='json')
            result = json.load(h)
            h.close()
        except urllib.error.HTTPError:
            return False
        try:
            return str(result["linksets"][0]["linksetdbs"][0]["links"][0])
        except KeyError:
            return False

    def _get_tax_id_method2(self, ncbi_id):
        try:
            h = Entrez.esummary(db="nucleotide", id=ncbi_id, retmode='json')
            result = json.load(h)
            h.close()
        except urllib.error.HTTPError:
            return False
        try:
            uid = result["result"]["uids"][0]
            return str(result["result"][uid]["taxid"])
        except KeyError:
            return False

    def get_tax_id(self, ncbi_id):
        taxid = self._get_tax_id_method1(ncbi_id)
        if taxid:
            return taxid

        return self._get_tax_id_method2(ncbi_id)

    def get_assembly_id_from_tax_id(self, tax_id):
        try:
            h = Entrez.elink(dbfrom="taxonomy",
                             db="assembly",
                             linkname="taxonomy_assembly",
                             id=tax_id,
                             retmode='json')
            result = json.load(h)
            h.close()
        except urllib.error.HTTPError:
            return False
        try:
            return str(result["linksets"][0]["linksetdbs"][0]["links"][0])
        except KeyError:
            return False

    def get_assembly_status(self, assembly_id):
        try:
            h = Entrez.esummary(db="assembly", id=assembly_id, retmode='json')
            result = json.load(h)
            h.close()
        except urllib.error.HTTPError:
            return False
        try:
            return str(result["result"][assembly_id]["assemblystatus"])
        except KeyError:
            return False

    def assembly_has_complete_genome(self, assembly_id):
        status = self.get_assembly_status(assembly_id)
        if status == "Complete Genome" or status == "Chromosome":
            return True
        else:
            return False

    def get_accessions_from_assembly_id(self, assembly_id):
        try:
            h = Entrez.elink(dbfrom="assembly",
                             db="nucleotide",
                             linkname="assembly_nuccore_refseq",
                             id=assembly_id,
                             retmode='json',
                             idtype='acc')
            result = json.load(h)
            h.close()
        except urllib.error.HTTPError:
            return []
        try:
            return result["linksets"][0]["linksetdbs"][0]["links"]
        except KeyError:
            return []
        except IndexError:
            return []

    def get_genome_type_from_accession(self, accession_id):
        try:
            h = Entrez.esummary(db="nucleotide",
                                id=accession_id,
                                retmode='json')
            result = json.load(h)
            h.close()
        except urllib.error.HTTPError:
            return False
        try:
            uid = result["result"]["uids"][0]
            return str(result["result"][uid]["genome"])
        except KeyError:
            return False

    def is_genome_from_chromosome(self, accession_id):
        genome_type = self.get_genome_type_from_accession(accession_id)
        return genome_type == 'chromosome'

    def load_taxfile(self, taxfile, use_merged=False, cache_tree=False):
        if self._tree:
            return self._tree
        self._tree = TaxTree(taxfile, use_merged=use_merged)
        return self._tree


if __name__ == '__main__':
    tt = TaxonomyTools()
    taxid = tt.get_tax_id('NC_012782')
    print(taxid)
    assemblyid = tt.get_assembly_id_from_tax_id(taxid)
    print(assemblyid)
    print(tt.get_assembly_status(assemblyid))
    print(tt.get_accessions_from_assembly_id(assemblyid))
    print()
    tt.load_taxfile('/home/ageorgiou/Downloads/taxdump.tar.gz')
    tree = tt.tree
    # print(tree.collect_path('66845'))
    leaves = tree.get_all_leaves('33958')
    for taxid in leaves:
        assembly_id = tt.get_assembly_id_from_tax_id(taxid)
        if assembly_id:
            if tt.assembly_has_complete_genome(assembly_id):
                accessions = tt.get_accessions_from_assembly_id(assembly_id)
                print()
                for acc in accessions:
                    print(acc)
                    print(tt.get_genome_type_from_accession(acc))
