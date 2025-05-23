import json
import collections
import os
import sys
import shelve
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from multiprocessing import Pool
import multiprocessing as mp
from random import random, randrange, randint, shuffle, choice
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from tokenizers import BertBasicTokenizer
import pickle
import transformers

parser = ArgumentParser()
parser.add_argument('--train_corpus', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--vocab_path', type=str, required=True)
parser.add_argument('--lowercase', action='store_true', help='Whether to do lowercase')
parser.add_argument('--do_whole_word_mask', action='store_true', help='Whether to use whole word masking rather than per-WordPiece masking.')

parser.add_argument('--num_workers', type=int, default=1, help='The number of workers to use to write the files')
parser.add_argument('--epochs_to_generate', type=int, default=1, help='Number of epochs of data to pregenerate')
parser.add_argument('--max_seq_len', type=int, default=128)
parser.add_argument('--short_seq_prob', type=float, default=0.1, help='Probability of making a short sentence as a training example')
parser.add_argument('--masked_lm_prob', type=float, default=0.15, help='Probability of masking each token for the LM task')
parser.add_argument('--max_predictions_per_seq', type=int, default=20, help='Maximum number of tokens to mask in each sequence')
parser.add_argument('--one_seq', action='store_true')

args = parser.parse_args()


class DocumentDatabase:
    def __init__(self):
        #self.document_shelf = shelve.open(str(args.output_dir / 'shelf.db'), flag='n', protocol=-1)
        self.document_shelf = [] #shelve.open(str(args.output_dir / 'shelf.db'), flag='n', protocol=-1)
        self.documents = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None

    def add_document(self, document):
        if not document:
            return
        current_idx = len(self.doc_lengths)
        #self.document_shelf[current_idx] = document
        self.document_shelf.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)

        assert sampled_doc_index != current_idx
        return self.document_shelf[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        return self.document_shelf[item]

    def __enter__(self):
        return self

    #def __exit__(self, exc_type, exc_val, traceback):
    #    pass
    #    #self.document_shelf.close()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    '''Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo.'''
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


MaskedLmInstance = collections.namedtuple('MaskedLmInstance', ['index', 'label'])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    '''Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables.'''
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith('##')):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = '[MASK]'
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    
    mask_indices, masked_token_labels = None, None
    if num_to_mask > 0:
        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        mask_indices = [p.index for p in masked_lms]
        masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


def create_instances_from_document(docs, doc_idx, doc_database, max_seq_length, short_seq_prob,
                                   masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list, bi_text=True):
    '''This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task.'''
    document = docs[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments 'A' and 'B' based on the actual 'sentences' provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []

                # Random next
                if bi_text and (len(current_chunk) == 1 or random() < 0.5) :
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # Sample a random document, with longer docs being sampled more frequently
                    random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)

                    random_start = randrange(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we 'put them back' so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                if not tokens_a or len(tokens_a) == 0:
                    tokens_a = ['.']

                if not tokens_b or len(tokens_b) == 0:
                    tokens_b = ['.']

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)

                instance = {
                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'is_random_next': is_random_next,
                    'masked_lm_positions': masked_lm_positions,
                    'masked_lm_labels': masked_lm_labels}

                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


docs_ = None
doc_base_ = None

def _init(docs, doc_base):
    global docs_
    global doc_base_
    docs_ = docs
    doc_base_ = doc_base


def _create_training_file(*a):
    #docs, vocab_list, args, epoch_num, bi_text = a
    rank, workers, doc_base, vocab_list, args, epoch_num, bi_text = a
    #doc_base = doc_base_
    size = len(doc_base) // workers
    if rank == workers - 1:
        docs = doc_base.document_shelf[rank*size:]
    else:
        docs = doc_base.document_shelf[rank*size:(rank+1)*size]
    results = []
    for doc_idx in range(len(docs)):
        if doc_idx % 10000 == 0:
            print(f'{doc_idx}/{len(docs)}')
        doc_instances = create_instances_from_document(
            docs, doc_idx, doc_base, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
            masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
            whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list, bi_text=bi_text)
        doc_instances = [json.dumps(instance) for instance in doc_instances]
        results.extend(doc_instances)
    return results


def create_training_file(docs, doc_database, vocab_list, args, epoch_num, bi_text=True):
    workers = 80
    size_group = len(docs) // workers
    #lines_group = []
    #for idx in range(workers):
    #    if idx != workers - 1:
    #        lines_group.append(docs[idx*size_group:(idx+1)*size_group])
    #    else:
    #        lines_group.append(docs[idx*size_group:])
    print('array_split')
    #with mp.get_context('forkserver').Pool(processes=workers, initializer=_init, initargs=(None, doc_database)) as pool:
    '''
    with mp.get_context('fork').Pool(processes=workers) as pool:
        #res = pool.map(_create_training_file,
        #                  [(lines, doc_database, vocab_list, args, epoch_num, bi_text) for lines in lines_group])
        #res = pool.starmap(_create_training_file,
        #                  [(lines, vocab_list, args, epoch_num, bi_text) for lines in lines_group])
        res = pool.starmap(_create_training_file,
                          [(lines, workers, doc_database, vocab_list, args, epoch_num, bi_text) for lines in range(workers)])
        pool.close()
    '''

    res = [_create_training_file(0, 1, doc_database, vocab_list, args, epoch_num, bi_text)]
    doc_instances = []
    for worker in res:
        doc_instances.extend(worker)

    epoch_filename = args.output_dir / 'epoch_{}.json'.format(epoch_num)
    num_instances = 0
    with epoch_filename.open('w') as epoch_file:
        for instance in doc_instances:
            epoch_file.write(instance + '\n')
            num_instances += 1

    metrics_filename = args.output_dir / 'epoch_{}_metrics.json'.format(epoch_num)
    with metrics_filename.open('w') as metrics_file:
        metrics = {'num_training_examples': num_instances, 'max_seq_len': args.max_seq_len}
        metrics_file.write(json.dumps(metrics))

    return epoch_filename, metrics_filename

def tokenize_lines(args):
    lines, tokenizer = args
    docs = []
    doc = []
    doc_num = 0
    for idx, line in enumerate(lines):
        if idx % 10000 == 0:
            print(f'{idx}/{len(lines)}')
        line = line.strip()
        if line == '':
            docs.append(doc)
            doc = []
            doc_num += 1
            #if doc_num % 1000 == 0:
            #    print('loaded {} docs!'.format(doc_num))
        else:
            tokens = tokenizer.tokenize(line)
            doc.append(tokens)
    if doc:
        docs.append(doc)
    return docs



def main():
    #if args.num_workers > 1 and args.reduce_memory:
    #    raise ValueError('Cannot use multiple workers while reducing memory')

    args.output_dir.mkdir(exist_ok=True)
    tokenizer = BertBasicTokenizer(args.lowercase, args.vocab_path)
    vocab_list = list(tokenizer.vocab_map.keys())
    doc_num = 0
    if True:
        #with DocumentDatabase() as docs:
        docs = DocumentDatabase()
        with args.train_corpus.open() as f:
            f = f.readlines()
            print(f'Loaded data: {len(f)} lines.')
            '''
            doc = []
            for line in tqdm(f, desc='Loading Dataset', unit=' lines'):
                line = line.strip()
                if line == '':
                    docs.add_document(doc)
                    doc = []
                    doc_num += 1
                    if doc_num % 100 == 0:
                        print('loaded {} docs!'.format(doc_num))
                else:
                    tokens = tokenizer.tokenize(line)
                    doc.append(tokens)
            if doc:
                docs.add_document(doc)  # If the last doc didn't end on a newline, make sure it still gets added
            '''
            workers = 120
            #lines_group = np.array_split(f, workers)
            size_group = len(f) // workers
            lines_group = []
            for idx in range(workers):
                if idx != workers - 1:
                    lines_group.append(f[idx*size_group:(idx+1)*size_group])
                else:
                    lines_group.append(f[idx*size_group:])
            print('array_split')
            with mp.get_context('fork').Pool(processes=80) as pool:
                t_lines = pool.map(tokenize_lines,
                                  [(lines, tokenizer) for lines in lines_group])
                pool.close()
            #t_lines = np.concatenate(t_lines, axis=0)
    
            doc_list = []
            for doc in t_lines:
                for d in doc:
                    docs.add_document(doc)
                    #doc_list.append(d)
    
        #pickle.dump(docs, open('docs.pkl', 'wb'))
        #pickle.dump(doc_list, open('doc_list.pkl', 'wb'))

    else:
        #docs = pickle.load(open('docs.pkl', 'rb'))
        #print('load docs done.')
        docs = DocumentDatabase()
        doc_list = pickle.load(open('doc_list.pkl', 'rb'))
        print('load doc done.')
        for doc in doc_list:
            docs.add_document(doc)
        del doc_list
        doc_list = []


    if len(docs) <= 1:
        exit('ERROR: No document breaks were found in the input file! These are necessary to allow the script to '
             'ensure that random NextSentences are not sampled from the same document. Please add blank lines to '
             'indicate breaks between documents in your input file. If your dataset does not contain multiple '
             'documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, '
             'sections or paragraphs.')

    if args.num_workers > 1:
        writer_workers = Pool(min(args.num_workers, args.epochs_to_generate))
        arguments = [(docs, vocab_list, args, idx) for idx in range(args.epochs_to_generate)]
        writer_workers.starmap(create_training_file, arguments)
    else:
        for epoch in trange(args.epochs_to_generate, desc='Epoch'):
            bi_text = True if not args.one_seq else False
            create_training_file(doc_list, docs, vocab_list, args, epoch, bi_text=bi_text)


if __name__ == '__main__':
    main()
