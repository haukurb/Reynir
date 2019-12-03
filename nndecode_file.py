#!/usr/bin/env python3
"""
    Reynir: Natural language processing for Icelandic

    Neural Network Query Client

    Copyright (C) 2018 Miðeind ehf.

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.
       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.


    This module implements a file translation client that connects to a
    middleware neural network server (see nnclient and
    nnserver/nnserver.py), which in turn connects to a TensorFlow model server.

    Usage:
        NN_PARSING_ENABLED=1 \
        NN_PARSING_HOST=$PHOST \
        NN_PARSING_PORT=$PPORT \
        python nndecode_file.py \
        -i=source.txt -o=outputs.txt \
        -t=parse
"""


import datetime
import logging
import os
import re
from subprocess import PIPE, run
import time
import itertools
import sys

from nnclient import ParsingClient, TranslateClient


_logging_info_fn = print
_RETRY_WAIT_TIME = 10  # seconds
_MAX_LINE_LENGTH_CHARS = 1500
_MAX_ENTRIES_IN_BATCH = 500
_MAX_RETRIES_IN_ROW = 2
_DEF_B_SIZE_LINES = 15
_DEF_B_SIZE_CHARS = 3000
_DEF_B_SIZE_TOKS = 1500
_DEFAULT_BATCH_SIZE = dict(
    lines=_DEF_B_SIZE_LINES, chars=_DEF_B_SIZE_CHARS, tokens=_DEF_B_SIZE_TOKS
)


def estimate_token_count(line):
    """ Estimates subtoken count in line as used in
        the tensor2tensor's SubwordTextEncoder. """
    pattern = r"([\w0-9]{1,6}|[^ \w0-9])"
    subtoken = re.compile(pattern, re.IGNORECASE)
    matches = subtoken.findall(line)
    return len(matches)


WEIGHT_FNS = dict(
    chars=lambda x: len(x), lines=lambda x: 1, tokens=estimate_token_count
)


class AvgRing:
    """ Ring buffer for keeping track of running averages """

    def __init__(self, maxsize):
        self._ring = dict()
        self.maxsize = maxsize
        self._cursor = 0
        self._begin = 0

    def put(self, numberlike, weight=1, times=1):
        self._ring[self._cursor] = (numberlike, weight, times)
        self._cursor = (self._cursor + 1) % self.maxsize

    @property
    def average(self):
        num = sum([(n * t) for (n, w, t) in self._ring.values()])
        denom = sum([t for (n, w, t) in self._ring.values()])
        return 0 if denom == 0 else num / denom

    @property
    def weighted_average(self):
        num = sum([(n * t) for (n, w, t) in self._ring.values()])
        denom = sum([w * t for (n, w, t) in self._ring.values()])
        return 0 if denom == 0 else num / denom


def count_lines_in_path(path):
    if not os.path.isfile(path):
        raise ValueError("Expected {path} to be a file but it is not".format(path=path))
    result = run(["wc", "-l", path], stdout=PIPE)
    line_count = result.stdout.decode("utf-8").split(" ", 1)[0]
    return int(line_count)


def get_completed(path):
    """ Assumes path points to a tab seperated file
        with indices in the first field.
        Returns the set of indices found. """
    if not os.path.isfile(path):
        return set()
    ids = []
    with open(path, "r") as f:
        for entry in f:
            idx, line = entry.split("\t", 1)
            ids.append(int(idx))
    return set(ids)


def reorder_by(lines, key_fn, completed=None, mixing_size=1000):
    """ Reorder line stream with key_fn as the key function using
        a mixing bucket of size mixing_size.
        Line indices contained in completed are skipped. """
    mixer = []
    for (idx, line) in enumerate(lines):
        if completed is not None and idx in completed:
            continue
        line = line.strip("\n")
        mixer.append((idx, line, key_fn(line)))
        if len(mixer) > mixing_size:
            _logging_info_fn("Bucketing...")
            for item in sorted(mixer, key=(lambda x: x[2])):
                yield item
            mixer = []
    if len(mixer) > 0:
        _logging_info_fn("Bucketing...")
        for item in sorted(mixer, key=(lambda x: x[2])):
            yield item


def padded_size(batch):
    w = [estimate_token_count(e[1]) for e in batch]
    size_padded = max(w) * len(w)
    return size_padded


def unpadded_size(batch):
    w = [estimate_token_count(e[1]) for e in batch]
    return sum(w)


def batch_generator(lines, completed, batch_size):
    """ Batch line stream by batching generator determined
        by estimated subtoken count such that size of batch (including padding)
        is less than batch_size.
        Line stream is reordered before batching. """
    batch = []
    accum = 0
    batch_width = 0
    last_weight = batch_size
    for (idx, line, weight) in reorder_by(
        lines, estimate_token_count, completed=completed
    ):
        if len(line) > _MAX_LINE_LENGTH_CHARS:
            line = ""
        curr_size = batch_width * len(batch)
        next_size = max(batch_width, weight) * (len(batch) + 1)
        if (
            curr_size >= batch_size
            or len(batch) >= _MAX_ENTRIES_IN_BATCH
            or (2 * weight < last_weight and batch)
        ):
            yield (batch, accum)
            batch = []
            batch_width, accum = 0, 0
        elif next_size >= batch_size:
            # next line would make batch too large
            if batch:
                yield (batch, accum)
                batch = []
                batch_width, accum = 0, 0
            if weight > batch_size:
                # line is too large
                end = len(line)
                end = int(end * 0.9)
                trunc_line = line[:end]
                while estimate_token_count(trunc_line) > batch_size:
                    end = int(end * 0.9)
                    trunc_line = line[:end]
                line = trunc_line
                weight = estimate_token_count(trunc_line)
        accum += weight
        last_weight = weight
        batch_width = max(batch_width, weight)
        batch.append((idx, line))
    if len(batch) > 0:
        yield (batch, accum)


def chain_split_batch(batch_num, batch, iterator):
    half = len(batch) // 2
    split1 = batch[:half]
    split1 = (batch_num, (split1, unpadded_size(split1)))
    split2 = batch[half:]
    split2 = (batch_num, (split2, unpadded_size(split2)))
    return itertools.chain([split1, split2], iterator)


def tsv_splitter(lines):
    """ Return line if it is only one field in tsv format, otherwise field 2"""
    line = next(lines)
    lines = itertools.chain([line], lines)
    if "\t" not in line:
        yield from lines
    else:
        for line in lines:
            segment_id, text, *rest = line.split("\t", 3)
            yield text


def translate_file(in_path, out_path, verb, batch_size):
    """ Translate file pointed to by in_path with translation task
        determined by verb. Input is batched such that the padded estimated
        subtoken count does not exceed batch_size.
        Results are saved to a file pointed to by out_path. """
    completed = get_completed(out_path)

    total_lines = count_lines_in_path(in_path)
    remaining_lines = total_lines - len(completed)
    ring = AvgRing(maxsize=10)

    _logging_info_fn("Translating {0}".format(in_path))
    _logging_info_fn("Output file is {0}".format(out_path))
    if remaining_lines <= 0:
        _logging_info_fn("Done")
        return
    _logging_info_fn("Batch size is {0} subtokens".format(batch_size))
    _logging_info_fn(
        "Currently {0}/{1} entries are done".format(len(completed), total_lines)
    )

    bad_responses_in_row = 0
    skipped = []
    offset = 0
    running = True
    _logging_info_fn("Submitting batches...")
    with open(in_path, "r") as in_file:
        line_gen = tsv_splitter(in_file)
        batches = enumerate(batch_generator(line_gen, completed, batch_size))
        begin_time = time.time()
        while running:
            batch_begin_btime = time.time()
            try:
                batch_num, (batch, b_weight) = next(batches)
                out_batch = translate_batch(batch, verb)
                out_batch = sorted(out_batch, key=lambda x: x[0])
            except (IOError) as e:
                import traceback

                traceback.print_exc()
                if bad_responses_in_row <= 4:
                    batches = chain_split_batch(batch_num, batch, batches)
                    time.sleep(1 + bad_responses_in_row * 5)
                    bad_responses_in_row += 1
                    continue
                else:
                    print("Exiting...")
                    sys.exit(0)
            except (TypeError, KeyError, ValueError) as e:
                # split batch in two in case its too big
                print("Batch size too large: splitting batch in two")
                if len(batch) < 2:
                    continue
                batches = chain_split_batch(batch_num, batch, batches)
                continue
            except StopIteration as e:
                running = False
                continue



            with open(out_path, "a") as out_file:
                for (idx, outputs, scores) in out_batch:
                    msg = "{0}\t{1}\t{2}\n".format(idx, outputs, scores)
                    out_file.write(msg)

                completed.update([entry[0] for entry in out_batch])
                remaining_lines -= len(out_batch)
                elaps = round(time.time() - batch_begin_btime, 2)
                s_per_ex = elaps / len(batch)
                ring.put(s_per_ex, weight=b_weight / len(batch), times=len(batch))

                msg = (
                    "Batch {batch_num:>4d}: {batch_lines:4d} lines (width {width:>3d}) in {elaps:>5.2f}s, "
                    "{batch_weight:>4d} subtokens, {ms_per_key:>6.2f} ms/subtoken, "
                    "{avg_ms_per_ex:>5.2f} ms/line"
                )
                _logging_info_fn(
                    msg.format(
                        batch_num=batch_num,
                        elaps=elaps,
                        batch_lines=len(batch),
                        ms_per_key=1000 * ring.weighted_average,
                        avg_ms_per_ex=1000 * ring.average,
                        batch_weight=b_weight,
                        width=max([estimate_token_count(e[1]) for e in batch]),
                    )
                )
            bad_responses_in_row = 0

    _logging_info_fn("Finished all batches")
    elaps_run_time = datetime.timedelta(0, int(time.time() - begin_time))
    _logging_info_fn("Total run time {0}".format(elaps_run_time))


def translate_batch(batch, verb):
    """ Send a single translation batch of translation task
        determined by verb. The host of the neural network server
        and middleware are determined by Settings.py and/or
        environment variables """
    ids, sents = zip(*batch)
    client = ParsingClient if verb == "parse" else TranslateClient
    result = client._request(sents)

    out_batch = []
    for idx, inst in zip(ids, result):
        outputs = inst["outputs"]
        scores = inst["scores"]
        out_batch.append((idx, outputs, scores))
    return out_batch


def main(in_path, out_path, verb, batch_size):
    translate_file(in_path, out_path, verb, batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Translate a file by sending incremental batches to a tensorflow model server "
    )

    parser.add_argument(
        "-i",
        dest="IN_FILE",
        type=str,
        required=True,
        help="File that contains the source text to be translated",
    )
    parser.add_argument(
        "-o",
        dest="OUT_FILE",
        type=str,
        required=True,
        help="File that will contain the output of the translation system",
    )
    parser.add_argument(
        "-t",
        dest="VERB",
        choices=["parse", "translate"],
        type=str,
        required=True,
        help="Type of translation task to be performed",
    )
    parser.add_argument(
        "-b",
        dest="BATCH_SIZE",
        type=int,
        required=False,
        help="Batch size defaults to {0} (padded) approximate subtokens".format(
            _DEF_B_SIZE_LINES
        ),
    )

    args = parser.parse_args()

    main(args.IN_FILE, args.OUT_FILE, args.VERB, args.BATCH_SIZE)
