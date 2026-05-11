#!/usr/bin/env python

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2016, Johns Hopkins University (Author: Daniel Povey).
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script was copied from https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/lang/make_phone_lm.py
# with minor python3 related changes.

import argparse
import math
import sys
from collections import defaultdict

# note, this was originally based

parser = argparse.ArgumentParser(
    description="""
This script creates a language model that's intended to be used in modeling
phone sequences (either of sentences or of dictionary entries), although of
course it will work for any type of data.  The easiest way
to describe it is as a a Kneser-Ney language model (unmodified, with addition)
with a fixed discounting constant equal to 1, except with no smoothing of the
bigrams (and hence no unigram state).  This is (a) because we want to keep the
graph after context expansion small, (b) because languages tend to have
constraints on which phones can follow each other, and (c) in order to get valid
sequences of word-position-dependent phones so that lattice-align-words can
work.  It also includes have a special entropy-based pruning technique that
backs off the statistics of pruned n-grams to lower-order states.
This script reads lines from its standard input, each
consisting of a sequence of integer symbol-ids (which should be > 0),
representing the phone sequences of a sentence or dictionary entry.
This script outputs a backoff language model in FST format""",
    epilog="See also utils/lang/make_phone_bigram_lang.sh",
)


parser.add_argument(
    "--phone-disambig-symbol",
    type=int,
    required=False,
    help="Integer corresponding to an otherwise-unused "
    "phone-level disambiguation symbol (e.g. #5).  This is "
    "inserted at the beginning of the phone sequence and "
    "whenever we back off.",
)
parser.add_argument(
    "--ngram-order",
    type=int,
    default=4,
    choices=[2, 3, 4, 5, 6, 7],
    help="Order of n-gram to use (but see also --num-extra-states;" "the effective order after pruning may be less.",
)
parser.add_argument(
    "--num-extra-ngrams",
    type=int,
    default=20000,
    help="Target number of n-grams in addition to the n-grams in "
    "the bigram LM states which can't be pruned away.  n-grams "
    "will be pruned to reach this target.",
)
parser.add_argument(
    "--no-backoff-ngram-order",
    type=int,
    default=2,
    choices=[1, 2, 3, 4, 5],
    help="This specifies the n-gram order at which (and below which) "
    "no backoff or pruning should be done.  This is expected to normally "
    "be bigram, but for testing purposes you may want to set it to "
    "1.",
)
parser.add_argument(
    "--print-as-arpa",
    type=str,
    default="false",
    choices=["true", "false"],
    help="If true, print LM in ARPA format (default is to print "
    "as FST).  You must also set --no-backoff-ngram-order=1 or "
    "this is not allowed.",
)
parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help="Verbose level")

args = parser.parse_args()

if args.verbose >= 1:
    print(" ".join(sys.argv), file=sys.stderr)


class CountsForHistory(object):
    ## This class (which is more like a struct) stores the counts seen in a
    ## particular history-state.  It is used inside class NgramCounts.
    ## It really does the job of a dict from int to float, but it also
    ## keeps track of the total count.
    def __init__(self):
        # The 'lambda: defaultdict(float)' is an anonymous function taking no
        # arguments that returns a new defaultdict(float).
        self.word_to_count = defaultdict(int)
        self.total_count = 0

    def Words(self):
        return list(self.word_to_count.keys())

    def __str__(self):
        # e.g. returns ' total=12 3->4 4->6 -1->2'
        return " total={0} {1}".format(
            str(self.total_count),
            " ".join(["{0} -> {1}".format(word, count) for word, count in self.word_to_count.items()]),
        )

    ## Adds a certain count (expected to be integer, but might be negative).  If
    ## the resulting count for this word is zero, removes the dict entry from
    ## word_to_count.
    ## [note, though, that in some circumstances we 'add back' zero counts
    ## where the presence of n-grams would be structurally required by the arpa,
    ## specifically if a higher-order history state has a nonzero count,
    ## we need to structurally have the count there in the states it backs
    ## off to.
    def AddCount(self, predicted_word, count):
        self.total_count += count
        assert self.total_count >= 0
        old_count = self.word_to_count[predicted_word]
        new_count = old_count + count
        if new_count < 0:
            print("predicted-word={0}, old-count={1}, count={2}".format(predicted_word, old_count, count))
        assert new_count >= 0
        if new_count == 0:
            del self.word_to_count[predicted_word]
        else:
            self.word_to_count[predicted_word] = new_count


class NgramCounts(object):
    ## A note on data-structure.  Firstly, all words are represented as
    ## integers.  We store n-gram counts as an array, indexed by (history-length
    ## == n-gram order minus one) (note: python calls arrays "lists") of dicts
    ## from histories to counts, where histories are arrays of integers and
    ## "counts" are dicts from integer to float.  For instance, when
    ## accumulating the 4-gram count for the '8' in the sequence '5 6 7 8', we'd
    ## do as follows: self.counts[3][[5,6,7]][8] += 1.0 where the [3] indexes an
    ## array, the [[5,6,7]] indexes a dict, and the [8] indexes a dict.
    def __init__(self, ngram_order):
        assert ngram_order >= 2
        # Integerized counts will never contain negative numbers, so
        # inside this program, we use -3 and -2 for the BOS and EOS symbols
        # respectively.
        # Note: it's actually important that the bos-symbol is the most negative;
        # it helps ensure that we print the state with left-context <s> first
        # when we print the FST, and this means that the start-state will have
        # the correct value.
        self.bos_symbol = -3
        self.eos_symbol = -2
        # backoff_symbol is kind of a pseudo-word, it's used in keeping track of
        # the backoff counts in each state.
        self.backoff_symbol = -1
        self.total_num_words = 0  # count includes EOS but not BOS.
        self.counts = []
        for n in range(ngram_order):
            self.counts.append(defaultdict(lambda: CountsForHistory()))

    # adds a raw count (called while processing input data).
    # Suppose we see the sequence '6 7 8 9' and ngram_order=4, 'history'
    # would be (6,7,8) and 'predicted_word' would be 9; 'count' would be
    # 1.
    def AddCount(self, history, predicted_word, count):
        self.counts[len(history)][history].AddCount(predicted_word, count)

    # 'line' is a string containing a sequence of integer word-ids.
    # This function adds the un-smoothed counts from this line of text.
    def AddRawCountsFromLine(self, line):
        try:
            words = [self.bos_symbol] + [int(x) for x in line.split()] + [self.eos_symbol]
        except Exception:
            sys.exit("make_phone_lm.py: bad input line {0} (expected a sequence " "of integers)".format(line))

        for n in range(1, len(words)):
            predicted_word = words[n]
            history_start = max(0, n + 1 - args.ngram_order)
            history = tuple(words[history_start:n])
            self.AddCount(history, predicted_word, 1)
            self.total_num_words += 1

    def AddRawCountsFromStandardInput(self):
        lines_processed = 0
        while True:
            line = sys.stdin.readline()
            if line == "":
                break
            self.AddRawCountsFromLine(line)
            lines_processed += 1
        if lines_processed == 0 or args.verbose > 0:
            print(
                "make_phone_lm.py: processed {0} lines of input".format(lines_processed),
                file=sys.stderr,
            )

    # This backs off the counts by subtracting 1 and assigning the subtracted
    # count to the backoff state.  It's like a special case of Kneser-Ney with D
    # = 1.  The optimal D would likely be something like 0.9, but we plan to
    # later do entropy-pruning, and the remaining small counts of 0.1 would
    # essentially all get pruned away anyway, so we don't lose much by doing it
    # like this.
    def ApplyBackoff(self):
        # note: in the normal case where args.no_backoff_ngram_order == 2 we
        # don't do backoff for history-length = 1 (i.e. for bigrams)... this is
        # a kind of special LM where we're not going to back off to unigram,
        # there will be no unigram.
        if args.verbose >= 1:
            initial_num_ngrams = self.GetNumNgrams()
        for n in reversed(list(range(args.no_backoff_ngram_order, args.ngram_order))):
            this_order_counts = self.counts[n]
            for hist, counts_for_hist in this_order_counts.items():
                backoff_hist = hist[1:]
                backoff_counts_for_hist = self.counts[n - 1][backoff_hist]
                this_discount_total = 0
                for word in counts_for_hist.Words():
                    counts_for_hist.AddCount(word, -1)
                    # You can interpret the following line as incrementing the
                    # count-of-counts for the next-lower order.  Note, however,
                    # that later when we remove n-grams, we'll also add their
                    # counts to the next-lower-order history state, so the
                    # resulting counts won't strictly speaking be
                    # counts-of-counts.
                    backoff_counts_for_hist.AddCount(word, 1)
                    this_discount_total += 1
                counts_for_hist.AddCount(self.backoff_symbol, this_discount_total)

        if args.verbose >= 1:
            # Note: because D == 1, we completely back off singletons.
            print(
                "make_phone_lm.py: ApplyBackoff() reduced the num-ngrams from "
                "{0} to {1}".format(initial_num_ngrams, self.GetNumNgrams()),
                file=sys.stderr,
            )

    # This function prints out to stderr the n-gram counts stored in this
    # object; it's used for debugging.
    def Print(self, info_string):
        print(info_string, file=sys.stderr)
        # these are useful for debug.
        total = 0.0
        total_excluding_backoff = 0.0
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                print(str(hist) + str(counts_for_hist), file=sys.stderr)
                total += counts_for_hist.total_count
                total_excluding_backoff += counts_for_hist.total_count
                if self.backoff_symbol in counts_for_hist.word_to_count:
                    total_excluding_backoff -= counts_for_hist.word_to_count[self.backoff_symbol]
        print(
            "total count = {0}, excluding backoff = {1}".format(total, total_excluding_backoff),
            file=sys.stderr,
        )

    def GetHistToStateMap(self):
        # This function, called from PrintAsFst, returns a map from
        # history to integer FST-state.
        hist_to_state = dict()
        fst_state_counter = 0
        for n in range(0, args.ngram_order):
            for hist in self.counts[n].keys():
                hist_to_state[hist] = fst_state_counter
                fst_state_counter += 1
        return hist_to_state

    # Returns the probability of word 'word' in history-state 'hist'.
    # If 'word' is self.backoff_symbol, returns the backoff prob
    # of this history-state.
    # Returns None if there is no such word in this history-state, or this
    # history-state does not exist.
    def GetProb(self, hist, word):
        if len(hist) >= args.ngram_order or not hist in self.counts[len(hist)]:
            return None
        counts_for_hist = self.counts[len(hist)][hist]
        total_count = float(counts_for_hist.total_count)
        if not word in counts_for_hist.word_to_count:
            print(
                "make_phone_lm.py: no prob for {0} -> {1} " "[no such count]".format(hist, word),
                file=sys.stderr,
            )
            return None
        prob = float(counts_for_hist.word_to_count[word]) / total_count
        if len(hist) > 0 and word != self.backoff_symbol and self.backoff_symbol in counts_for_hist.word_to_count:
            prob_in_backoff = self.GetProb(hist[1:], word)
            backoff_prob = float(counts_for_hist.word_to_count[self.backoff_symbol]) / total_count
            try:
                prob += backoff_prob * prob_in_backoff
            except Exception:
                sys.exit("problem, hist is {0}, word is {1}".format(hist, word))
        return prob

    def PruneEmptyStates(self):
        # Removes history-states that have no counts.

        # It's possible in principle for history-states to have no counts and
        # yet they cannot be pruned away because a higher-order version of the
        # state exists with nonzero counts, so we have to keep track of this.
        protected_histories = set()

        states_removed_per_hist_len = [0] * args.ngram_order

        for n in reversed(list(range(args.no_backoff_ngram_order, args.ngram_order))):
            num_states_removed = 0
            for hist, counts_for_hist in self.counts[n].items():
                l = len(counts_for_hist.word_to_count)
                assert l > 0 and self.backoff_symbol in counts_for_hist.word_to_count
                if l == 1 and not hist in protected_histories:  # only the backoff symbol has a count.
                    del self.counts[n][hist]
                    num_states_removed += 1
                else:
                    # if this state was not pruned away, then the state that
                    # it backs off to may not be pruned away either.
                    backoff_hist = hist[1:]
                    protected_histories.add(backoff_hist)
            states_removed_per_hist_len[n] = num_states_removed
        if args.verbose >= 1:
            print(
                "make_phone_lm.py: in PruneEmptyStates(), num states removed for "
                "each history-length was: " + str(states_removed_per_hist_len),
                file=sys.stderr,
            )

    def EnsureStructurallyNeededNgramsExist(self):
        # makes sure that if an n-gram like (6, 7, 8) -> 9 exists,
        # then counts exist for (7, 8) -> 9 and (8,) -> 9.  It does so
        # by adding zero counts where such counts were absent.
        # [note: () -> 9 is guaranteed anyway by the backoff method, if
        # we have a unigram state].
        if args.verbose >= 1:
            num_ngrams_initial = self.GetNumNgrams()
        for n in reversed(list(range(args.no_backoff_ngram_order, args.ngram_order))):

            for hist, counts_for_hist in self.counts[n].items():
                # This loop ensures that if we have an n-gram like (6, 7, 8) -> 9,
                # then, say, (7, 8) -> 9 and (8) -> 9 exist.
                reduced_hist = hist
                for m in reversed(list(range(args.no_backoff_ngram_order, n))):
                    reduced_hist = reduced_hist[1:]  # shift an element off
                    # the history.
                    counts_for_backoff_hist = self.counts[m][reduced_hist]
                    for word in counts_for_hist.word_to_count.keys():
                        counts_for_backoff_hist.word_to_count[word] += 0
                # This loop ensures that if we have an n-gram like (6, 7, 8) -> 9,
                # then, say, (6, 7) -> 8 and (6) -> 7 exist.  This will be needed
                # for FST representations of the ARPA LM.
                reduced_hist = hist
                for m in reversed(list(range(args.no_backoff_ngram_order, n))):
                    this_word = reduced_hist[-1]
                    reduced_hist = reduced_hist[:-1]  # pop an element off the
                    # history
                    counts_for_backoff_hist = self.counts[m][reduced_hist]
                    counts_for_backoff_hist.word_to_count[this_word] += 0
        if args.verbose >= 1:
            print(
                "make_phone_lm.py: in EnsureStructurallyNeededNgramsExist(), "
                "added {0} n-grams".format(self.GetNumNgrams() - num_ngrams_initial),
                file=sys.stderr,
            )

    # This function prints the estimated language model as an FST.
    def PrintAsFst(self, word_disambig_symbol):
        # n is the history-length (== order + 1).  We iterate over the
        # history-length in the order 1, 0, 2, 3, and then iterate over the
        # histories of each order in sorted order.  Putting order 1 first
        # and sorting on the histories
        # ensures that the bigram state with <s> as the left context comes first.
        # (note: self.bos_symbol is the most negative symbol)

        # History will map from history (as a tuple) to integer FST-state.
        hist_to_state = self.GetHistToStateMap()

        for n in [1, 0] + list(range(2, args.ngram_order)):
            this_order_counts = self.counts[n]
            # For order 1, make sure the keys are sorted.
            keys = this_order_counts.keys() if n != 1 else sorted(this_order_counts.keys())
            for hist in keys:
                word_to_count = this_order_counts[hist].word_to_count
                this_fst_state = hist_to_state[hist]

                for word in word_to_count.keys():
                    # work out this_cost.  Costs in OpenFst are negative logs.
                    this_cost = -math.log(self.GetProb(hist, word))

                    if word > 0:  # a real word.
                        next_hist = hist + (word,)  # appending tuples
                        while not next_hist in hist_to_state:
                            next_hist = next_hist[1:]
                        next_fst_state = hist_to_state[next_hist]
                        print(this_fst_state, next_fst_state, word, word, this_cost)
                    elif word == self.eos_symbol:
                        # print final-prob for this state.
                        print(this_fst_state, this_cost)
                    else:
                        assert word == self.backoff_symbol
                        backoff_fst_state = hist_to_state[hist[1 : len(hist)]]
                        print(
                            this_fst_state,
                            backoff_fst_state,
                            word_disambig_symbol,
                            0,
                            this_cost,
                        )

    # This function returns a set of n-grams that cannot currently be pruned
    # away, either because a higher-order form of the same n-gram already exists,
    # or because the n-gram leads to an n-gram state that exists.
    # [Note: as we prune, we remove any states that can be removed; see that
    # PruneToIntermediateTarget() calls PruneEmptyStates().

    def GetProtectedNgrams(self):
        ans = set()
        for n in range(args.no_backoff_ngram_order + 1, args.ngram_order):
            for hist, counts_for_hist in self.counts[n].items():
                # If we have an n-gram (6, 7, 8) -> 9, the following loop will
                # add the backed-off n-grams (7, 8) -> 9 and (8) -> 9 to
                # 'protected-ngrams'.
                reduced_hist = hist
                for _ in reversed(list(range(args.no_backoff_ngram_order, n))):
                    reduced_hist = reduced_hist[1:]  # shift an element off
                    # the history.

                    for word in counts_for_hist.word_to_count.keys():
                        if word != self.backoff_symbol:
                            ans.add(reduced_hist + (word,))
                # The following statement ensures that if we are in a
                # history-state (6, 7, 8), then n-grams (6, 7, 8) and (6, 7) are
                # protected.  This assures that the FST states are accessible.
                reduced_hist = hist
                for _ in reversed(list(range(args.no_backoff_ngram_order, n))):
                    ans.add(reduced_hist)
                    reduced_hist = reduced_hist[:-1]  # pop an element off the
                    # history
        return ans

    def PruneNgram(self, hist, word):
        counts_for_hist = self.counts[len(hist)][hist]
        assert word != self.backoff_symbol and word in counts_for_hist.word_to_count
        count = counts_for_hist.word_to_count[word]
        del counts_for_hist.word_to_count[word]
        counts_for_hist.word_to_count[self.backoff_symbol] += count
        # the next call adds the count to the symbol 'word' in the backoff
        # history-state, and also updates its 'total_count'.
        self.counts[len(hist) - 1][hist[1:]].AddCount(word, count)

    # The function PruningLogprobChange is the same as the same-named
    # function in float-counts-prune.cc in pocolm.  Note, it doesn't access
    # any class members.

    # This function computes the log-likelihood change (<= 0) from backing off
    # a particular symbol to the lower-order state.
    # The value it returns can be interpreted as a lower bound the actual log-likelihood
    # change.  By "the actual log-likelihood change" we mean of data generated by
    # the model itself before making the change, then modeled with the changed model
    # [and comparing the log-like with the log-like before changing the model].  That is,
    # it's a K-L divergence, but with the caveat that we don't normalize by the
    # overall count of the data, so it's a K-L divergence multiplied by the training-data
    # count.

    #  'count' is the count of the word (call it 'a') in this state.  It's an integer.
    #  'discount' is the discount-count in this state (represented as the count
    #         for the symbol self.backoff_symbol).  It's an integer.
    #  [note: we don't care about the total-count in this state, it cancels out.]
    #  'backoff_count' is the count of word 'a' in the lower-order state.
    #                 [actually it is the augmented count, treating any
    #                  extra probability from even-lower-order states as
    #                  if it were a count].  It's a float.
    #  'backoff_total' is the total count in the lower-order state.  It's a float.
    def PruningLogprobChange(self, count, discount, backoff_count, backoff_total):
        if count == 0:
            return 0.0

        assert discount > 0 and backoff_total >= backoff_count and backoff_total >= 0.99 * discount

        # augmented_count is like 'count', but with the extra count for symbol
        # 'a' due to backoff included.
        augmented_count = count + discount * backoff_count / backoff_total

        # We imagine a phantom symbol 'b' that represents all symbols other than
        # 'a' appearing in this history-state that are accessed via backoff.  We
        # treat these as being distinct symbols from the same symbol if accessed
        # not-via-backoff.  (Treating same symbols as distinct gives an upper bound
        # on the divergence).  We also treat them as distinct from the same symbols
        # that are being accessed via backoff from other states.  b_count is the
        # observed count of symbol 'b' in this state (the backed-off count is
        # zero).  b_count is also the count of symbol 'b' in the backoff state.
        # Note: b_count will not be negative because backoff_total >= backoff_count.
        b_count = discount * ((backoff_total - backoff_count) / backoff_total)
        assert b_count >= -0.001 * backoff_total

        # We imagine a phantom symbol 'c' that represents all symbols other than
        # 'a' and 'b' appearing in the backoff state, which got there from
        # backing off other states (other than 'this' state).  Again, we imagine
        # the symbols are distinct even though they may not be (i.e. that c and
        # b represent disjoint sets of symbol, even though they might not really
        # be disjoint), and this gives us an upper bound on the divergence.
        c_count = backoff_total - backoff_count - b_count
        assert c_count >= -0.001 * backoff_total

        # a_other is the count of 'a' in the backoff state that comes from
        # 'other sources', i.e. it was backed off from history-states other than
        # the current history state.
        a_other_count = backoff_count - discount * backoff_count / backoff_total
        assert a_other_count >= -0.001 * backoff_count

        # the following sub-expressions are the 'new' versions of certain
        # quantities after we assign the total count 'count' to backoff.  it
        # increases the backoff count in 'this' state, and also the total count
        # in the backoff state, and the count of symbol 'a' in the backoff
        # state.
        new_backoff_count = backoff_count + count  # new count of symbol 'a' in
        # backoff state
        new_backoff_total = backoff_total + count  # new total count in
        # backoff state.
        new_discount = discount + count  # new discount-count in 'this' state.

        # all the loglike changes below are of the form
        # count-of-symbol * log(new prob / old prob)
        # which can be more conveniently written (by canceling the denominators),
        # count-of-symbol * log(new count / old count).

        # this_a_change is the log-like change of symbol 'a' coming from 'this'
        # state.  bear in mind that
        # augmented_count = count + discount * backoff_count / backoff_total,
        # and the 'count' term is zero in the numerator part of the log expression,
        # because symbol 'a' is completely backed off in 'this' state.
        this_a_change = augmented_count * math.log(
            (new_discount * new_backoff_count / new_backoff_total) / augmented_count
        )

        # other_a_change is the log-like change of symbol 'a' coming from all
        # other states than 'this'.  For speed reasons we don't examine the
        # direct (non-backoff) counts of symbol 'a' in all other states than
        # 'this' that back off to the backoff state-- it would be slower.
        # Instead we just treat the direct part of the prob for symbol 'a' as a
        # distinct symbol when it comes from those other states... as usual,
        # doing so gives us an upper bound on the divergence.
        other_a_change = a_other_count * math.log(
            (new_backoff_count / new_backoff_total) / (backoff_count / backoff_total)
        )

        # b_change is the log-like change of phantom symbol 'b' coming from
        # 'this' state (and note: it only comes from this state, that's how we
        # defined it).
        # note: the expression below could be more directly written as a
        # ratio of pseudo-counts as follows, by converting the backoff probabilities
        # into pseudo-counts in 'this' state:
        #  b_count * logf((new_discount * b_count / new_backoff_total) /
        #                 (discount * b_count / backoff_total),
        # but we cancel b_count to give us the expression below.
        b_change = b_count * math.log((new_discount / new_backoff_total) / (discount / backoff_total))

        # c_change is the log-like change of phantom symbol 'c' coming from
        # all other states that back off to the backoff sate (and all prob. mass of
        # 'c' comes from those other states).  The expression below could be more
        # directly written as a ratio of counts, as c_count * logf((c_count /
        # new_backoff_total) / (c_count / backoff_total)), but we simplified it to
        # the expression below.
        c_change = c_count * math.log(backoff_total / new_backoff_total)

        ans = this_a_change + other_a_change + b_change + c_change
        # the answer should not be positive.
        assert ans <= 0.0001 * (count + discount + backoff_count + backoff_total)
        if args.verbose >= 4:
            print(
                "pruning-logprob-change for {0},{1},{2},{3} is {4}".format(
                    count, discount, backoff_count, backoff_total, ans
                ),
                file=sys.stderr,
            )
        return ans

    def GetLikeChangeFromPruningNgram(self, hist, word):
        counts_for_hist = self.counts[len(hist)][hist]
        counts_for_backoff_hist = self.counts[len(hist) - 1][hist[1:]]
        assert word != self.backoff_symbol and word in counts_for_hist.word_to_count
        count = counts_for_hist.word_to_count[word]
        discount = counts_for_hist.word_to_count[self.backoff_symbol]
        backoff_total = counts_for_backoff_hist.total_count
        # backoff_count is a pseudo-count: it's like the count of 'word' in the
        # backoff history-state, but adding something to account for further
        # levels of backoff.
        try:
            backoff_count = self.GetProb(hist[1:], word) * backoff_total
        except Exception:
            print(
                "problem getting backoff count: hist = {0}, word = {1}".format(hist, word),
                file=sys.stderr,
            )
            sys.exit(1)

        return self.PruningLogprobChange(float(count), float(discount), backoff_count, float(backoff_total))

    # note: returns loglike change per word.
    def PruneToIntermediateTarget(self, num_extra_ngrams):
        protected_ngrams = self.GetProtectedNgrams()
        initial_num_extra_ngrams = self.GetNumExtraNgrams()
        num_ngrams_to_prune = initial_num_extra_ngrams - num_extra_ngrams
        assert num_ngrams_to_prune > 0

        num_candidates_per_order = [0] * args.ngram_order
        num_pruned_per_order = [0] * args.ngram_order

        # like_change_and_ngrams this will be a list of tuples consisting
        # of the likelihood change as a float and then the words of the n-gram
        # that we're considering pruning,
        # e.g. (-0.164, 7, 8, 9)
        # meaning that pruning the n-gram (7, 8) -> 9 leads to
        # a likelihood change of -0.164.  We'll later sort this list
        # so we can prune the n-grams that made the least-negative
        # likelihood change.
        like_change_and_ngrams = []
        for n in range(args.no_backoff_ngram_order, args.ngram_order):
            for hist, counts_for_hist in self.counts[n].items():
                for word, count in counts_for_hist.word_to_count.items():
                    if word != self.backoff_symbol:
                        if not hist + (word,) in protected_ngrams:
                            like_change = self.GetLikeChangeFromPruningNgram(hist, word)
                            like_change_and_ngrams.append((like_change,) + hist + (word,))
                            num_candidates_per_order[len(hist)] += 1

        like_change_and_ngrams.sort(reverse=True)

        if num_ngrams_to_prune > len(like_change_and_ngrams):
            print(
                "make_phone_lm.py: aimed to prune {0} n-grams but could only "
                "prune {1}".format(num_ngrams_to_prune, len(like_change_and_ngrams)),
                file=sys.stderr,
            )
            num_ngrams_to_prune = len(like_change_and_ngrams)

        total_loglike_change = 0.0

        for i in range(num_ngrams_to_prune):
            total_loglike_change += like_change_and_ngrams[i][0]
            hist = like_change_and_ngrams[i][1:-1]  # all but 1st and last elements
            word = like_change_and_ngrams[i][-1]  # last element
            num_pruned_per_order[len(hist)] += 1
            self.PruneNgram(hist, word)

        like_change_per_word = total_loglike_change / self.total_num_words

        if args.verbose >= 1:
            effective_threshold = (
                like_change_and_ngrams[num_ngrams_to_prune - 1][0] if num_ngrams_to_prune >= 0 else 0.0
            )
            print(
                "Pruned from {0} ngrams to {1}, with threshold {2}.  Candidates per order were {3}, "
                "num-ngrams pruned per order were {4}.  Like-change per word was {5}".format(
                    initial_num_extra_ngrams,
                    initial_num_extra_ngrams - num_ngrams_to_prune,
                    "%.4f" % effective_threshold,
                    num_candidates_per_order,
                    num_pruned_per_order,
                    like_change_per_word,
                ),
                file=sys.stderr,
            )

        if args.verbose >= 3:
            print(
                "Pruning: like_change_and_ngrams is:\n"
                + "\n".join([str(x) for x in like_change_and_ngrams[:num_ngrams_to_prune]])
                + "\n-------- stop pruning here: ----------\n"
                + "\n".join([str(x) for x in like_change_and_ngrams[num_ngrams_to_prune:]]),
                file=sys.stderr,
            )
            self.Print(
                "Counts after pruning to num-extra-ngrams={0}".format(initial_num_extra_ngrams - num_ngrams_to_prune)
            )

        self.PruneEmptyStates()
        if args.verbose >= 3:
            ngram_counts.Print("Counts after removing empty states [inside pruning algorithm]:")
        return like_change_per_word

    def PruneToFinalTarget(self, num_extra_ngrams):
        # prunes to a specified num_extra_ngrams.  The 'extra_ngrams' refers to
        # the count of n-grams of order higher than args.no_backoff_ngram_order.
        # We construct a sequence of targets that gradually approaches
        # this value.  Doing it iteratively like this is a good way
        # to deal with the fact that sometimes we can't prune a certain
        # n-gram before certain other n-grams are pruned (because
        # they lead to a state that must be kept, or an n-gram exists
        # that backs off to this n-gram).

        current_num_extra_ngrams = self.GetNumExtraNgrams()

        if num_extra_ngrams >= current_num_extra_ngrams:
            print(
                "make_phone_lm.py: not pruning since target num-extra-ngrams={0} is >= "
                "current num-extra-ngrams={1}".format(num_extra_ngrams, current_num_extra_ngrams),
                file=sys.stderr,
            )
            return

        target_sequence = [num_extra_ngrams]
        # two final iterations where the targets differ by factors of 1.1,
        # preceded by two iterations where the targets differ by factors of 1.2.
        for this_factor in [1.1, 1.2]:
            for n in range(0, 2):
                if int((target_sequence[-1] + 1) * this_factor) < current_num_extra_ngrams:
                    target_sequence.append(int((target_sequence[-1] + 1) * this_factor))
        # then change in factors of 1.3
        while True:
            this_factor = 1.3
            if int((target_sequence[-1] + 1) * this_factor) < current_num_extra_ngrams:
                target_sequence.append(int((target_sequence[-1] + 1) * this_factor))
            else:
                break

        target_sequence = list(set(target_sequence))  # only keep unique targets.
        target_sequence.sort(reverse=True)

        print(
            "make_phone_lm.py: current num-extra-ngrams={0}, pruning with "
            "following sequence of targets: {1}".format(current_num_extra_ngrams, target_sequence),
            file=sys.stderr,
        )
        total_like_change_per_word = 0.0
        for target in target_sequence:
            total_like_change_per_word += self.PruneToIntermediateTarget(target)

        if args.verbose >= 1:
            print(
                "make_phone_lm.py: K-L divergence from pruning (upper bound) is " "%.4f" % total_like_change_per_word,
                file=sys.stderr,
            )

    # returns the number of n-grams on top of those that can't be pruned away
    # because their order is <= args.no_backoff_ngram_order.
    def GetNumExtraNgrams(self):
        ans = 0
        for hist_len in range(args.no_backoff_ngram_order, args.ngram_order):
            # note: hist_len + 1 is the actual order.
            ans += self.GetNumNgrams(hist_len)
        return ans

    def GetNumNgrams(self, hist_len=None):
        ans = 0
        if hist_len is None:
            for hist_len in range(args.ngram_order):
                # note: hist_len + 1 is the actual order.
                ans += self.GetNumNgrams(hist_len)
            return ans
        else:
            for counts_for_hist in self.counts[hist_len].values():
                ans += len(counts_for_hist.word_to_count)
                if self.backoff_symbol in counts_for_hist.word_to_count:
                    ans -= 1  # don't count the backoff symbol, it doesn't produce
                    # its own n-gram line.
            return ans

    # this function, used in PrintAsArpa, converts an integer to
    # a string by either printing it as a string, or for self.bos_symbol
    # and self.eos_symbol, printing them as "<s>" and "</s>" respectively.
    def IntToString(self, i):
        if i == self.bos_symbol:
            return "<s>"
        elif i == self.eos_symbol:
            return "</s>"
        else:
            assert i != self.backoff_symbol
            return str(i)

    def PrintAsArpa(self):
        # Prints out the FST in ARPA format.
        assert args.no_backoff_ngram_order == 1  # without unigrams we couldn't
        # print as ARPA format.

        print("\\data\\")
        for hist_len in range(args.ngram_order):
            # print the number of n-grams.  Add 1 for the 1-gram
            # section because of <s>, we print -99 as the prob so we
            # have a place to put the backoff prob.
            print(
                "ngram {0}={1}".format(
                    hist_len + 1,
                    self.GetNumNgrams(hist_len) + (1 if hist_len == 0 else 0),
                )
            )

        print("")

        for hist_len in range(args.ngram_order):
            print("\\{0}-grams:".format(hist_len + 1))

            # print fake n-gram for <s>, for its backoff prob.
            if hist_len == 0:
                backoff_prob = self.GetProb((self.bos_symbol,), self.backoff_symbol)
                if backoff_prob != None:
                    print("-99\t<s>\t{0}".format("%.5f" % math.log10(backoff_prob)))

            for hist in self.counts[hist_len].keys():
                for word in self.counts[hist_len][hist].word_to_count.keys():
                    if word != self.backoff_symbol:
                        prob = self.GetProb(hist, word)
                        assert prob != None and prob > 0
                        backoff_prob = self.GetProb((hist) + (word,), self.backoff_symbol)
                        line = "{0}\t{1}".format(
                            "%.5f" % math.log10(prob),
                            " ".join(self.IntToString(x) for x in hist + (word,)),
                        )
                        if backoff_prob != None:
                            line += "\t{0}".format("%.5f" % math.log10(backoff_prob))
                        print(line)
            print("")
        print("\\end\\")


ngram_counts = NgramCounts(args.ngram_order)
ngram_counts.AddRawCountsFromStandardInput()

if args.verbose >= 3:
    ngram_counts.Print("Raw counts:")
ngram_counts.ApplyBackoff()
if args.verbose >= 3:
    ngram_counts.Print("Counts after applying Kneser-Ney discounting:")
ngram_counts.EnsureStructurallyNeededNgramsExist()
if args.verbose >= 3:
    ngram_counts.Print("Counts after adding structurally-needed n-grams (1st time):")
ngram_counts.PruneEmptyStates()
if args.verbose >= 3:
    ngram_counts.Print("Counts after removing empty states:")
ngram_counts.PruneToFinalTarget(args.num_extra_ngrams)

ngram_counts.EnsureStructurallyNeededNgramsExist()
if args.verbose >= 3:
    ngram_counts.Print("Counts after adding structurally-needed n-grams (2nd time):")


if args.print_as_arpa == "true":
    ngram_counts.PrintAsArpa()
else:
    if args.phone_disambig_symbol is None:
        sys.exit("make_phone_lm.py: --phone-disambig-symbol must be provided (unless " "you are writing as ARPA")
    ngram_counts.PrintAsFst(args.phone_disambig_symbol)


## Below are some little test commands that can be used to look at the detailed stats
## for a kind of sanity check.
# test comand:
# (echo 6 7 8 4; echo 7 8 9; echo 7 8; echo 7 4; echo 8 4 ) | utils/lang/make_phone_lm.py --phone-disambig-symbol=400  --verbose=3
#  (echo 6 7 8 4; echo 7 8 9; echo 7 8; echo 7 4; echo 8 4 ) | utils/lang/make_phone_lm.py --phone-disambig-symbol=400  --verbose=3 --num-extra-ngrams=0
# (echo 6 7 8 4; echo 6 7 ) | utils/lang/make_phone_lm.py --print-as-arpa=true --no-backoff-ngram-order=1  --verbose=3


## The following shows how we created some data suitable to do comparisons with
## other language modeling toolkits.  Note: we're running in a configuration
## where --no-backoff-ngram-order=1 (i.e. we have a unigram LM state) because
## it's the only way to get perplexity calculations and to write an ARPA file.
##
# cd egs/tedlium/s5_r2
# . ./path.sh
# mkdir -p lm_test
# ali-to-phones exp/tri3/final.mdl "ark:gunzip -c exp/tri3/ali.*.gz|" ark,t:-  | awk '{$1 = ""; print}' > lm_test/phone_seqs
# wc lm_test/phone_seqs
# 92464  8409563 27953288 lm_test/phone_seqs
# head -n 20000 lm_test/phone_seqs > lm_test/train.txt
# tail -n 1000 lm_test/phone_seqs > lm_test/test.txt

## This shows make_phone_lm.py with the default number of extra-lm-states (20k)
## You have to have SRILM on your path to ger perplexities [note: it should be on the
## path if you installed it and you sourced the tedlium s5b path.sh, as above.]
# utils/lang/make_phone_lm.py --print-as-arpa=true --no-backoff-ngram-order=1 --verbose=1 < lm_test/train.txt > lm_test/arpa_pr20k
# ngram -order 4 -unk -lm lm_test/arpa_pr20k -ppl lm_test/test.txt
# file lm_test/test.txt: 1000 sentences, 86489 words, 3 OOVs
# 0 zeroprobs, logprob= -80130.1 ppl=*8.23985* ppl1= 8.44325
# on training data: 0 zeroprobs, logprob= -1.6264e+06 ppl= 7.46947 ppl1= 7.63431

## This shows make_phone_lm.py without any pruning (make --num-extra-ngrams very large).
# utils/lang/make_phone_lm.py --print-as-arpa=true --num-extra-ngrams=1000000 --no-backoff-ngram-order=1 --verbose=1 < lm_test/train.txt > lm_test/arpa
# ngram -order 4 -unk -lm lm_test/arpa -ppl lm_test/test.txt
# file lm_test/test.txt: 1000 sentences, 86489 words, 3 OOVs
# 0 zeroprobs, logprob= -74976 ppl=*7.19459* ppl1= 7.36064
# on training data: 0 zeroprobs, logprob= -1.44198e+06 ppl= 5.94659 ppl1= 6.06279

## This is SRILM without pruning (c.f. the 7.19 above, it's slightly better).
# ngram-count -text lm_test/train.txt -order 4 -kndiscount2 -kndiscount3 -kndiscount4 -interpolate -lm lm_test/arpa_srilm
# ngram -order 4 -unk -lm lm_test/arpa_srilm -ppl lm_test/test.txt
# file lm_test/test.txt: 1000 sentences, 86489 words, 3 OOVs
# 0 zeroprobs, logprob= -74742.2 ppl= *7.15044* ppl1= 7.31494


## This is SRILM with a pruning beam tuned to get 20k n-grams above unigram
##  (c.f. the 8.23 above, it's a lot worse).
# ngram-count -text lm_test/train.txt -order 4 -kndiscount2 -kndiscount3 -kndiscount4 -interpolate -prune 1.65e-05 -lm lm_test/arpa_srilm.pr1.65e-5
# the model has 20249 n-grams above unigram [c.f. our 20k]
# ngram -order 4 -unk -lm lm_test/arpa_srilm.pr1.65e-5 -ppl lm_test/test.txt
# file lm_test/test.txt: 1000 sentences, 86489 words, 3 OOVs
# 0 zeroprobs, logprob= -86803.7 ppl=*9.82202* ppl1= 10.0849


## This is pocolm..
## Note: we have to hold out some of the training data as dev to
## estimate the hyperparameters, but we'll fold it back in before
## making the final LM. [--fold-dev-into=train]
# mkdir -p lm_test/data/text
# head -n 1000 lm_test/train.txt > lm_test/data/text/dev.txt
# tail -n +1001 lm_test/train.txt > lm_test/data/text/train.txt
## give it a 'large' num-words so it picks them all.
# export PATH=$PATH:../../../tools/pocolm/scripts
# train_lm.py --num-word=100000 --fold-dev-into=train lm_test/data/text 4 lm_test/data/lm_unpruned
# get_data_prob.py lm_test/test.txt lm_test/data/lm_unpruned/100000_4.pocolm
## compute-probs: average log-prob per word was -1.95956 (perplexity = *7.0962*) over 87489 words.
## Note: we can compare this perplexity with 7.15 with SRILM and 7.19 with make_phone_lm.py.

#   pruned_lm_dir=${lm_dir}/${num_word}_${order}_prune${threshold}.pocolm
# prune_lm_dir.py --target-num-ngrams=20100 lm_test/data/lm_unpruned/100000_4.pocolm lm_test/data/lm_unpruned/100000_4_pr20k.pocolm
# get_data_prob.py lm_test/test.txt lm_test/data/lm_unpruned/100000_4_pr20k.pocolm
## compute-probs: average log-prob per word was -2.0409 (perplexity = 7.69757) over 87489 words.
## note: the 7.69 can be compared with 9.82 from SRILM and 8.23 from pocolm.
## format_arpa_lm.py lm_test/data/lm_unpruned/100000_4_pr20k.pocolm | head
## .. it has 20488 n-grams above unigram.  More than 20k but not enough to explain the difference
## .. in perplexity.

## OK... if I reran after modifying prune_lm_dir.py to comment out the line
## 'steps += 'EM EM'.split()' which adds the two EM stages per step, and got the
## perplexity again, I got the following:
## compute-probs: average log-prob per word was -2.09722 (perplexity = 8.14353) over 87489 words.
## .. so it turns out the E-M is actually important.
