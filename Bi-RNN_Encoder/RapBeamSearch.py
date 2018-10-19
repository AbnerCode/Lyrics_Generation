#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : RapBeamSearch.py
# @Author: harry
# @Date  : 18-8-12 下午1:44
# @Desc  : Description goes here

import math
import operator


class RapBeamSearch(object):
    beams = []
    extending_beams = []
    stop_index = None
    encourage_index = []
    discourage_index = []

    class BeamNode:
        def __init__(self, prob, index, state, stop_index, force_stop=False):
            self.prob = prob
            self.log_prob = math.log2(prob)
            self.index = index
            self.state = state
            self.stopped = (self.index == stop_index)
            self.node_score = self.log_prob
            if force_stop:
                self.stopped = True

    def __init__(self, width=10, stop_index=35003, index2word=None, start_len=None, ensure_len=False, max_len=8):
        self.beam_width = width
        self.stop_index = stop_index
        self.index2word = index2word
        self.start_len = start_len
        self.ensure_len = ensure_len
        self.max_len = max_len

        # A reward for a sentence approaching target_long length, usually between 0.7-1.0.\
        # The length of the sentence user want
        self.reward_factor = 0.7
        self.target_long = 8

        self.beams = []
        start_node = self.BeamNode(prob=1, index=-1, state=None, stop_index=self.stop_index)
        self.beams.append([start_node])

    def add_prob(self, prob, index, state, beam_num):
        beam = self.beams[beam_num]
        force_stop = len(beam) > self.max_len
        node = self.BeamNode(prob, index, state, self.stop_index, force_stop)
        new_beam = beam + [node]
        self.extending_beams.append(new_beam)

    # empty extending_beams, keep the beams of top beam width by probability
    def shrink_beam(self):
        checking = self.extending_beams
        for beam in self.beams:
            if beam[-1].stopped:
                checking.append(beam)
        if len(checking) <= self.beam_width:
            self.beams = checking
            self.extending_beams = []
            return

        to_order = {}
        for i in range(len(checking)):
            prob = self.get_beam_score(checking[i])
            to_order[i] = prob
        to_order = sorted(to_order.items(), key=operator.itemgetter(1), reverse=True)
        to_order = to_order[:self.beam_width]
        self.beams = [checking[i] for (i, value) in to_order]
        self.extending_beams = []

    def get_best(self):
        best_beam = self.beams[0]
        best_prob = self.get_beam_score(self.beams[0])
        found = False
        for beam in self.beams:
            if self.ensure_len and self.get_beam_word_len(beam) != self.target_long:
                continue
            prob = self.get_beam_score(beam)
            if best_prob < prob:
                best_beam = beam
                best_prob = prob
                found = True
        if self.ensure_len and not found:
            for beam in self.beams:
                prob = self.get_beam_score(beam)
                if best_prob < prob:
                    best_beam = beam
                    best_prob = prob
        best_index = []
        for node in best_beam:
            best_index.append(node.index)
        return best_index[1:], best_beam[1:]

    def check_finished(self):
        for beam in self.beams:
            if not beam[-1].stopped:
                return False
        return True

    def get_beam_score(self, beam):
        prob = 0
        for node in beam:
            prob += node.node_score
        s_len = self.get_beam_word_len(beam)
        prob /= math.pow(s_len, 1)
        prob *= math.pow(abs(s_len - self.target_long) + 1, self.reward_factor)
        return prob

    def get_beam_word_len(self, beam):
        s_len = 0
        for node in beam:
            if self.index2word.get(node.index) is not None and node.index != self.stop_index:
                s_len += len(self.index2word[node.index])
        s_len += self.start_len
        return s_len


# usage
# Initial an object of beam search
beam_searcher = RapBeamSearch(width=self.beam_width,
                              stop_index=self.model.data.word_to_int['<EOS>'],
                              index2word=self.model.data.int_to_word,
                              start_len=len(last_word),
                              ensure_len=False)

ignore_words = ['<UNK>']
ignore_index = [self.model.data.word_to_int[word] for word in ignore_words]

while not beam_searcher.check_finished():
    beams = beam_searcher.beams
    beam_count = 0

    for beam in beams:
        beam_indexes = [node.index for node in beam]
        decode_x[0] = start_word + beam_indexes[1:]

        if beam[-1].stopped:
            beam_count += 1
            continue

        feed = {self.encode: encode_x,
                self.encode_length: [len(encode_x[0])],
                self.initial_state: new_state,
                self.decode_post_x: decode_x,
                self.decode_post_length: [len(decode_x[0])]}
        predict, state = self.sess.run([self.post_prediction, self.post_state], feed_dict=feed)

        sorted_probs = sort_prob(predict[-1])
        high_probs = []
        for item in sorted_probs:
            if item[0] not in ignore_index:
                high_probs.append(item)
            if len(high_probs) == self.beam_width:
                break
        for prob in high_probs:
            beam_searcher.add_prob(prob[1], prob[0], state, beam_count)
        beam_count += 1

    beam_searcher.shrink_beam()

best_line_index, best_beam = beam_searcher.get_best()
