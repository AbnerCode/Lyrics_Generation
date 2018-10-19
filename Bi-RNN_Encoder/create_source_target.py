#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : create_source_target.py
# @Author: harry
# @Date  : 18-8-12 下午1:32
# @Desc  : Description goes here

with open("data/full_data.txt", 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    # print(lines[0:3])
    with open("data/sources_full.txt", 'w') as sources:
        with open("data/targets_full.txt", 'w') as targets:
            for i in range(len(lines) - 1):
                sources.write(lines[i] + '\n')
                targets.write(lines[i + 1] + '\n')
