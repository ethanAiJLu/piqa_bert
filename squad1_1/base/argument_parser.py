import argparse
import os

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, description='base', **kwargs):
        super(ArgumentParser, self).__init__(description=description)

    def add_arguments(self):
        self.add_arguments("--start_step",type=int,default=0)
        self.add_arguments("--mode",type=str,default="train")

    def parse_args(self, **kwargs):
        args = super().parse_args()
        return args
