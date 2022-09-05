import torch

import Parser
import Pipeline

from Model import Model
from ModelWorker import ModelWorker
from Resources.Constants import Constants


class Generator(ModelWorker):
    def __init__(self):
        super().__init__()
        self.args = Parser.get_generate_args()
        self.sequence, self.char_to_idx, self.idx_to_char = Pipeline.get_pipelined_data()

    def generate(self):
        model = Model(input_size=len(self.idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
        model.load_state_dict(torch.load(Constants.SAVE_PATH_MODEL))

        print(self.evaluate(
            model,
            self.char_to_idx,
            self.idx_to_char,
            temp=0.3,
            prediction_len=self.args.length,
            start_text=self.args.prefix,
            )
        )
