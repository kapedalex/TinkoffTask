import numpy as np
import torch
import torch.nn.functional as func


class ModelWorker:

    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def evaluate(self, model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
        hidden = model.init_hidden()
        idx_input = [char_to_idx[char] for char in start_text]
        train = torch.LongTensor(idx_input).view(-1, 1, 1).to(self.device)
        predicted_text = start_text

        _, hidden = model(train, hidden)

        inp = train[-1].view(-1, 1, 1)

        for i in range(prediction_len):
            output, hidden = model(inp.to(self.device), hidden)
            output_logits = output.cpu().data.view(-1)
            p_next = func.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
            top_index = np.random.choice(len(char_to_idx), p=p_next)
            inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(self.device)
            predicted_char = idx_to_char[top_index]
            predicted_text += predicted_char

        return predicted_text