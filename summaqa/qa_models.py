 # code from huggingface see https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering

import torch
from transformers import BertTokenizer, BertForQuestionAnswering

class QA_Bert():
    def __init__(self):
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.SEP_id = self.tokenizer.encode('[SEP]')[0]
    
    def predict(self, question, text):
        
        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = self.tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(self.SEP_id) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.model(torch.tensor([input_ids]))

        start_scores = torch.functional.F.softmax(start_scores, -1) * torch.Tensor(token_type_ids)
        end_scores = torch.functional.F.softmax(end_scores, -1) * torch.Tensor(token_type_ids)

        start_values, start_indices = start_scores.topk(1)
        end_values, end_indices = end_scores.topk(1)

        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        asw = ' '.join(all_tokens[start_indices[0][0] : end_indices[0][0]+1])
        prob = start_values[0][0] * end_values[0][0]
        
        return asw, prob.item()