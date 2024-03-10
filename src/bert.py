import torch
from transformers import AutoModel


class MLBERT(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-cased")
        self.fc = torch.nn.Linear(768, n_classes)

    def forward(self, ids, mask, token_type_ids):
        output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        return self.fc(output.pooler_output)
