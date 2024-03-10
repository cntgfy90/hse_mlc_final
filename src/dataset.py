import torch
from torch.utils.data import Dataset


class MLDataset(Dataset):
    def __init__(self, df, max_len, tokenizer, target_cols):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.target_cols = target_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df["description"][index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": ids.clone().detach().flatten(),
            "mask": mask.clone().detach().flatten(),
            "token_type_ids": token_type_ids.clone().detach().flatten(),
            "targets": torch.tensor(
                self.df[self.target_cols].values[index], dtype=torch.float
            ),
        }
