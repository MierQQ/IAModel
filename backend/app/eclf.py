import torch.nn as nn
from transformers import AutoModel, BertTokenizer


class EmotionClf(nn.Module):

    def __init__(self):
        super(EmotionClf, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
        self.bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
