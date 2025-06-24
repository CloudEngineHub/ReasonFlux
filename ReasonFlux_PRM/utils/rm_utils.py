import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

def make_step_rewards(logits, token_masks):
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

class RewardDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=8192):
        self.tokenizer = tokenizer
        self.samples = data
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["instruction"] + sample["output"]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "reward": torch.tensor(sample["reward"], dtype=torch.float),
        }

class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        print("loading base model")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.hidden_size = self.base_model.config.hidden_size
        print("init reward head")
        self.reward_head = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        with torch.set_grad_enabled(self.training):
            output = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = output.hidden_states[-1]

            last_token_indices = attention_mask.sum(dim=1) - 1
            final_hidden = hidden[torch.arange(hidden.size(0)), last_token_indices]

            final_hidden = final_hidden.to(self.reward_head.weight.device)
            
            reward = self.reward_head(final_hidden).squeeze(-1)
            reward = torch.sigmoid(reward)
            return reward
            
    def save_pretrained(self, path):
        torch.save(self.reward_head.state_dict(), f"{path}/reward_head.pt")
        self.base_model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path, auth_token):
        print("model = cls")
        model = cls(model_name=path)
        reward_head_path = hf_hub_download(
            repo_id=path,
            filename="reward_head.pt",
            repo_type="model",
            use_auth_token=auth_token
        )
        state = torch.load(reward_head_path, map_location=model.reward_head.weight.device)
        model.reward_head.load_state_dict(state)
        return model
