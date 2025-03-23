import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ChatBotM(nn.Module):
    def __init__(self, in_s, out_s):
        super(ChatBotM, self).__init__()
        self.f1 = nn.Linear(in_s, 128)
        self.f2 = nn.Linear(128, 64)
        self.f3 = nn.Linear(64, out_s)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.f1(x))
        x = self.dropout(x)
        x = self.relu(self.f2(x))
        x = self.dropout(x)
        x = self.f3(x)  # Removed extra ReLU and Dropout here
        return x


class ChatBotA:
    def __init__(self, i_path):
        self.model = None
        self.i_path = i_path
        self.documents = []
        self.vocab = []
        self.i = []
        self.resps = {}
        self.X = []
        self.y = []

    @staticmethod
    def TockenizeAndLemmatize(text):
        lem = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lem.lemmatize(word.lower()) for word in words]
        return words

    def BagOfWords(self, words):
        return [1 if word in words else 0 for word in self.vocab]

    def parse_i(self):
        if os.path.exists(self.i_path):
            with open(self.i_path, 'r') as f:
                i_data = json.load(f)

            if "intents" not in i_data:
                raise KeyError("Invalid JSON structure. Missing 'intents' key.")

            intents_list = i_data["intents"]
            self.i = []
            self.documents = []
            self.resps = {}
            self.vocab = []

            for intent in intents_list:
                if intent['tag'] not in self.i:
                    self.i.append(intent['tag'])
                    self.resps[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_w = self.TockenizeAndLemmatize(pattern)
                    self.vocab.extend(pattern_w)
                    self.documents.append((pattern_w, intent['tag']))

            self.vocab = sorted(set(self.vocab))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.BagOfWords(words)
            intent_index = self.i.index(document[1])
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_M(self, b_size, lr, ep):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=b_size, shuffle=True)

        self.model = ChatBotM(self.X.shape[1], len(self.i))
        cr = nn.CrossEntropyLoss()
        opt = optim.Adam(self.model.parameters(), lr=lr)

        for epp in range(ep):
            run_loss = 0.0
            for b_X, b_y in loader:
                opt.zero_grad()
                outputs = self.model(b_X)
                loss = cr(outputs, b_y)
                loss.backward()
                opt.step()
                run_loss += loss.item()

            print(f"Epoch {epp + 1}: Loss: {run_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.i)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatBotM(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path))

    def process_message(self, input_message):
        words = self.TockenizeAndLemmatize(input_message)
        bag = self.BagOfWords(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predict = self.model(bag_tensor)
        predict_class_index = torch.argmax(predict, dim=1).item()
        predict_intent = self.i[predict_class_index]

        if self.resps[predict_intent]:
            return random.choice(self.resps[predict_intent])
        else:
            return None


if __name__ == '__main__':
    assistant = ChatBotA('intents.json')
    assistant.parse_i()
    assistant.prepare_data()
    assistant.train_M(b_size=8, lr=0.001, ep=200)

    assistant.save_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input("Enter your message: ")

        if message.lower() == 'quit':
            break

        response = assistant.process_message(message)
        if response:
            print(response)
        else:
            print("Sorry, I didn't understand that.")