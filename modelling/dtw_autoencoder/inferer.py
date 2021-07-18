import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sys

from model import ScenarioModel
import dataloader

from random_seed import RANDOM_SEED


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# run this file from the root folder -> `python modelling/dtw_autoencoder/inferer.py`
class ScenarioModelInferer:
    def __init__(self, model_file_path: str) -> None:
        super().__init__()
        model = nn.DataParallel(ScenarioModel())
        model.load_state_dict(torch.load(model_file_path))
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        self.model = model

    def get_embedding(self, input):
        with torch.no_grad():
            result = self.model.module.embedding(input)
        return result


def infer_and_get_embeddings(dataloader, inferer):
    result_vectors = []
    for i, input in tqdm(enumerate(dataloader), total=len(dataloader)):
        if torch.cuda.is_available():
            input = input.type(torch.FloatTensor).cuda()
        embedding = inferer.get_embedding(input).squeeze()
        result_vectors.append(embedding.detach().cpu().numpy())
    result_vectors = np.vstack(result_vectors)
    return result_vectors


def run(modelpath="epoch_400.pth"):
    BATCH_SIZE = 4
    NUM_SCENARIOS = 1000  # number of scenarios to evaluate against
    real_train_loader, real_dev_loader = dataloader.get_real_train_test_loaders(
        batch_size=BATCH_SIZE, max_len=NUM_SCENARIOS
    )

    (
        anomalous_train_loader,
        anomalous_dev_loader,
    ) = dataloader.get_anomalous_train_test_loaders(
        batch_size=BATCH_SIZE, max_len=NUM_SCENARIOS
    )

    inferer = ScenarioModelInferer(modelpath)

    print("Generating Anomalous vectors...")
    anomalous_vectors = infer_and_get_embeddings(anomalous_dev_loader, inferer)
    np.save("anomalous_dev_embeddings.npy", anomalous_vectors)
    print("Saved Anomalous vectors")

    print("Generating Real data vectors...")
    real_vectors = infer_and_get_embeddings(real_dev_loader, inferer)
    np.save("real_dev_embeddings.npy", real_vectors)
    print("Saved Real data vectors")

    # Can add inference calls on the training dataloaders here as well
    print("Generating Anomalous vectors...")
    anomalous_vectors = infer_and_get_embeddings(anomalous_train_loader, inferer)
    np.save("anomalous_train_embeddings.npy", anomalous_vectors)
    print("Saved Anomalous vectors")

    print("Generating Real data vectors...")
    real_vectors = infer_and_get_embeddings(real_train_loader, inferer)
    np.save("real_train_embeddings.npy", real_vectors)
    print("Saved Real data vectors")


if __name__ == "__main__":
    args = sys.argv[1:]
    run(*args)
