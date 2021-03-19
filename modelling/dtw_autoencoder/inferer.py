import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


from model import ScenarioModel
import dataloader


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


def run():
    BATCH_SIZE = 4
    # real_train_loader, real_dev_loader = dataloader.get_real_train_test_loaders(
    #     batch_size=BATCH_SIZE, max_len=400
    # )

    (
        anomalous_train_loader,
        anomalous_dev_loader,
    ) = dataloader.get_anomalous_train_test_loaders(batch_size=BATCH_SIZE)

    inferer = ScenarioModelInferer("./epoch_399.pth")
    # breakpoint()

    print("Generating Anomalous vectors...")
    anomalous_vectors = []
    for i, input in tqdm(
        enumerate(anomalous_dev_loader), total=len(anomalous_dev_loader)
    ):
        if torch.cuda.is_available():
            input = input.type(torch.FloatTensor).cuda()
        embedding = inferer.get_embedding(input).squeeze()
        anomalous_vectors.append(embedding.detach().cpu().numpy())

    anomalous_vectors = np.vstack(anomalous_vectors)
    np.save("anomalous_tlsl_dev_embeddings2.npy", anomalous_vectors)

    print("Saved Anomalous vectors")

    # print("Generating Real data vectors...")
    # real_vectors = []
    # for i, input in tqdm(enumerate(real_dev_loader), total=len(real_dev_loader)):
    #     if torch.cuda.is_available():
    #         input = input.type(torch.FloatTensor).cuda()
    #     embedding = inferer.get_embedding(input).squeeze()
    #     real_vectors.append(embedding.detach().cpu().numpy())

    # real_vectors = np.vstack(real_vectors)
    # np.save("real_dev_embeddings.npy", real_vectors)
    # print("Saved Real data vectors")


if __name__ == "__main__":
    run()
