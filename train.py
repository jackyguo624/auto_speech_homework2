import torch
from model import DNNModel
from torch.autograd import Variable
from torch import optim, nn
from torch.utils.data import DataLoader
from data_feeder import load_data, ASVDataSet


# parameters
batch_size = 256


def use_cuda():
    is_cuda = torch.cuda.is_available()
    return is_cuda


def main():
    # loading train data
    train_data, train_label, train_wav_ids = load_data("train", "data/protocol/ASVspoof2017_train.trn.txt")
    train_dataset = ASVDataSet(train_data, train_label, train_wav_ids)
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)

    model = DNNModel(input_dim=13, hidden_dim=1024, output_dim=2)
    if use_cuda():
        model = model.cuda()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)

    for epoch in range(100):
        total_loss = 0
        for i, tmp in enumerate(train_dataloader):
            data = Variable(tmp['data'])
            label = Variable(tmp['label'])
            if use_cuda():
                data, label = data.cuda(), label.cuda()

            optimizer.zero_grad()
            predict = model(data)
            loss = cross_entropy(predict, label.long().view(-1))
            total_loss += loss.data[0]

            loss.backward()
            optimizer.step()
        print("Epoch {}, Loss: {}".format(epoch, total_loss / len(train_dataloader)))


if __name__ == '__main__':
    main()