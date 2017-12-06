import torch
from model import DNNModel
from torch.autograd import Variable
from torch import optim, nn
from torch.utils.data import DataLoader
from data_feeder import load_data, ASVDataSet


# parameters
print_str = "*"*10 + "{}" + "*"*10
batch_size = 4


def use_cuda():
    is_cuda = torch.cuda.is_available()
    return is_cuda


def main():
    # loading train data
    print(print_str.format("Loading Data"))
    train_data, train_label = load_data("train", "data/protocol/ASVspoof2017_train.trn.txt", mode="train")
    train_dataset = ASVDataSet(train_data, train_label, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)

    dev_data, dev_label, dev_wav_ids = load_data("dev", "data/protocol/ASVspoof2017_dev.trl.txt", mode="test")
    dev_dataset = ASVDataSet(dev_data, dev_label, wav_ids=dev_wav_ids, mode="test")
    print("train data len: {} \t dev data len: {}".format(len(train_dataloader), len(dev_dataset)))
    model = DNNModel(input_dim=39, hidden_dim=4096, output_dim=2)
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

        # to test the dev data
        total_dev_loss = 0.
        for i, tmp in enumerate(dev_dataset):
            data = Variable(tmp['data'])
            label = Variable(tmp['label'])
            if use_cuda():
                data, label = data.cuda(), label.cuda()
            predict = model(data)
            loss = cross_entropy(predict, label.long().view(-1))
            total_dev_loss += loss.data[0]

        print("Epoch {} \t Train Loss: {} \t Dev Loss: {}".format(
            epoch, total_loss / len(train_dataloader), total_dev_loss / len(dev_dataset)))

if __name__ == '__main__':
    main()