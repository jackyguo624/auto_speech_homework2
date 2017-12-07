import torch
from model import DNNModel
from torch.autograd import Variable
from torch import optim, nn
from torch.utils.data import DataLoader
from data_feeder import load_data, ASVDataSet
from torch.optim.lr_scheduler import MultiStepLR

# parameters
print_str = "*"*10 + "{}" + "*"*10
batch_size = 1024
num_epochs = 10


def use_cuda():
    is_cuda = torch.cuda.is_available()
    return is_cuda


def get_test_accuracy(dataset, net):
    net.eval()
    correct = 0
    for i, tmp in enumerate(dataset):
        data = Variable(tmp['data'])
        label = tmp['label']
        if use_cuda():
            data = data.cuda()
        predict = net(data)
        _, predict_label = torch.max(predict.data, 1)
        final_label = torch.sum(predict_label) / predict_label.size(0)
        label = torch.sum(label)
        if label == 0 and final_label < 0.5:
            correct += 1
        if label == 1 and final_label >= 0.5:
            correct += 1
    return correct / len(dataset)


def main():
    # loading train data
    print(print_str.format("Loading Data"))
    train_data, train_label = load_data("train", "data/protocol/ASVspoof2017_train.trn.txt", mode="train")
    train_dataset = ASVDataSet(train_data, train_label, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    dev_data, dev_label, dev_wav_ids = load_data("dev", "data/protocol/ASVspoof2017_dev.trl.txt", mode="test")
    dev_dataset = ASVDataSet(dev_data, dev_label, wav_ids=dev_wav_ids, mode="test")

    # test_data, test_label, test_wav_ids = load_data("eval", "data/protocol/ASVspoof2017_eval_v2_key.trl.txt", mode="test")
    # test_dataset = ASVDataSet(test_data, test_label, wav_ids=test_wav_ids, mode="test")

    model = DNNModel(input_dim=351, hidden_dim=2048, output_dim=2)
    if use_cuda():
        model = model.cuda()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    scheduler = MultiStepLR(optimizer, milestones=[3, 7], gamma=0.1)

    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        for i, tmp in enumerate(train_dataloader):
            data = Variable(tmp['data'])
            label = Variable(tmp['label'])
            if use_cuda():
                data, label = data.cuda(), label.cuda()

            optimizer.zero_grad()
            predict = model(data)
            loss = cross_entropy(predict, label.long().view(-1))

            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f,' % (
                    epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]), end="")

                # to test the dev data and test data
                dev_accuracy = get_test_accuracy(dev_dataset, model)
                test_accuracy = 0.0
                # test_accuracy = get_test_accuracy(test_dataset, model)
                print(" Dev Acc: %.2f Test Acc: %.2f" % (dev_accuracy, test_accuracy))


if __name__ == '__main__':
    main()
