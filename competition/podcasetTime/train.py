import torch
import mode
import MyDataset
from torch.utils.data import DataLoader
import DataProcess
import torch.nn as nn

def train(data,test,label,model,loss_fn,batchsize=1024,epochs=10,lr=0.001):
    train_set = MyDataset.MyDataset(data.values, "train", label)
    test_set = MyDataset.MyDataset(test.values, "test")
    train_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batchsize, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    alloss=[]
    for epoch in range(epochs):
        model.train()
        loss_all = 0.0
        for data, target in train_loader:
            data, target = data.to("cuda"), target.to("cuda")
            optimizer.zero_grad()
            output = model(data)
            output = output.squeeze(1)
            loss = loss_fn(output, target)
            loss_all += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, train loss {loss_all / len(train_loader)}")
    return loss_all/len(train_loader)

if __name__ == "__main__":
    pathtrain = "./train.csv"
    pathtest = "./test.csv"
    data_train,test,train_labels,_ = DataProcess.dataprocess(pathtrain,pathtest)
    # train_set = MyDataset.MyDataset(data_train.values, "train", train_labels)
    #
    # train_loader = DataLoader(dataset=train_set, batch_size=1024, shuffle=True)
    # data,target = next(iter(train_loader))
    # print(data)
    # print(target)
    #
    # input_size = data_train.shape[1]
    # output_size = 1
    #
    # model = mode.model(input_size, output_size).to("cuda")
    # data = data.to("cuda")
    # output = model(data)
    # print(output)

    epochs = [10,15,20]
    lrs =[0.1,0.01,0.005,0.0001]
    batchsizes=[256,512,1024,2048,4096]
    train_best = 100000
    input_size = data_train.shape[1]
    output_size = 1
    loss_fn = nn.MSELoss()
    # model = None
    # for epoch in epochs:
    #     for batchsize in batchsizes:
    #         for lr in lrs:
    #             model = mode.model(input_size, output_size).to("cuda")
    #             train_loss = train(data_train,test,train_labels,model,loss_fn,batchsize,epoch,lr)
    #             if train_loss < train_best:
    #                 train_best = train_loss
    #                 torch.save(model.state_dict(), './model.pth')
    #             print(f"epoch:{epoch},batchsize:{batchsize},lr:{lr},train:{train_loss}")
    model = mode.model(input_size, output_size).to("cuda")
    train_loss = train(data_train, test, train_labels, model, loss_fn, 1024, 15, 0.001)
    print(train_loss)
    if train_loss < train_best:
        train_best = train_loss
        torch.save(model.state_dict(), './model7.pth')
    # print(f"epoch:{epoch},batchsize:{batchsize},lr:{lr},train:{train_loss}")
