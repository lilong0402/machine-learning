import torch
from torch.utils.data import DataLoader
import MyDataset
import DataProcess
import pandas as pd
import mode
def test(model,test,batchsize):
    test_set = MyDataset.MyDataset(test.values, "test")
    test_data_loader= DataLoader(test_set,batch_size=batchsize,shuffle=False)
    model.eval().to('cuda')
    outputs = []
    with torch.no_grad():
        for data in test_data_loader:
            data = data.to('cuda')
            out = model(data)
            outputs.append(out)
    return outputs


if __name__=='__main__':

    batchsize = 1024
    pathtrain = "./train.csv"
    pathtest = "./test.csv"
    _, test_data, _, ids = DataProcess.dataprocess(pathtrain, pathtest)
    input_size = test_data.shape[1]


    model = mode.model(input_size,1).to('cuda')
    state_dict = torch.load('model6.pth')
    model.load_state_dict(state_dict)

    outputs = test(model, test_data, batchsize)
    outputs_tensor = torch.cat(outputs, dim=0).cpu().detach().numpy().flatten()
    df_results = pd.DataFrame({
        "id": ids,
        "Listening_Time_minutes": outputs_tensor  # 模型预测值
    })
    df_results.to_csv("test_predictions7.csv", index=False)
    print("预测结果已保存为 test_predictions7.csv")


