from torch.utils.data import DataLoader
from data.dataset import dataset,data_split
from model import model
import torch.optim as optim
import torch
import wandb
import os

# Initialize wandb
os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = "offline"

wandb.init(project='model', name='best2')



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    result_data = dataset('data/a.csv',if_normalize=True)
    data_train, data_test = data_split(result_data, 0.9)
    train_data = DataLoader(data_train, batch_size=512, shuffle=True)
    test_data = DataLoader(data_test, batch_size=1, shuffle=True)

    epochs = 15000
    input_size = 5
    hidden_size = 256
    output_size = 7
    lr = 0.001

    model = model(input_size, hidden_size, output_size).to(device)
    optim = optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = 0
        train_corrects = 0
        num_batches = len(train_data)

        for i, (inputs, labels) in enumerate(train_data):
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)

            out = model(inputs)
            loss = loss_func(out, labels)
            train_loss += loss.item()

            # 计算train的准确率
            prediction = torch.max(out, 1)[1]
            train_corrects += torch.sum(prediction == labels).item()

            optim.zero_grad()
            loss.backward()
            optim.step()

        avg_train_loss = train_loss / num_batches
        train_acc = train_corrects / len(data_train)

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                total = len(data_test)
                test_corrects = 0

                for i, (data, labels) in enumerate(test_data):
                    data = data.to(device)
                    labels = labels.to(device)

                    outputs = model(data)
                    loss = loss_func(outputs, labels)
                    test_loss += loss.item()

                    prediction = torch.max(outputs, 1)[1]
                    test_corrects += torch.sum(prediction == labels).item()

                avg_test_loss = test_loss / len(test_data)
                test_acc = test_corrects / total
                print(
                    f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}, Train Acc: {train_acc}, Test Loss: {avg_test_loss}, Test Acc: {test_acc}")
                #保存模型
                torch.save(model.state_dict(), 'save/model.pt')
                # Log to wandb
                wandb.log({"Train Loss": avg_train_loss, "Train Acc": train_acc,
                           "Test Loss": avg_test_loss, "Test Acc": test_acc})
    wandb.finish()