import torch 
import torch.optim as optim
from schrodinger import SchrodingerPINN

#We can however create changes as per our requirement
layers = [2,50,50,50,1]
m = 0.5
x = torch.linspace(-5,5,100)
t = torch.linspace(0,2,100)
x_train , t_train = torch.meshgrid(x,t,indexing='ij')
x_train = x_train.reshape(-1,1)
t_train = t_train.reshape(-1,1)
epochs = 1000

def main():
    model = SchrodingerPINN(layers,m)
    optimizer = optim.Adam(model.parameters(),lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss_function(x_train,t_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

if __name__ == "__main__":
    main()