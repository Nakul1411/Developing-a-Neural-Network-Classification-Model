# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="762" height="897" alt="image" src="https://github.com/user-attachments/assets/0ec163c7-034d-42ba-854e-c85fafed32d7" />

## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).
### STEP 2: 
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.
### STEP 3: 
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.
### STEP 4: 
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.
### STEP 5: 
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.
### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.

## PROGRAM
## NAME :NAKUL R 
## REG NO: 212223240102
```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)


    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs,labels in train_loader:
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)
```

### Dataset Information
<img width="1562" height="317" alt="image" src="https://github.com/user-attachments/assets/84cc4300-e48d-4676-89a0-77e21b3b3845" />

### OUTPUT
<img width="927" height="107" alt="image" src="https://github.com/user-attachments/assets/f441746d-b7e1-4e3a-aadd-06cbdc594c2a" />

## Confusion Matrix

<img width="937" height="793" alt="image" src="https://github.com/user-attachments/assets/7bba27f3-9b08-48ce-b9a0-639af01f81eb" />


## Classification Report

<img width="747" height="472" alt="image" src="https://github.com/user-attachments/assets/b564c200-26cc-4caa-8acf-889bede090a8" />

### New Sample Data Prediction
<img width="1105" height="158" alt="image" src="https://github.com/user-attachments/assets/54fe1583-9eb6-4723-b7b1-b210a36b33c5" />


## RESULT
This program has been executed successfully.
