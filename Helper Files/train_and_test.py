
models_save_path = Path('Saved Models')
if not models_save_path.exists():
  models_save_path.mkdir(parents=True, exist_ok=True)


def train_step(model:nn.Module,
               opt:torch.optim,
               loss_fn:nn.Module,
               train_dataloader:torch.utils.data.DataLoader,
               device:str):
  
  model.train()
  train_loss,train_acc = 0,0
  for X,y in train_dataloader:
    X = X.to(device)
    y = y.to(device)

    train_logit = model(X)


    #these loss and acc metrics are being computed per batch in loop, not per sample
    loss = loss_fn(train_logit,y)

    train_loss += loss.item()

    opt.zero_grad()

    loss.backward()
    opt.step()

    train_pred_labels = torch.argmax(torch.softmax(train_logit,dim=1),dim=1)

    train_acc += (train_pred_labels==y).sum().item()/len(y)

  #get average loss and acc per batch
  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  return train_loss,train_acc

def test_step(model:nn.Module,
               loss_fn:nn.Module,
               test_dataloader:torch.utils.data.DataLoader,
               device:str):
  model.eval()
  test_loss,test_acc = 0,0
  with torch.inference_mode():
    for X,y in test_dataloader:
      X = X.to(device)
      y = y.to(device)

      test_logit = model(X)

      loss = loss_fn(test_logit,y)

      test_loss += loss.item()

    
      test_pred_labels = torch.argmax(torch.softmax(test_logit,dim=1),dim=1)

      test_acc += (test_pred_labels==y).sum().item()/len(y)

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
  return test_loss,test_acc



# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device=device,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
  # 2. Create empty results dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []
             }

  # 3. Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       train_dataloader=train_loader,
                                       loss_fn=loss_fn,
                                       opt=optimizer,
                                       device=device)

    test_loss, test_acc = test_step(model=model,
                                    test_dataloader=test_loader,
                                    loss_fn=loss_fn,
                                    device=device)

    # 4. Print out what's happening
    print(
      f"Epoch: {epoch + 1} | "
      f"train_loss: {train_loss:.4f} | "
      f"train_acc: {train_acc:.4f} | "
      f"test_loss: {test_loss:.4f} | "
      f"test_acc: {test_acc:.4f}"
    )

    # 5. Update results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    model_save_path = 'Saved Models/VGGNETINSPIRED_replica_epoch' + str(epoch + 1) + '.pth'
    torch.save(model.state_dict(), model_save_path)
  # 6. Return the filled results at the end of the epochs
  return results

