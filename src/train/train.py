#libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataloader import AudioDataset
from model.model import NN
import torchmetrics as tm
from tqdm import tqdm
from clearml import Task, Logger
from pathlib import Path
from torch.optim.lr_scheduler import LinearLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def trainer(CONFIGURATION):
  #device
  device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

  #clearml
  task= Task.init(project_name='profanity_detection', task_name='adima_training_try')
  task.connect(CONFIGURATION)



  #dataloaders
  combined_dataset = AudioDataset (csv_file = str(Path.home())+'/Daniyal/profanity_detection/data/features/train_val_oversampled.csv', root_dir =str(Path.home())+ "/Daniyal/profanity_detection/data/features/")
  combined_loader = DataLoader (dataset=combined_dataset, batch_size=CONFIGURATION['BATCH_SIZE'], shuffle=True)#, sampler=train_sampler)


  test_dataset = AudioDataset (csv_file = str(Path.home())+'/Daniyal/profanity_detection/data/features/test/divided_test.csv', root_dir =str(Path.home())+ "/Daniyal/profanity_detection/data/features/test/")
  test_loader = DataLoader(dataset=test_dataset, batch_size=CONFIGURATION['BATCH_SIZE'], shuffle=True)


  val_dataset = AudioDataset (csv_file = str(Path.home())+'/Daniyal/profanity_detection/data/features/test/np_test.csv', root_dir =str(Path.home())+ "/Daniyal/profanity_detection/data/features/test/")
  val_loader = DataLoader (dataset=val_dataset, batch_size=CONFIGURATION['BATCH_SIZE'], shuffle=True)

  #model
  model= NN(CONFIGURATION['DROPOUTS']).to(device)

  #performance metrics
  train_accuracy_list=[]
  val_accuracy_list=[]
  train_precision_list=[]
  val_precision_list=[]
  train_recall_list=[]
  val_recall_list=[]
  loss_list=[]
  val_loss_list=[]


  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(),
                          lr=CONFIGURATION['LEARNING_RATE'], 
                          betas=CONFIGURATION['ADAM_BETAS'], 
                          eps=CONFIGURATION['ADAM_EPS'], 
                          weight_decay=CONFIGURATION['L2_REGULARISATION'])
  scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=50)

  #training loop
  for epoch in range(CONFIGURATION['EPOCHS']):
    model.train()

    print(f"epoch: {epoch+1}/{CONFIGURATION['EPOCHS']}")
    
    pred=torch.tensor([]).to(device=device)
    label=torch.tensor([]).to(device=device)
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(combined_loader):

      data = data.to(device=device)
      targets = targets.to(device=device)

      data = data.reshape (data.shape [0], -1)

      scores = model(data)
      loss = criterion(scores, targets) 

      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      running_loss += loss.item()
      _, prediction = torch.max(scores, 1)
      pred=torch.cat([pred, prediction])
      label=torch.cat([label, targets])

    num_batches = len(combined_loader)
    avg_loss = running_loss / num_batches

    loss_list.append(avg_loss)


    train_acc=accuracy_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy())
    train_accuracy_list.append(train_acc)
    train_precision= precision_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), zero_division=0, pos_label=1)
    train_precision_list.append(train_precision)
    train_recall=recall_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), pos_label=1)
    train_recall_list.append(train_recall)
    macro_f1=f1_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='macro')

    #print(f'Training: Epoch: {epoch}, loss: {avg_loss:.2f}, train_accuracy: {accuracy(pred, label):.2f}, train_precision: {precision(pred, label): .2f}, train_recall: {recall(pred, label): .2f},')
    print(f'Training: Epoch: {epoch}, loss: {avg_loss}, train_accuracy: {train_acc:.2f}, train_precision: {train_precision: .2f}, train_recall: {train_recall: .2f}, train_f1_score: {macro_f1}, ')


    val_pred=torch.tensor([]).to(device=device)
    val_label=torch.tensor([]).to(device=device)
    val_running_loss = 0.0

    model.eval()
    with torch.no_grad(): 
      for batch_idx, (data, targets) in enumerate(val_loader):

        data = data.to(device=device)
        targets = targets.to(device=device)

        #data = data.reshape (data.shape [0], -1)

        scores = model(data)
        val_loss = criterion(scores, targets)
        val_running_loss += val_loss.item()

        _, prediction = torch.max(scores, 1)

        val_pred=torch.cat([val_pred, prediction])
        val_label=torch.cat([val_label, targets])


      num_val_batches = len(val_loader)
      avg_val_loss = val_running_loss / num_val_batches
      val_loss_list.append(avg_val_loss)

      val_acc=accuracy_score(val_label.cpu().detach().numpy(), val_pred.cpu().detach().numpy() )
      val_accuracy_list.append(val_acc)
      val_precision=precision_score(val_label.cpu().detach().numpy(), val_pred.cpu().detach().numpy(), zero_division=0, pos_label=1 )
      val_precision_list.append(val_precision)
      val_recall=recall_score(val_label.cpu().detach().numpy(), val_pred.cpu().detach().numpy(), pos_label=1 )
      val_recall_list.append(val_recall)
      macro_f1 = f1_score(val_label.cpu().detach().numpy(), val_pred.cpu().detach().numpy(), average='macro')

      #print(f'val_accuracy: {accuracy(val_pred, val_label):.2f}, val_precision: {precision(val_pred, val_label): .2f}, val_recall: {recall(val_pred, val_label): .2f},')
      print(f"val_loss: {avg_val_loss}, val_accuracy: {val_acc:.2f}, val_precision: {val_precision: .2f}, val_recall: {val_recall: .2f}, val_f1_score: {macro_f1}, lr: {optimizer.param_groups[0]['lr']}")

    Logger.current_logger().report_scalar("Loss", "train", iteration=epoch, value=avg_loss)
    Logger.current_logger().report_scalar("Loss", "val", iteration=epoch, value=avg_val_loss)
    Logger.current_logger().report_scalar("Accuracy", "train", iteration=epoch, value=train_acc)
    Logger.current_logger().report_scalar("Accuracy", "val", iteration=epoch, value=val_acc)
    Logger.current_logger().report_scalar("Precision", "train", iteration=epoch, value=train_precision)
    Logger.current_logger().report_scalar("Precision", "val", iteration=epoch, value=val_precision)
    Logger.current_logger().report_scalar("Recall", "train", iteration=epoch, value=train_recall)
    Logger.current_logger().report_scalar("Recall", "val", iteration=epoch, value=val_precision)

    if CONFIGURATION['LINEAR_LR']:
      scheduler.step()
  
  task.close()
  return model




