import time
from numpy import vstack
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import torch
from torch.optim import SGD
from torch.nn import BCELoss
import time
import math

def train_model(train_dl, model, epochs=100, lr=0.01, 
                momentum=0.9, save_path='model.pth'):
    # Define your optimisation function for reducing loss when weights are calculated 
    # and propogated through the network
    start = time.time()
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    loss = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        model.train()
        # Iterate through training data loader
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data,1) #Get the class labels
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                  for p in model.parameters())

            loss = loss + l2_lambda * l2_norm
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        torch.save(model, save_path)
    time_delta = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_delta // 60, time_delta % 60
    ))
    
    return model

def evaluate_model(test_dl, model, beta=1.0):
    preds = []
    actuals = []
    for (i, (inputs, targets)) in enumerate(test_dl):
        #Evaluate the model on the test set
        yhat = model(inputs)
        #Retrieve a numpy weights array
        yhat = yhat.detach().numpy()
        # Extract the weights using detach to get the numerical values in an ndarray, instead of tensor
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # Round to get the class value i.e. sick vs not sick
        yhat = yhat.round()
        # Store the predictions in the empty lists initialised at the start of the class
        preds.append(yhat)
        actuals.append(actual)
    
    # Stack the predictions and actual arrays vertically
    preds, actuals = vstack(preds), vstack(actuals)
    #Calculate metrics
    cm = confusion_matrix(actuals, preds)
    # Get descriptions of tp, tn, fp, fn
    tn, fp, fn, tp = cm.ravel()
    total = sum(cm.ravel())
    
    metrics = {
        'accuracy': accuracy_score(actuals, preds),
        'AU_ROC': roc_auc_score(actuals, preds),
        'f1_score': f1_score(actuals, preds),
        'average_precision_score': average_precision_score(actuals, preds),
        'f_beta': ((1+beta**2) * precision_score(actuals, preds) * recall_score(actuals, preds)) / (beta**2 * precision_score(actuals, preds) + recall_score(actuals, preds)),
        'matthews_correlation_coefficient': (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
        'precision': precision_score(actuals, preds),
        'recall': recall_score(actuals, preds),
        'true_positive_rate_TPR':recall_score(actuals, preds),
        'false_positive_rate_FPR':fp / (fp + tn) ,
        'false_discovery_rate': fp / (fp +tp),
        'false_negative_rate': fn / (fn + tp) ,
        'negative_predictive_value': tn / (tn+fn),
        'misclassification_error_rate': (fp+fn)/total ,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        #'confusion_matrix': confusion_matrix(actuals, preds), 
        'TP': tp,
        'FP': fp, 
        'FN': fn, 
        'TN': tn
    }
    return metrics, preds, actuals

def eval_model(test_dl, model, cm_out_name='confusion_mat.csv',
               beta=1, export_index=False):
    results = evaluate_model(test_dl, model, beta)
    model_metrics = results[0]
    metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index', columns=['metric'])
    metrics_df.index.name = 'metric_type'
    metrics_df.reset_index(inplace=True)
    metrics_df.to_csv(cm_out_name, index=export_index)
    print(metrics_df)
    return metrics_df, model_metrics, results