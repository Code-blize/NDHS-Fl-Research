import numpy as np, torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, log_loss)
from .model import NDHSMLP, set_parameters

def evaluate_pooled(parameters, validation_npz, input_dim):
    with np.load(validation_npz, allow_pickle=True) as z:
        X=np.asarray(z["X"],dtype=np.float32); y=np.asarray(z["y"],dtype=np.int64)
    model=NDHSMLP(input_dim); set_parameters(model,parameters); model.eval()
    with torch.no_grad(): prob=torch.sigmoid(model(torch.from_numpy(X))).numpy()
    pred=(prob>=0.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    both=len(np.unique(y))==2
    return {"n":int(len(y)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
      "accuracy":float(accuracy_score(y,pred)),
      "precision":float(precision_score(y,pred,zero_division=0)),
      "sensitivity":float(recall_score(y,pred,zero_division=0)),
      "specificity":float(tn/(tn+fp)) if tn+fp else float("nan"),
      "f1":float(f1_score(y,pred,zero_division=0)),
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "auroc":float(roc_auc_score(y,prob)) if both else float("nan"),
      "auprc_average_precision":float(average_precision_score(y,prob)) if both else float("nan"),
      "brier":float(brier_score_loss(y,prob)),
      "binary_cross_entropy":float(log_loss(y,np.column_stack([1-prob,prob]),labels=[0,1])),
      "observed_prevalence":float(y.mean()),"predicted_prevalence":float(pred.mean())}
