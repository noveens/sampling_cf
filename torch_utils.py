import torch

is_cuda_available = torch.cuda.is_available()

if is_cuda_available: 
    print("Using CUDA...\n")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor
    BoolTensor = torch.cuda.BoolTensor
else:
    LongTensor = torch.LongTensor
    FloatTensor = torch.FloatTensor
    BoolTensor = torch.BoolTensor

def get_model_class(hyper_params):
    from pytorch_models import MF, MVAE, SASRec, SVAE

    return {
        "bias_only": MF.MF,
        "MF_dot": MF.MF,
        "MF": MF.MF,
        "MVAE": MVAE.MVAE,
        "SVAE": SVAE.SVAE,
        "SASRec": SASRec.SASRec,
    }[hyper_params['model_type']]

def xavier_init(model):
    for _, param in model.named_parameters():
        try: torch.nn.init.xavier_uniform_(param.data)
        except: pass # just ignore those failed init layers
