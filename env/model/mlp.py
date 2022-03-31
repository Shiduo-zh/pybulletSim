
import torch
import torch.nn.functional as F
import numpy as np

class MLP(torch.nn.Module):
    def __init__(self,input_size=93,hidden_size=256,output_size=128):
        super(MLP,self).__init__()     
        self.hidden = torch.nn.Linear(input_size,hidden_size)  
        self.output = torch.nn.Linear(hidden_size,output_size)   
        
    def forward(self,x):       
        out = F.relu(self.hidden(x))   
        out = F.relu(self.output(out))
        return out

class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
            zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        self.sequence = list()
        for layer in hidden_layers:
            self.sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            self.sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*self.sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size
    
    def linear_params(self):
        # weight=np.array([])
        # bias=np.array([])
        for layer in self.sequence:
            weights=list()
            if(not isinstance(layer,torch.nn.ReLU)):
                weight=layer.weight.cpu().detach().numpy()
                bias=layer.bias.cpu().detach().numpy()
                print(weight.shape,bias.shape)
        return weight,bias    


