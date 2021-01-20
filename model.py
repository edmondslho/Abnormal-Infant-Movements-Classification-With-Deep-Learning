from torch import nn
import torch.nn.functional as F

###############################################################################
# This is the implementation of the Abnormal Infant Movements Classification Framework proposed in
# [1] McCay et al., "Abnormal Infant Movements Classification with Deep Learning on Pose-based Features", IEEE Access, vol. 8, pp. 51582-51592, 2020.
#
# The HOJO2D and HOJD2D features are extracted using the methods proposed in
# [2] McCays et al., "Establishing Pose Based Features Using Histograms for the Detection of Abnormal Infant Movements", IEEE EMBC, pp. 5469-5472, July 2019.
#
# GitHub page: https://github.com/edmondslho/SMARTBabies
#
# Please feel free to contact the corresponding author Edmond S. L. Ho (e.ho@northumbria.ac.uk or edmond@edho.net) for any questions and comments
#
# Please cite these papers in your publications if it helps your research:
#
# @ARTICLE{McCay:DeepBaby,
#   author={K. D. {McCay} and E. S. L. {Ho} and H. P. H. {Shum} and G. {Fehringer} and C. {Marcroft} and N. D. {Embleton}},
#   journal={IEEE Access},
#   title={Abnormal Infant Movements Classification With Deep Learning on Pose-Based Features},
#   year={2020},
#   volume={8},
#   pages={51582-51592},
#   doi={10.1109/ACCESS.2020.2980269}
# }
#
# @INPROCEEDINGS{McCay:PoseBaby,
#    author={K. D. {McCay} and E. S. L. {Ho} and C. {Marcroft} and N. D. {Embleton}},
#    booktitle={2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
#    title={Establishing Pose Based Features Using Histograms for the Detection of Abnormal Infant Movements},
#    year={2019},
#    pages={5469-5472},
#    doi={10.1109/EMBC.2019.8857680}
# }
#
###############################################################################

class FCNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        for linear in self.hidden_layers:
            nn.init.xavier_uniform_(linear.weight)
          
        self.output = nn.Linear(hidden_layers[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)
            
        self.dropout = nn.Dropout(p=drop_p)
        self.pRelu = nn.PReLU()
            
    def forward(self, x):
        for linear in self.hidden_layers:
            x = self.pRelu(linear(x))
            x = self.dropout(x)
            
        x = self.output(x)
            
        return F.log_softmax(x, dim=1)    

class Conv1DNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5, version=1):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Conv1d(hidden_layers[0], hidden_layers[1], 3, stride=3, padding=1)])
            
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[1:-1], hidden_layers[2:])
        self.hidden_layers.extend([nn.Conv1d(h1, h2, 3, stride=3, padding=1) for h1, h2 in layer_sizes])

        for linear in self.hidden_layers:
            nn.init.xavier_uniform_(linear.weight)

        if version==1:      # determine if it is Conv1DNet-1 or Conv1DNet-2
            self.output = nn.Linear(64, output_size)
        else:
            self.output = nn.Linear(128, output_size)

        nn.init.xavier_uniform_(self.output.weight)
            
        self.dropout = nn.Dropout(p=drop_p)
        self.maxpool1d = nn.MaxPool1d(3, 3)
        self.pRelu = nn.PReLU()
            
    def forward(self, x):
        oriD = x.size()
        x = x.view(oriD[0], -1, oriD[1])
            
        for linear in self.hidden_layers:
            x = self.pRelu(linear(x))
            x = self.maxpool1d(x)
            x = self.dropout(x)
        
        oriD = x.size()
        x = x.view(oriD[0], -1)

        x = self.output(x)
            
        return F.log_softmax(x, dim=1)

class Conv2DNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5, version=1, binSize=8):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Conv2d(hidden_layers[0], hidden_layers[1], 3, stride=1, padding=1)])

        layer_sizes = zip(hidden_layers[1:-1], hidden_layers[2:])
        self.hidden_layers.extend([nn.Conv2d(h1, h2, 3, stride=1, padding=1) for h1, h2 in layer_sizes])

        for linear in self.hidden_layers:
            nn.init.xavier_uniform_(linear.weight)

        if version==1:      # determine if it is Conv2DNet-1 or Conv2DNet-2
            if binSize==16:
                self.output = nn.Linear(60, output_size)
            else:
                self.output = nn.Linear(44, output_size)
        else:
            if binSize==16:
                self.output = nn.Linear(120, output_size)
            else:
                self.output = nn.Linear(88, output_size)

        nn.init.xavier_uniform_(self.output.weight)
            
        self.dropout = nn.Dropout(p=drop_p)
        self.maxpool2d = nn.MaxPool2d(3, 2)
        self.pRelu = nn.PReLU()

        self.binSize = binSize
            
    def forward(self, x):
        oriD = x.size()
        x = x.view(oriD[0], 1, -1, self.binSize)

        for linear in self.hidden_layers:
            x = self.pRelu(linear(x))
            x = self.maxpool2d(x)
            x = self.dropout(x)
        
        oriD = x.size()
        x = x.view(oriD[0], -1)

        x = self.output(x)
            
        return F.log_softmax(x, dim=1)