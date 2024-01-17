from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv



class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(nout)
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:,0,:]
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder

class WGAN(nn.Module):
    def __init__(self, n_in, n_hid):
        super(WGAN, self).__init__()

        self.n_hid = n_hid
        self.n_in = n_in

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, 1)

    def forward(self, text,mol):
        text_pred = nn.LeakyReLU(negative_slope=0.2)(self.fc1(text))
        text_pred = nn.LeakyReLU(negative_slope=0.2)(self.fc2(text_pred))
        text_pred = nn.Sigmoid()(self.fc3(text_pred))
        mol_pred = nn.LeakyReLU(negative_slope=0.2)(self.fc1(mol))
        mol_pred = nn.LeakyReLU(negative_slope=0.2)(self.fc2(mol_pred))
        mol_pred = nn.Sigmoid()(self.fc3(mol_pred))
        return text_pred, mol_pred  

class AMAN(nn.Module):
    def __init__ (self, model_name, num_node_features, nout, graph_hidden_channels,nhid_WGAN=100,n_layers = 3):
        super(AMAN, self).__init__()

        self.graph_encoder = GATEncoder(num_node_features, nout, graph_hidden_channels,n_layers)
        self.text_encoder = TextEncoder(model_name)
        self.ln = nn.LayerNorm(nout)
        self.lc = nn.Linear(nout, nout)
        self.relu = nn.ReLU()
        self.WGAN= WGAN(nout,nhid_WGAN)

    def forward(self, graph_batch, input_ids, attention_mask):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
      
        graph_encoded = self.ln(self.graph_encoder(graph_batch))

        text_encoded = self.ln(self.text_encoder(input_ids, attention_mask))

        text_prob, mol_prob = self.WGAN(text_encoded,graph_encoded)
        return graph_encoded, text_encoded, text_prob, mol_prob




class GATEncoder(nn.Module):
    def __init__ (self, num_node_features, nout, graph_hidden_channels,n_layers = 3):
        super(GATEncoder, self).__init__()

        self.nout = nout
        self.relu = nn.ReLU()
        self.ln_graph_hidden_channels=nn.LayerNorm((graph_hidden_channels))
        self.ln = nn.LayerNorm((nout))
        self.lc = nn.Linear(graph_hidden_channels, nout)
        self.layers = []
        self.layers += [GATv2Conv(num_node_features, graph_hidden_channels)]
        
        for i in range(n_layers-1):
            self.layers += [GATv2Conv(graph_hidden_channels, graph_hidden_channels)]
        
        self.layers = nn.ModuleList(self.layers)

  

    def forward(self, graph_batch):

        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        for layer in self.layers:

            
            x = layer(x, edge_index)

            x = self.relu(x)
            x=self.ln_graph_hidden_channels(x)

        x = global_mean_pool(x,batch)
        x = self.lc(x)
        x=self.relu(x)
        return x

   
   