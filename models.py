import torch
import torch.nn as nn
import time

class GRU_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, hhidden_size, num_layers, dropout_p=0.5):
        super(GRU_Encoder, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.gru2 = nn.GRU(hidden_size, hhidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        x, hidden1 = self.gru1(x)
        x = self.dropout(x)
        x, hidden2 = self.gru2(x)
        return x, hidden1, hidden2

class GRU_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, hhidden_size, num_layers, dropout_p=0.5):
        super(GRU_Decoder, self).__init__()
        self.gru1 = nn.GRU(hhidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.gru2 = nn.GRU(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        x, hidden1 = self.gru1(x)
        x = self.dropout(x)
        x, hidden2 = self.gru2(x)
        return x, hidden1, hidden2

class CNN_Encoder(nn.Module):
    def __init__(self, input_size, out_channels1, out_channels2, kernel_size, stride=1):
        super(CNN_Encoder, self).__init__()
        self.cnn1 = nn.Conv1d(input_size, out_channels1, kernel_size, stride=stride)
        self.cnn2 = nn.Conv1d(out_channels1, out_channels2, kernel_size, stride=stride)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = x.transpose(1, 2)

        return x

class CNN_Decoder(nn.Module):
    def __init__(self, input_size, out_channels1, out_channels2, kernel_size, stride=1):
        super(CNN_Decoder, self).__init__()
        self.cnt1 = nn.ConvTranspose1d(input_size, out_channels1, kernel_size, stride=stride)
        self.cnt2 = nn.ConvTranspose1d(out_channels1, out_channels2, kernel_size, stride=stride)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnt1(x)
        x = self.relu(x)
        x = self.cnt2(x)
        x = self.relu(x)
        x = x.transpose(1, 2)

        return x

class GRU_AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, hhidden_size, num_layers, path_vecs, sys_vecs, device):
        super(GRU_AutoEncoder, self).__init__()
        
        self.path_emb = nn.Embedding.from_pretrained(torch.tensor(path_vecs, dtype=torch.float).cuda(), freeze=True)
        self.sys_emb = nn.Embedding.from_pretrained(torch.tensor(sys_vecs, dtype=torch.float).cuda(), freeze=True)

        self.encoder = GRU_Encoder(input_size=input_size, hidden_size=hidden_size, hhidden_size=hhidden_size, num_layers=num_layers)
        self.reconstruct_decoder = GRU_Decoder(input_size=input_size, hidden_size=hidden_size, hhidden_size=hhidden_size, num_layers=num_layers)
        self.input_size = input_size

        self.device = device

        self.criterion = nn.MSELoss()

    def forward(self, batch):
        batch, _  = batch

        batch = batch.to(self.device)

        batch_size, sequence_length, _ = batch.size()
        vector_size = self.input_size
        
        path_batch = self.path_emb(batch[:,:,0])
        sys_batch = self.sys_emb(batch[:,:,1])

        batch = path_batch + sys_batch
        batch = batch.reshape(batch_size, sequence_length, vector_size)
        
        outputs, hidden1, hidden2 = self.encoder(batch)
        outputs = self.reconstruct_decoder(outputs)
        
        
        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        reconstruct_loss = self.criterion(outputs, batch[:, inv_idx, :])

        batch = batch.to("cpu")
        torch.cuda.empty_cache()

        return outputs[:, -1,:], reconstruct_loss

class GRU_Repeat_AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, hhidden_size, latent_size, num_layers, path_vecs, sys_vecs, device):
        super(GRU_Repeat_AutoEncoder, self).__init__()
        
        self.path_emb = nn.Embedding.from_pretrained(torch.tensor(path_vecs, dtype=torch.float).cuda(), freeze=True)
        self.sys_emb = nn.Embedding.from_pretrained(torch.tensor(sys_vecs, dtype=torch.float).cuda(), freeze=True)

        self.encoder = GRU_Encoder(input_size=input_size, hidden_size=hidden_size, hhidden_size=hhidden_size, num_layers=num_layers)
        self.linear1 = nn.Linear(hhidden_size, latent_size)

        self.reconstruct_decoder = GRU_hidden_Decoder(input_size=hidden_size, hidden_size=hhidden_size, hhidden_size=latent_size, num_layers=num_layers)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.input_size = input_size
        self.latent_size = latent_size

        self.device = device

        self.criterion = nn.MSELoss()

    def forward(self, batch):
        batch, _  = batch

        batch = batch.to(self.device)

        batch_size, sequence_length, _ = batch.size()
        vector_size = self.input_size
        
        path_batch = self.path_emb(batch[:,:,0])
        sys_batch = self.sys_emb(batch[:,:,1])

        batch = path_batch + sys_batch
        batch = batch.reshape(batch_size, sequence_length, vector_size)
        
        outputs, _ = self.encoder(batch)
        latent = self.linear1(outputs[:, -1, :])
        x = latent.reshape(batch_size, 1, self.latent_size)

        hidden1 = None
        hidden2 = None
        reconstruct_output = []

        for t in range(sequence_length):
            temp_output, hidden1, hidden2 = self.reconstruct_decoder(x, hidden1, hidden2)
            reconstruct_output.append(self.linear2(temp_output))
        
        reconstruct_output = torch.cat(reconstruct_output, dim=1)

        #inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        #reconstruct_loss = self.criterion(reconstruct_output, batch[:, inv_idx, :])
        reconstruct_loss = self.criterion(reconstruct_output, batch)

        batch = batch.to("cpu")
        torch.cuda.empty_cache()

        return latent, reconstruct_loss

class CNN_AutoEncoder(nn.Module):
    def __init__(self, input_size, out_channels1, out_channels2, latent_size, path_vecs, sys_vecs, device, kernal_size = 3, squence_lenth = 50, stride=1):
        super(CNN_AutoEncoder, self).__init__()
        
        self.path_emb = nn.Embedding.from_pretrained(torch.tensor(path_vecs, dtype=torch.float).cuda(), freeze=True)
        self.sys_emb = nn.Embedding.from_pretrained(torch.tensor(sys_vecs, dtype=torch.float).cuda(), freeze=True)

        self.encoder = CNN_Encoder(input_size=input_size, out_channels1=out_channels1, out_channels2=out_channels2, kernel_size=kernal_size, stride=stride)

        L = squence_lenth - 2 * (kernal_size - 1)
        self.linear1 = nn.Linear(L * out_channels2, latent_size)

        self.reconstruct_decoder = CNN_Decoder(input_size=latent_size, out_channels1=out_channels2, out_channels2=out_channels1, kernel_size=kernal_size, stride=stride)
        self.linear2 = nn.Linear(out_channels1, input_size)

        self.input_size = input_size
        self.latent_size = latent_size
        self.kernal_size = kernal_size
        self.device = device

        self.criterion = nn.MSELoss()

    def forward(self, batch):
        batch, _  = batch

        batch = batch.to(self.device)

        batch_size, sequence_length, _ = batch.size()
        vector_size = self.input_size
        
        path_batch = self.path_emb(batch[:,:,0])
        sys_batch = self.sys_emb(batch[:,:,1])

        batch = path_batch + sys_batch
        batch = batch.reshape(batch_size, sequence_length, vector_size)
        
        outputs = self.encoder(batch)
        outputs = torch.flatten(outputs, start_dim=1)
        latent = self.linear1(outputs)

        L = sequence_length - 2 * (self.kernal_size - 1)
        outputs = latent.reshape(batch_size, 1, -1)
        outputs = torch.repeat_interleave(outputs, L, dim=1)

        decoder_output = self.reconstruct_decoder(outputs)

        reconstruct_output = []
        for t in range(sequence_length):
            reconstruct_output.append(self.linear2(decoder_output[:, t, :]).reshape(batch_size, 1, -1))

        reconstruct_output = torch.cat(reconstruct_output, dim=1)

        reconstruct_loss = self.criterion(reconstruct_output, batch)

        batch = batch.to("cpu")
        torch.cuda.empty_cache()

        return latent, reconstruct_loss

def run(model, train_loader, vali_loader, learning_rate, max_epochs, early_stop):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = range(max_epochs)
    
    loss_list = []
    min_loss = 1
    count = 0
    
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()

        for i, batch_data in enumerate(train_loader):

            _, reconstruct_loss = model(batch_data)

            loss = reconstruct_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()        
        epoch_loss = 0

        with torch.no_grad() :
            for i, batch_data in enumerate(vali_loader):
                _, reconstruct_loss = model(batch_data)
                epoch_loss = epoch_loss + reconstruct_loss
            
            epoch_loss = epoch_loss / len(vali_loader)
            loss_list.append(epoch_loss)

            print(f"[{epoch + 1}/{max_epochs}] loss : {epoch_loss}  -- " + 
                    f"{time.strftime('%H:%M:%S', time.localtime(time.time()))}")

        if min_loss > epoch_loss :
            min_loss = epoch_loss
            save_model = model
            count = 0
        else :
            count = count + 1

        #torch.save(model, f"GRU_Positive_training_Auto_encoder_epoch{epoch}.model")

        if count >= early_stop :
            break
            
    return save_model, min_loss