import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


def batch_iterate(features, labels, dest_features, dest_labels, batch_size):    
    for batch_idx in range((features.shape[0]+batch_size-1)//batch_size):
        start_idx = batch_idx*batch_size
        tmp_features_batch = features[start_idx:(batch_idx+1)*batch_size]
        eff_batch_size = tmp_features_batch.shape[0]
        dest_features[:eff_batch_size] = tmp_features_batch
        if labels is None:
            yield start_idx, eff_batch_size, dest_features[:eff_batch_size], None
        else:
            tmp_labels_batch = labels[start_idx:(batch_idx+1)*batch_size]
            dest_labels[:eff_batch_size] = tmp_labels_batch
            yield start_idx, eff_batch_size, dest_features[:eff_batch_size], dest_labels[:eff_batch_size]
            

class ES_Regression:
    def __init__(self, device,
                 alpha=0.95,
                 fit_intercept=True,
                 epochs=800,
                 lr=0.01,
                 num_batch = 8,
                 weight_decay = 0):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
        self.epochs = epochs 
        self.lr = lr 
        self.num_batch = num_batch 
        self.device = device 
        self.weight_decay = weight_decay
        
        self.coef_ = None

    def fit(self,
            X,
            y,
            weight_decay = None,
            verbose=None):
        device = self.device
        
        lr = self.lr
        epochs = self.epochs
        
        X = torch.as_tensor(X, dtype = torch.float32).to(device)
        y = torch.as_tensor(y, dtype = torch.float32).reshape(-1, 1).to(device)
        
        self.X_mean = X.mean(dim = 0, keepdim = True)
        self.X_std = X.std(dim = 0, keepdim = True)+1e-8
        self.y_mean = 0#y.mean(dim = 0, keepdim = True)
        self.y_std = y.std(dim = 0, keepdim = True)+1e-8
        
        if weight_decay is None:
            weight_decay = self.weight_decay
        
        N, dimX = X.shape
        num_batch = self.num_batch
        BS = N//(num_batch-1) if num_batch>1 else N
        
        if self.coef_ is None:
            W = torch.empty( size = (dimX,1), dtype = torch.float32).to(device)
        else:
            W = torch.empty_like(torch.tensor(self.coef_), dtype = torch.float32).reshape(-1,1).to(device)
        
        u = torch.zeros( size = (1,), dtype = torch.float32).to(device)
        torch.nn.init.normal_(W)
        torch.nn.init.normal_(u)
        
        W.requires_grad = True
        u.requires_grad = True

        optimizer = torch.optim.Adam([W, u], lr=lr, weight_decay= weight_decay)
        shuffled_ind = np.random.permutation(N)
        X = X[shuffled_ind]
        y = y[shuffled_ind]
        best_err = np.inf
        best_u = u.data.clone()
        best_W = W.data.clone()
        
        losses = []
        for e in range(epochs):
            for i in range(num_batch):
                pred = torch.matmul((X[i*BS:(i+1)*BS]- self.X_mean)/ self.X_std, W)
                resi = (y[i*BS:(i+1)*BS]-self.y_mean)/self.y_std - pred 
                resi_central = resi - resi.mean()

                loss = u + 1 / (1 - self.alpha) * torch.mean(torch.relu(resi_central - u)) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                err = 0
                for i in range(num_batch):
                    pred = torch.matmul((X[i*BS:(i+1)*BS]- self.X_mean)/ self.X_std, W)
                    resi = (y[i*BS:(i+1)*BS]-self.y_mean)/self.y_std - pred
                    resi_central = resi - resi.mean()
                    err += u*BS  + 1 / (1 - self.alpha) * torch.sum(torch.relu(resi_central - u))
                err /= N
                err = float(err.data.item())
                losses.append(err)

            if err < best_err:
                best_err = err
                best_u = u.data.clone()
                best_W = W.data.clone()

            if (verbose is not None) and (e % verbose == 0):
                print('[iter {}] err = {}'.format(e, round(err, 3)))
        
        plt.plot(np.log(losses))
        plt.title('Training losses against epoch when performing ES hedging')
        plt.xlabel('Epoch')
        plt.ylabel('Log of losses')
        plt.show()
        with torch.no_grad():
            u.copy_(best_u)
            W.copy_(best_W)
            self.coef_torch = (W.reshape(1,-1)/self.X_std*self.y_std)#.cpu().numpy()
            self.coef_ = self.coef_torch.cpu().numpy()

            if self.fit_intercept:
                self.intercept_ = float((y - torch.matmul(X,self.coef_torch.reshape(-1,1))).mean().cpu().numpy()) 
            else:
                self.intercept_ = 0
    def predict(self, X):
        return np.matmul(X, self.coef_.reshape(-1,1)) + self.intercept_
    
def poly_square(X):
    return np.concatenate([X, X**2], axis = 1)

def hedging_error_standardized(model, X, d_cva):
    sc_mean = StandardScaler(with_mean= True,with_std=False)
    
    return sc_mean.fit_transform(d_cva.reshape(-1,1) - model.predict(X).reshape(-1,1))/d_cva.std()

def pl_explain(model, X, d_cva):
    
    return model.predict(X).reshape(-1,1) + (d_cva.mean()-model.predict(X).reshape(-1,1).mean()) #sc_mean.fit_transform(d_cva[time].reshape(-1,1) - model.predict(X[time]).reshape(-1,1))

    
def es_score(m, q =0.95):
    the = np.quantile(m,q)
    return m[m>=the].mean()
def var_score(m, q =0.99):
    the = np.quantile(m,q)
    return the

def twin_score(prediction, y0, y1, CVA0 = 1):
    prediction = prediction.ravel()
    y0 = y0.ravel()
    y1 = y1.ravel()
    statistic =  prediction**2 + y0*y1 - prediction * (y0 + y1)
    return (np.sqrt(statistic.mean())/CVA0,  np.sqrt(statistic.mean() + 2 * statistic.std() / np.sqrt(len(statistic)))/CVA0)

