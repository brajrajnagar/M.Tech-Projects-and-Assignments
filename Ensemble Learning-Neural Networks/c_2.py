import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as met
import neural


train = pd.read_csv(neural.train_path,header = None, index_col=False).to_numpy()
test = pd.read_csv(neural.test_path,header = None, index_col=False).to_numpy()



class NN_classifier:
    
    def __init__(self,HLayers_list,n_classes,n_features,batch_size,eta,activation_fn,converge_limit,LR_type,cost_fn_type):
        self.HLayers_list = HLayers_list
        self.n_classes = n_classes
        self.n_features = n_features
        self.batch_size = batch_size
        self.deltas_list = [0]*(len(HLayers_list)+1)
        self.outputs_list = [0]*(len(HLayers_list)+1)
        self.HLayer_list_len = len(HLayers_list)
        self.total_n_layers = len(HLayers_list)+1
        self.cost_fn_type = cost_fn_type
        self.eta = eta
        self.thetas_list = []
        self.one_hot_enc_Y = None
        self.epoch = 0
        self.LR_type = LR_type
        self.activation_fn = activation_fn
        self.converge_limit = converge_limit

    def class_prediction(self,X):
        examples_count = X.shape[0]
        self.forward_prop(X,examples_count)
        predicted_outputs = self.outputs_list[-1]
        prediction_best = np.array([])
        for j in range(predicted_outputs.shape[0]):
            prediction_best = np.append(prediction_best, np.argmax(predicted_outputs[j]))
        return prediction_best
    
    def init_weights(self):
        thetas_list = []
        if self.HLayer_list_len==0:
            n_rows = self.n_classes
            n_cols = self.n_features+1
            thetas_list.append(np.random.normal(0,0.06,(n_rows,n_cols)))
        else:
            n_rows = self.HLayers_list[0]
            n_cols = self.n_features+1
            thetas_list.append(np.random.normal(0,0.06,(n_rows,n_cols)))
            for i in range(1,self.HLayer_list_len):
                n_rows = self.HLayers_list[i]
                n_cols = (self.HLayers_list[i-1]+1)
                thetas_list.append(np.random.normal(0,0.06,(n_rows,n_cols)))

            n_rows = self.n_classes
            n_cols = (self.HLayers_list[-1]+1)
            thetas_list.append(np.random.normal(0,0.06,(n_rows,n_cols)))
        
        self.thetas_list = thetas_list
        
    def cost_functions(self,Y_pred,Y_org):
        m = Y_org.shape[0]
        if (self.cost_fn_type == 'MSE'):
            return (np.sum((Y_org-Y_pred)**2)/(2*m))
        else:   #cross-entropy
            return -(1/m)*(np.sum(Y_org*np.log(Y_pred) + (1-Y_org)*np.log(1-Y_pred)))

    
    def relu_derivative(self,O_j):
        b = O_j.flatten()
        for i in range(len(b)):
            if b[i] > 0:
                b[i] = 1.0
            else:
                b[i] = 0.0
        rd = b.reshape(O_j.shape)
        return rd
    
    def act_function(self,z):
        if self.activation_fn == "sigmoid":
            a_out = (1/(1+np.exp(-z)))
            return a_out
        elif self.activation_fn == "relu":
            a_out = np.maximum(np.zeros(z.shape),z)
            return a_out
    
    def one_hot_enc(self,Y):
        self.one_hot_enc_Y = np.zeros((Y.shape[0], self.n_classes), dtype=float)
        for j in range(len(self.one_hot_enc_Y)):
            self.one_hot_enc_Y[j][Y[j]] = 1
        
    
    def forward_prop(self,X_in,batchSize):
        X_in_new = np.append(np.ones((batchSize,1)), X_in, axis=1)

        for j in range(self.HLayer_list_len):
            z = np.dot(X_in_new,np.transpose(self.thetas_list[j]))
            act_output = self.act_function(z)
            self.outputs_list[j] = act_output
            X_in_new = np.append(np.ones((batchSize,1)), act_output, axis=1)
        
        z_out_layer = np.dot(X_in_new,np.transpose(self.thetas_list[self.HLayer_list_len]))
        self.outputs_list[self.HLayer_list_len] = (1/(1+np.exp(-z_out_layer)))
    
    def back_prop(self,Y_in):
        
        O_j = self.outputs_list[self.total_n_layers-1] 
        doj_dnetj = O_j * (1 - O_j)
        delta_j = doj_dnetj * (Y_in - O_j)
        self.deltas_list[-1] = delta_j
        
        for j in range(self.total_n_layers-2,-1,-1):
            if self.activation_fn == "sigmoid": 
                O_j_hidden = self.outputs_list[j]
                doj_dnetj = O_j_hidden * (1-O_j_hidden)
            elif self.activation_fn == "relu":
                doj_dnetj = self.relu_derivative(self.outputs_list[j])
                
            dE_doj = np.dot(self.deltas_list[j+1],self.thetas_list[j+1][:,1:])
            delta_j = doj_dnetj * dE_doj
            self.deltas_list[j] = delta_j 
    
    def w_update(self,X_in): 
        for j in range(self.total_n_layers):
            if j == 0:
                prev_layer_in = X_in
            else:
                prev_layer_in = self.outputs_list[j-1]

            o_i = np.append(np.ones((self.batch_size,1)),prev_layer_in,axis=1)
            del_wij = (self.eta/self.batch_size) * np.dot(np.transpose(self.deltas_list[j]),o_i)
            self.thetas_list[j] = self.thetas_list[j] + del_wij
    
    def init_NN(self,train_X,train_Y):
        n_examples = train_X.shape[0]
        n_batches = int(train_X.shape[0]/self.batch_size)
        self.one_hot_enc(train_Y)
        self.init_weights()
        
        initial_cost = 1e10
        final_cost = -1e-10
        i = 0
        not_converged = True
        while not_converged:
            cost_cal = 0
            for l in range(n_batches):

                X_batch = train_X[i:i+self.batch_size,:]
                Y_encoded_batch = self.one_hot_enc_Y[i:i+self.batch_size,:]
                self.forward_prop(X_batch,self.batch_size)
                cost_cal += self.cost_functions(self.outputs_list[-1],Y_encoded_batch)
                self.back_prop(Y_encoded_batch)
                self.w_update(X_batch)

                if (l == (n_batches-1)):
                    self.epoch+=1
                    if self.LR_type == "adaptive":
                        self.eta = 0.1/np.sqrt(self.epoch)

                i=(i+self.batch_size)%n_examples

            initial_cost = final_cost
            final_cost = cost_cal/n_batches
            # print(abs(initial_cost-final_cost))
            if (abs(initial_cost-final_cost) < self.converge_limit and self.epoch>50):
                not_converged = False





def Accuracy(Y_org,Y_pred):
    n = Y_pred.shape[0]
    return (np.sum(Y_org==Y_pred)/n)


hidden_layers_list = [5, 10, 15, 20, 25]

train_acc_adapt = []
training_time_adapt = []
test_acc_adapt = []
epoch_list_adapt = []

for neurons in hidden_layers_list:
    nn = NN_classifier([neurons],10,784,100,0.1,"sigmoid",(5e-5),"adaptive", 'MSE')
    start = time.time()
    nn.init_NN(train[:,:-1]/255,train[:,-1])
    end = time.time()
    training_time_adapt.append(end-start)
    epoch_list_adapt.append(nn.epoch)
    predictedTrain = nn.class_prediction(train[:,:-1]/255)
    predictedTest = nn.class_prediction(test[:,:-1]/255)
    train_acc_adapt.append((Accuracy(train[:,-1],predictedTrain)*100))
    test_acc_adapt.append((Accuracy(test[:,-1],predictedTest)*100))

    confusion_matrix = met.confusion_matrix(test[:,-1], predictedTest)
    cm_display = met.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.savefig(neural.out_path+'/q2c_'+str(neurons)+'_adaptive.jpg')





# Writing outputs
with open(neural.out_path+"/c.txt", "w+") as file1:
    for i in range(len(train_acc_adapt)):
        file1.write(f"Train accuracy for hidden layer of {hidden_layers_list[i]} units is: {train_acc_adapt[i]}% \n")
    file1.write("\n")
    for i in range(len(test_acc_adapt)):
        file1.write(f"Test accuracy for hidden layer of {hidden_layers_list[i]} units is: {test_acc_adapt[i]}% \n")
    file1.write("\n")
    for i in range(len(training_time_adapt)):
        file1.write(f"Training time for hidden layer of {hidden_layers_list[i]} units is: {training_time_adapt[i]}% \n")
    file1.write("\n")
    for i in range(len(epoch_list_adapt)):
        file1.write(f"Total no. of epochs for hidden layer of {hidden_layers_list[i]} units is: {epoch_list_adapt[i]}% \n")
    file1.write("\n")





fig1 = plt.figure()
plt.plot(hidden_layers_list,train_acc_adapt,label='Train Accuracies')
plt.plot(hidden_layers_list,test_acc_adapt,label = 'Test Accuracies')
plt.xlabel('Number of Hidden Layer Units')
plt.ylabel('Accuracy in %')
plt.legend()
plt.savefig(neural.out_path+'/q2c_accuracies.jpg')




fig2 = plt.figure()
plt.plot(hidden_layers_list, training_time_adapt, label='Training Times')
plt.xlabel('Number of Hidden Layer Units')
plt.ylabel('Training Time (in secs)')
plt.legend()
plt.savefig(neural.out_path+'/q2c_traintime.jpg')




