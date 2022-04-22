import numpy as np
import random
import os
from operator import itemgetter

#import autograd.numpy as np  # Thinly-wrapped numpy
#from autograd import grad
#from matplotlib import pyplot
from sklearn.decomposition import PCA
import math

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from model import Linear_Feature_Fusion_Approximate, Linear_Feature_Fusion, Linear_Feature_Fusion_No_Normal, Linear_Feature_Fusion_Approximate2, Linear_Feature_Fusion_Approximate3, Linear_Feature_Fusion_Batch, Linear_Feature_Fusion_Approximate2_Batch
from data_generation import data_gen

from ROC import New_ROC_AUC



def approximate_inv_norm(x_in):

    coeffs = [[-9.81663423,19.8459398,-13.57853979,4.38423127]]
    #coeffs = [[-3.5507169411544033, 5.007510260923221, -2.8600040042166497, 1.8188528918604154], [3.6282080345829653, -8.063566001389777, 6.0138261097265975, -0.4337306099760562]]
    
    coeffs = [[4.39946584, -6.80578761, 3.66480865]]
    
    coeffs = [[0.42084296,-1.81897596,2.51308415]] #large range
    x = torch.linalg.norm(x_in)**2
    #truth = 1/x**0.5
    result = 0
    for coeff_list in coeffs:
        result = coeff_list[0]
        for i in range(1,len(coeff_list)):
            result = result * x + coeff_list[i]
        x = result
        result = 0
    #print(truth, x)
    return x



def train_exact(gamma,iters,spec_margin=None,spec_lamb=None):
    random.seed(123)
    torch.manual_seed(123)
    
    a = []
    A_infile = open("feature-extraction/extractions/VGGFace_vgg_cplfw_new.txt",'r')
    for line in A_infile:
        line = line.strip().split()
        a.append(torch.tensor([float(char) for char in line]))

    
    
    L = []
    #L_infile = open("feature-extraction/extractions/deep_speaker_librispeech_big_labels.txt",'r')
    L_infile = open("feature-extraction/extractions/deep_speaker_librispeech_google_labels.txt",'r')
    
    
    
    
    l_dict = {}
    for line in L_infile:
        line = line.strip()
        if line not in l_dict:
            l_dict[line] = 0
        l_dict[line] += 1
        
        L.append(line.strip())
        #print(d[line])
        #L.append(d[line])
        #L.append(int(line.strip()))
    #print(l_dict)
    #print(min(l_dict.values()))
    
    #convert from unique string labels to integers
    #from https://stackoverflow.com/questions/43203215/map-unique-strings-to-integers-in-python
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(L)))])
    L = [d[x] for x in L]
    #print(len(L))
    
    l_dict = {}
    for l in L:
        if l not in l_dict:
            l_dict[l] = 0
        l_dict[l] += 1
    #print(L)
    
    b = []
    #B_infile = open("feature-extraction/extractions/deep_speaker_librispeech_big.txt",'r')
    #B_infile = open("feature-extraction/extractions/vggvox_librispeech.txt",'r')
    B_infile = open("feature-extraction/extractions/deep_speaker_librispeech_google.txt",'r')
    for line in B_infile:
        line = line.strip().split()
        b.append(torch.tensor([float(char) for char in line]))
    
    
    b, L = (list(t) for t in zip(*sorted(zip(b, L),key=itemgetter(1))))
    
    #num_classes = 275
    #num_each_class = 3

    
    completed = 0
    a_index = 0
    a_sub_index = 0
    b_index = 0
    samples_per_face = 2
    samples_per_voice = 10
    num_each_class = samples_per_face * samples_per_voice
    true_a = []
    true_b = []
    true_L = []
    
    
    num_classes = 188
    split1 = 0
    split2 = 0
    
    last_l = L[0]
    i = 0
    while completed <= 188:
        #for i, label in enumerate(L):
        label = L[b_index]
        for j in range(2):
            a_sub_index = j
            #b_index = i
            #print(label, a_index+a_sub_index)
            print(a_index+a_sub_index, b_index, label)
            true_a.append(a[a_index+a_sub_index])
            true_b.append(b[b_index])
            true_L.append(L[b_index])
        b_index += 1
        
        if b_index >= len(L):
            break
        label = L[b_index]
        #if a_sub_index == 1:
            #b_index += 1
            #a_index += 2
        #a_sub_index = 1-a_sub_index
        if label != last_l:
            last_l = label
            a_index += 2
            completed += 1
            if split1 == 0 and a_index/2 >= math.floor(num_classes * 0.2):
                split1 = i
            if split2 == 0 and a_index/2 >= math.floor(num_classes * 0.4):
                split2 = i
        i += 1
        #a_index += 2
        #completed += 1
        
    print(len(true_a), len(true_b))
    
    
    """
    print("pairs")
    while completed < num_classes:
        true_a.append(a[a_index+a_sub_index])
        true_b.append(b[b_index])
        true_L.append(L[b_index])
        print(a_index+a_sub_index, b_index, L[b_index])
        a_sub_index += 1
        if a_sub_index >= samples_per_face:
            a_sub_index = 0
            b_index += 1
            if b_index % samples_per_voice == 0:
                a_index += samples_per_face
                completed += 1
    """
    #split1 = math.floor(num_classes * 0.6)*num_each_class
    #split2 = math.floor(num_classes * 0.8)*num_each_class
    
    #split1 = math.floor(num_classes * 0.2)*num_each_class
    #split2 = math.floor(num_classes * 0.4)*num_each_class
    print("Splits:",split1,split2)
    
    A = torch.stack(true_a)
    print("A:",A.shape)
    B = torch.stack(true_b)
    print("B:",B.shape)
    
    
    
    L = torch.tensor(true_L)
    print("L:",L.shape)
    print(L)
    
    
    
    X = torch.cat((A,B),dim=1)
    
    
    #print(split1,split2)
    
    
    #X_train = X[:split1,:]
    X_train = X[split2:,:]
    
    
    
    X_test = X[split1:split2,:]
    X_val = X[:split1,:]
    
    #L_train = L[:split1]
    L_train = L[split2:]
    L_test = L[split1:split2]
    #L_test = L[split2:]
    L_val = L[:split1]
    
    #randomize train set order to aid in batches
    temp = list(zip(X_train.tolist(), L_train))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    X_train_list, L_train_list = list(res1), list(res2)
    
    X_train = torch.tensor(X_train_list)
    L_train = torch.tensor(L_train_list)
    
    print("X_train shuffled:",X_train.shape)
    print("L_train shuffled:",L_train.shape)
    
    if not os.path.exists("data4"):
        os.mkdir("data4")
    if not os.path.exists("data4/dataset"):
        os.mkdir("data4/dataset")
    if not os.path.exists("data4/exact_results"):
        os.mkdir("data4/exact_results")
    if not os.path.exists("data4/exact_results/"):
        os.mkdir("data4/dataset")

    outfile_a = open("data4/dataset/A_values.txt",'w')
    for row in A.tolist():
        for item in row:
            outfile_a.write(str(f'{item:.9f}'))
            outfile_a.write(" ")
        outfile_a.write("\n")
    outfile_a.close()
    
    outfile_b = open("data4/dataset/B_values.txt",'w')
    for row in B.tolist():
        for item in row:
            outfile_b.write(str(f'{item:.9f}'))
            outfile_b.write(" ")
        outfile_b.write("\n")
    outfile_b.close()
    
    
    outfile_a_train = open("data4/dataset/A_values_train.txt",'w')
    #outfile_a_test.write("[")
    #for row in A.tolist()[:split1]:
    for row in A.tolist()[split2:]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_train.write(str(f'{item:.9f}'))
            outfile_a_train.write(" ")
        #outfile_a_test.write("]")
        outfile_a_train.write("\n")
    outfile_a_train.close()

    
    outfile_b_train = open("data4/dataset/B_values_train.txt",'w')
    #for row in B.tolist()[:split1]:
    for row in B.tolist()[split2:]:
        for item in row:
            outfile_b_train.write(str(f'{item:.9f}'))
            outfile_b_train.write(" ")
        outfile_b_train.write("\n")
    outfile_b_train.close()

    outfile_a_test = open("data4/dataset/A_values_test.txt",'w')
    #outfile_a_test.write("[")
    #for row in A.tolist()[split2:]:
    for row in A.tolist()[split1:split2]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_test.write(str(f'{item:.9f}'))
            outfile_a_test.write(" ")
        #outfile_a_test.write("]")
        outfile_a_test.write("\n")
    outfile_a_test.close()

    
    outfile_b_test = open("data4/dataset/B_values_test.txt",'w')
    #for row in B.tolist()[split2:]:
    for row in B.tolist()[split1:split2]:
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    
    
    outfile_a_test = open("data4/dataset/A_values_test_transpose.txt",'w')
    for row in A[split1:split2,:].T.tolist():
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_test.write(str(f'{item:.9f}'))
            outfile_a_test.write(" ")
        #outfile_a_test.write("]")
        outfile_a_test.write("\n")
    outfile_a_test.close()

    
    outfile_b_test = open("data4/dataset/B_values_test_transpose.txt",'w')
    for row in B[split1:split2,:].T.tolist():
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    

    
    outfile_a_val = open("data4/dataset/A_values_val.txt",'w')
    for row in A.tolist()[:split1]:
        outfile_a_val.write("[")
        for item in row:
            outfile_a_val.write(str(f'{item:.9f}'))
            outfile_a_val.write(" ")
        outfile_a_val.write("]")
        outfile_a_val.write("\n")
    outfile_a_val.close()
    
    
    outfile_b_val = open("data4/dataset/B_values_val.txt",'w')
    for row in B.tolist()[:split1]:
        outfile_b_val.write("[")
        for item in row:
            outfile_b_val.write(str(f'{item:.9f}'))
            outfile_b_val.write(" ")
        outfile_b_val.write("]")
        outfile_b_val.write("\n")
    outfile_b_val.close()
    

    
    outfile_x_test = open("data4/dataset/X_values_test.txt",'w')
    #for row in X.tolist()[split2:]:
    for row in X.tolist()[split1:split2]:
        outfile_x_test.write("[")
        for item in row:
            outfile_x_test.write(str(f'{item:.9f}'))
            outfile_x_test.write(" ")
        outfile_x_test.write("]")
        outfile_x_test.write("\n")
    outfile_x_test.close()
    
    outfile_L = open("data4/dataset/L_values_val.txt",'w')
    outfile_L.write(str(L_val.tolist()))
    outfile_L.close()
    
    outfile_L = open("data4/dataset/L_values_test.txt",'w')
    outfile_L.write(str(L_test.tolist()))
    outfile_L.close()
    
    #Hyperparameters
    #gamma = 512
    lambs = [0.01,0.1,0.25,0.5,0.75,0.99]
    #lambs = [0.01,0.1,0.25,0.5]
    #lambs = [0.1,0.25,0.5]
    #lambs = [0.5]
    #margins = [0.0,0.25,0.5]
    margins = [0.1,0.25,0.5,0.75,1.0]
    #lambs = [0.25]
    #margins = [0.1]
    #margins = [1.0]
    #margins = [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]
    #margins = [0.5,0.75,1.0]
    #lambs = [0.1]
    iterations = iters

    regularizers = [0]

    rates = {2:1000, 32:500, 64:10000, 128: 10000, 256:10000, 512:10000}
    #rates[64] = 500
    rates[64] = 20000
    rates[64] = 0.005
    #rates[64] = 500
    #rates[64] = 50
    anneal_rate = 0.995
    anneal_rate = 0.9999
    decay = 0.0001
    
    #anneal_rate = 0.8
    
    #anneal_rate = 1
    #anneal_rate = 0.9
    #anneal_rate = 0.90
    if spec_margin:
        margins = [spec_margin]
    if spec_lamb:
        lambs = [spec_lamb]
    
    
    f = 0
    randoms = [i*5 for i in range(21,31)]
    
    #lamb = 0.2
    aucs = []
    for reg in regularizers:
        for margin in margins:
            for lamb in lambs:
                rate = rates[gamma]
                print("Lambda value of", lamb, "Margin value of", margin, "regularizer of",reg)
                
                M = []
                V = []
                """
                for i in range(X_train.shape[0]):
                    for j in range(i+1,X_train.shape[0]):
                        if L_train[i] == L_train[j]:
                            M.append((i,j))
                            for k in range(X_train.shape[0]):
                                if L_train[k] != L_train[i]:
                                    V.append((i,j,k))
                                    
                print("Same class",len(M))
                print("Trios:",len(V))"""
                #randie = randoms[f]
                #f+=1
                #randie = 24
                randie = 30
                randie = 0
                randie = 123
                print("seed of:",randie)
                model = Linear_Feature_Fusion_Batch(X_train,M,V,L_train,gamma,margin,lamb,regularization=reg,seed=randie)
                #model = Linear_Feature_Fusion(X_train,M,V,gamma,margin,lamb,regularization=reg)
                
                best_loss = model.loss()
                best_P = model.P
                #best_scale = model.scale
                print("Initial loss:",best_loss)
                #P_history = []
                #P_history_matrices = []
                losses = []
                #optimizer = torch.optim.SGD(model.parameters(), lr=rate)#, momentum=0.9)
                optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=decay)#, momentum=0.9)
                #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=anneal_rate)
                #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=anneal_rate)
                for i in range(iterations):
                    

                    loss = model.loss()
                    #print("Loss:",loss)
                    if loss <= best_loss:#+0.003
                        best_loss = loss
                        best_P = model.P
                        #best_scale = model.scale

                    losses.append(loss)

                    loss.backward()
                    optimizer.step()
                    #scheduler.step()
                    #model.P = torch.mul(torch.div(model.P,torch.linalg.norm(model.P)),3.0)
                    if i%10 == 0:
                        print("Iteration",str(i) + "/" + str(iterations))
                        print(model.loss())

                #best_P = torch.div(best_P,torch.linalg.norm(best_P))



                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                #print("new NOT normalized validation X_prime:", X_prime)
                #print("one vec:",X_prime[0,:].shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                auc = New_ROC_AUC(X_prime, L_val)
                print("approx AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)


                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)

                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                #auc = New_ROC_AUC(X_prime, L_val)
                #print("EXACTED AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                
                #X_prime = torch.mm(X_val,best_P)
                #print(X_prime.shape)

                
                auc = New_ROC_AUC(X_prime, L_val)
                aucs.append((margin,lamb,auc))
                print("AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)

                p_best_file_name = "data4/exact_results/exact_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #p_best_file_name = "data/exact_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_p_t = open(p_best_file_name,'w')
                P_final_t = best_P.T
                P_final_t = str(P_final_t.tolist())
                outfile_p_t.write(P_final_t)
                outfile_p_t.close()
                
                loss_file_name = "data4/exact_results/exact_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #loss_file_name = "data/exact_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_loss = open(loss_file_name,'w')
                for loss_value in losses:
                    outfile_loss.write(str(loss_value.tolist()))
                    outfile_loss.write("\n")
                outfile_loss.close()

                X_prime_filename = "data4/exact_results/exact_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_val.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                #x_str = str(x_list)
                #outfile_x.write(x_str)
                outfile_x.close()
                
                #X_prime = torch.mm(X_val,P_history_matrices[-1])
                X_prime = torch.mm(X_test,best_P)
                
                
                X_filename = "data4/exact_results/exact_labels_X_NOT_NORMAL_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
                
                
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    #X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                #print("new normalized test x_prime:", X_prime)
                
                X_prime_filename = "data4/exact_results/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
    print("(margin, lambda, Validation AUC)")
    print(aucs)

def train(gamma,iters,spec_margin=None,spec_lamb=None):
    random.seed(123)
    torch.manual_seed(123)
    
    a = []
    A_infile = open("feature-extraction/extractions/VGGFace_vgg_cplfw.txt",'r')
    for line in A_infile:
        line = line.strip().split()
        #print(torch.linalg.norm(torch.tensor([float(char) for char in line])))
        a.append(torch.tensor([float(char) for char in line]))

    
    L = []
    #L_infile = open("feature-extraction/extractions/deep_speaker_librispeech_big_labels.txt",'r')
    L_infile = open("feature-extraction/extractions/deep_speaker_librispeech_google_labels.txt",'r')
    
    
    
    
    l_dict = {}
    for line in L_infile:
        line = line.strip()
        if line not in l_dict:
            l_dict[line] = 0
        l_dict[line] += 1
        
        L.append(line.strip())
        #print(d[line])
        #L.append(d[line])
        #L.append(int(line.strip()))
    #print(l_dict)
    #print(min(l_dict.values()))
    
    #convert from unique string labels to integers
    #from https://stackoverflow.com/questions/43203215/map-unique-strings-to-integers-in-python
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(L)))])
    L = [d[x] for x in L]
    
    l_dict = {}
    for l in L:
        if l not in l_dict:
            l_dict[l] = 0
        l_dict[l] += 1
    #print(L)
    
    b = []
    #B_infile = open("feature-extraction/extractions/deep_speaker_librispeech_big.txt",'r')
    #B_infile = open("feature-extraction/extractions/vggvox_librispeech.txt",'r')
    B_infile = open("feature-extraction/extractions/deep_speaker_librispeech_google.txt",'r')
    for line in B_infile:
        line = line.strip().split()
        #print(torch.linalg.norm(torch.tensor([float(char) for char in line])))
        b.append(torch.tensor([float(char) for char in line]))
    
    
    b, L = (list(t) for t in zip(*sorted(zip(b, L),key=itemgetter(1))))
    
    #num_classes = 275
    #num_each_class = 3

    
    completed = 0
    a_index = 0
    a_sub_index = 0
    b_index = 0
    samples_per_face = 2
    samples_per_voice = 10
    num_each_class = samples_per_face * samples_per_voice
    true_a = []
    true_b = []
    true_L = []
    
    
    num_classes = 188
    split1 = 0
    split2 = 0
    
    last_l = L[0]
    i = 0
    while completed <= 188:
        #for i, label in enumerate(L):
        label = L[b_index]
        for j in range(2):
            a_sub_index = j
            #b_index = i
            #print(label, a_index+a_sub_index)
            print(a_index+a_sub_index, b_index, label)
            true_a.append(a[a_index+a_sub_index])
            true_b.append(b[b_index])
            true_L.append(L[b_index])
        b_index += 1
        
        if b_index >= len(L):
            break
        label = L[b_index]
        #if a_sub_index == 1:
            #b_index += 1
            #a_index += 2
        #a_sub_index = 1-a_sub_index
        if label != last_l:
            last_l = label
            a_index += 2
            completed += 1
            if split1 == 0 and a_index/2 >= math.floor(num_classes * 0.2):
                split1 = i
            if split2 == 0 and a_index/2 >= math.floor(num_classes * 0.4):
                split2 = i
        i += 1
        #a_index += 2
        #completed += 1
        
    print(len(true_a), len(true_b))
    
    
    """
    print("pairs")
    while completed < num_classes:
        true_a.append(a[a_index+a_sub_index])
        true_b.append(b[b_index])
        true_L.append(L[b_index])
        print(a_index+a_sub_index, b_index, L[b_index])
        a_sub_index += 1
        if a_sub_index >= samples_per_face:
            a_sub_index = 0
            b_index += 1
            if b_index % samples_per_voice == 0:
                a_index += samples_per_face
                completed += 1
    """
    #split1 = math.floor(num_classes * 0.6)*num_each_class
    #split2 = math.floor(num_classes * 0.8)*num_each_class
    
    #split1 = math.floor(num_classes * 0.2)*num_each_class
    #split2 = math.floor(num_classes * 0.4)*num_each_class
    print("Splits:",split1,split2)
    
    A = torch.stack(true_a)
    print("A:",A.shape)
    B = torch.stack(true_b)
    print("B:",B.shape)
    
    
    
    L = torch.tensor(true_L)
    print("L:",L.shape)
    print(L)
    
    
    
    X = torch.cat((A,B),dim=1)
    
    
    #print(split1,split2)
    
    
    #X_train = X[:split1,:]
    X_train = X[split2:,:]
    
    
    
    X_test = X[split1:split2,:]
    X_val = X[:split1,:]
    
    #L_train = L[:split1]
    L_train = L[split2:]
    L_test = L[split1:split2]
    #L_test = L[split2:]
    L_val = L[:split1]
    
    #randomize train set order to aid in batches
    temp = list(zip(X_train.tolist(), L_train))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    X_train_list, L_train_list = list(res1), list(res2)
    
    X_train = torch.tensor(X_train_list)
    L_train = torch.tensor(L_train_list)
    
    print("X_train shuffled:",X_train.shape)
    print("L_train shuffled:",L_train.shape)
    
    
    

    
    if not os.path.exists("data4/degree=3strict"):
        os.mkdir("data4/degree=3strict")
    if not os.path.exists("data4/degree=2strict"):
        os.mkdir("data4/degree=2strict")
    
    
    #Hyperparameters
    #gamma = 512
    lambs = [0.01,0.1,0.25,0.5,0.75,0.99]
    #lambs = [0.01,0.1,0.25,0.5]
    #lambs = [0.1,0.25,0.5]
    #lambs = [0.5]
    #margins = [0.0,0.25,0.5]
    margins = [0.1,0.25,0.5,0.75,1.0]
    #lambs = [0.25]
    #margins = [0.1]
    #margins = [1.0]
    #margins = [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]
    #margins = [0.5,0.75,1.0]
    #lambs = [0.1]
    iterations = iters

    regularizers = [0]

    rates = {2:1000, 32:500, 64:10000, 128: 10000, 256:10000, 512:10000}
    #rates[64] = 500
    rates[64] = 20000
    rates[64] = 0.005
    #rates[64] = 0.0005
    #rates[64] = 0.05
    #rates[64] = 500
    #rates[64] = 50
    anneal_rate = 0.995
    anneal_rate = 0.9999
    decay = 0.0001
    #decay = 0
    #anneal_rate = 0.8
    if spec_margin:
        margins = [spec_margin]
    if spec_lamb:
        lambs = [spec_lamb]
    
    
    #anneal_rates = [1,0.9999,0.995,0.95,0.9]
    f = 0
    randoms = [i*5 for i in range(21,31)]
    special_counter = 0
    #lamb = 0.2
    aucs = []
    for reg in regularizers:
        for margin in margins:
            for lamb in lambs:
                rate = rates[gamma]
                #anneal_rate = anneal_rates[special_counter]
                #special_counter += 1
                print("Lambda value of", lamb, "Margin value of", margin, "regularizer of",reg)
                
                M = []
                V = []
                """
                for i in range(X_train.shape[0]):
                    for j in range(i+1,X_train.shape[0]):
                        if L_train[i] == L_train[j]:
                            M.append((i,j))
                            for k in range(X_train.shape[0]):
                                if L_train[k] != L_train[i]:
                                    V.append((i,j,k))
                print("Same class",len(M))
                print("Trios:",len(V))
                """
                #randie = randoms[f]
                #f+=1
                #randie = 24
                randie = 30
                randie = 1
                randie = 0
                randie = 123
                #randie = 2
                print("seed of:",randie)
                print("rate, anneal_rate:", rate, anneal_rate)
                #model = Linear_Feature_Fusion_Approximate2(X_train,M,V,gamma,margin,lamb,regularization=reg,seed=randie)
                model = Linear_Feature_Fusion_Approximate2_Batch(X_train,M,V,L_train,gamma,margin,lamb,regularization=reg,seed=randie)
                #model = Linear_Feature_Fusion(X_train,M,V,gamma,margin,lamb,regularization=reg)
                
                
                #first we select a scale such that our polynomial function will work
                total = 0.0
                P_temp = torch.div(model.P,torch.linalg.norm(model.P))
                min_norm = 10000
                for c in range(X_train.shape[0]):
                    norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                    total += norm
                total = float(total)
                avg = total / X_train.shape[0]
                model.scale = 0.7245688373 / avg #0.525 is the median of our valid range, therefore we want squared norm to be 0.525, therefore we use the sqrt of 0.525 as the numerator to make the new norm that
                model.scale = 0.4 / avg
                model.scale = 0.5 / avg
                model.scale = 1.525**0.5 / avg
                #model.scale = 0.304764**0.5 / avg
                
                #model.scale = 0.7245688373 / avg
                
                
                best_loss = model.loss()
                best_P = model.P
                best_scale = model.scale
                print("Initial loss:",best_loss)
                #P_history = []
                #P_history_matrices = []
                losses = []
                optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=decay)#, momentum=0.9)
                #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=anneal_rate)
                for i in range(iterations):
                    
                    #first we select a scale such that our polynomial function will work
                    total = 0.0
                    P_temp = torch.div(model.P,torch.linalg.norm(model.P))
                    min_norm = 10000
                    for c in range(X_train.shape[0]):
                        norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                        total += norm
                        if norm < min_norm:
                            min_norm = norm
                    min_norm = float(min_norm)
                    total = float(total)
                    #print("total",total)
                    avg = total / X_train.shape[0]
                    
                    #print("average:",avg)
                    #model.scale = 0.63245553 / avg #0.4 is the median of our valid range, therefore we want squared norm to be 0.4, therefore we use the sqrt of 0.4 as the numerator to make the new norm that
                    model.scale = 0.7245688373 / avg #0.525 is the median of our valid range, therefore we want squared norm to be 0.525, therefore we use the sqrt of 0.525 as the numerator to make the new norm that
                    model.scale = 0.4 / avg
                    model.scale = 0.5 / avg
                    model.scale = 1.525**0.5 / avg
                    #model.scale = 0.304764**0.5 / avg
                    
                    
                    #model.scale = 0.7245688373 / avg
                    #P_temp = torch.mul(P_temp,model.scale)
                    #for c in range(X_train.shape[0]):
                        #norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                        #print(norm**2)
                        
                    
                    #model.scale = 0.59581876 / avg #0.355 is the median of our valid range, therefore we want squared norm to be 0.4, therefore we use the sqrt of 0.4 as the numerator to make the new norm that
                    
                    #model.scale = 0.316227766 / min_norm #0.1 is the lowest of our valid range, therefore we can scale everything up to within our range
                    #model.scale = 0.70710678118 / avg
                    #print("new scale:",model.scale)
                    #P_temp = torch.mul(P_temp,model.scale)
                    #for c in range(X_train.shape[0]):
                        #print(torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))**2)
                    
                    
                    #model.P = torch.div(model.P, torch.linalg.norm(model.P))
                    #model.P = torch.mul(model.P, model.scale)
                    
                    loss = model.loss()
                    #print("Loss:",loss)
                    if loss <= best_loss:#+0.003
                        best_loss = loss
                        best_P = model.P
                        best_scale = model.scale
                    #else:
                        #print("Loss increased, ending training")
                        #break
                    losses.append(loss)
                    #P_history.append(str(model.P.tolist()))
                    #P_history_matrices.append(model.P)
                    loss.backward()
                    optimizer.step()
                    #scheduler.step()
                    #model.P = torch.mul(torch.div(model.P,torch.linalg.norm(model.P)),3.0)
                    if i%10 == 0:
                        print("Iteration",str(i) + "/" + str(iterations))
                        print(loss)
                    #rate *= anneal_rate
                #print("final loss:",model.loss())
                #print("old best p norm:",torch.linalg.norm(best_P))
                best_P = torch.div(best_P,torch.linalg.norm(best_P))
                
                print("percentage of escape:",model.escape/model.tote)
                
                #calculate scale for best P
                total = 0.0
                min_norm = 10000
                for c in range(X_train.shape[0]):
                    norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                    total += norm
                total = float(total)
                avg = total / X_train.shape[0]
                best_scale = 0.7245688373 / avg
                best_scale = 0.5 / avg
                best_scale = 1.525**0.5 / avg
                #best_scale = 0.304764**0.5 / avg
                
                #best_scale = 0.59581876 / avg
                #best_scale = 0.316227766 / min_norm
                #best_scale = 0.70710678118 / avg
                
                best_P = torch.mul(best_P, best_scale)
                
                
                
                
                
                
                #for i in range(best_P.shape[1]):
                    #best_P[:,i]=torch.div(best_P[:,i], torch.linalg.norm(best_P[:,1]))
                #print("new best p norm:",torch.linalg.norm(best_P))
                
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                #print("new NOT normalized validation X_prime:", X_prime)
                #print("one vec:",X_prime[0,:].shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                auc = New_ROC_AUC(X_prime, L_val)
                print("EXACTED AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                #print("new NOT normalized validation X_prime:", X_prime)
                #print("one vec:",X_prime[0,:].shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                    #print("new norm:",torch.linalg.norm(X_prime[i,:]))
                    #print(X_prime[i,:])
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    #print(X_prime[i,:])
                    #print()
                
                auc = New_ROC_AUC(X_prime, L_val)
                aucs.append((margin,lamb,auc,best_scale))
                print("AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                #P values file gets too large for github
                #p_file_name = "data/features_P_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + ".txt"
                #outfile_p = open(p_file_name,'w')
                #for P_value in P_history:
                    #outfile_p.write(P_value)
                    #outfile_p.write("\n")
                #outfile_p.close()
                
                p_best_file_name = "data4/degree=2strict/large_approximate_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #p_best_file_name = "data/exact_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_p_t = open(p_best_file_name,'w')
                P_final_t = best_P.T
                P_final_t = str(P_final_t.tolist())
                outfile_p_t.write(P_final_t)
                outfile_p_t.close()
                
                loss_file_name = "data4/degree=2strict/large_approximate_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #loss_file_name = "data/exact_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_loss = open(loss_file_name,'w')
                for loss_value in losses:
                    outfile_loss.write(str(loss_value.tolist()))
                    outfile_loss.write("\n")
                outfile_loss.close()
                #print()
                
                #X_prime = torch.mm(X_val,P_history_matrices[-1])
                #X_prime = torch.mm(X_val,best_P)
                #for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                #print("new normalized validation X_prime:", X_prime)
                
                #print("norm of best P:",torch.linalg.norm(best_P))
                
                X_prime_filename = "data4/degree=2strict/large_approximate_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_val.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                #x_str = str(x_list)
                #outfile_x.write(x_str)
                outfile_x.close()
                
                #X_prime = torch.mm(X_val,P_history_matrices[-1])
                X_prime = torch.mm(X_test,best_P)
                
                
                X_filename = "data4/degree=2strict/large_approximate_labels_X_NOT_NORMAL_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
                
                
                for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                #print("new normalized test x_prime:", X_prime)
                
                X_prime_filename = "data4/degree=2strict/large_approximate_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
    print("(margin, lambda, Validation AUC), Scale")
    print(aucs)
    
    
    
if __name__ == "__main__":
    #train_exact(64,250,0.25,0.25) #auc of 0.9549
    
    #train(256,100)
    #train(64,1000,0.25,0.1)
    
    #train(64,250)
    #train(64,250,0.1)
    
    
    #train_exact(32,250) #0.1,0.1, auc of 0.9595
    
    
    #train_exact(32,1000,0.1,0.01) #lambda = 0.01, margin = 0.1, auc of 0.9605, I think like 0.965 now that it's 1k epochs instead of 400
    
    #train(32,250) #lambda = 0.01, margin = 0.1, auc of 0.9587
    #train(32,1000,0.1,0.01) #lambda = 0.01, margin = 0.1, auc of 0.9547, test auc of 0.9380
    
    #train(64,1000,0.25)
    
    #Overnight Tests :)
    #train_exact(64,1000)
    #train(64,1000)
    
    
    train_exact(32,1000)
    #train(32,1000)
    #train(32,1000,0.25)

