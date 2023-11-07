import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import math
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import random_split
from Datasets.DataGenerator import Dataset_690


class Constructor:
    """
        Using CNN and self-attention mechanism to extract features in an interactive way
    """

    def __init__(self, model, model_name='dss_ctba'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=1)
        self.loss_function = nn.BCELoss()

        self.batch_size = 64
        self.epochs = 15

    def train(self, TrainLoader, ValidateLoader):
        path = os.path.abspath(os.curdir)
        best = 1
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)
                seq, shape, label = data
                output1, output2 = self.model(seq.unsqueeze(1).to(self.device), shape.unsqueeze(1).to(self.device))
                loss = 0.5 * self.loss_function(output1, label.float().to(self.device)) + 0.5 * self.loss_function(
                    output2, label.float().to(self.device))
                #output=self.model(seq.unsqueeze(1).to(self.device))
                #loss = self.loss_function(output, label.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())

                loss.backward()
                self.optimizer.step()



            validate_loss = self.validate(ValidateLoader)

            # self.model.eval()
            # with torch.no_grad():
            #     for valid_seq, valid_shape, valid_labels in ValidateLoader:
            #         valid_output = self.model(valid_seq.unsqueeze(1).to(self.device),
            #                                   valid_shape.unsqueeze(1).to(self.device))
            #         valid_labels = valid_labels.float().to(self.device)
            #
            #         validate_loss.append(self.loss_function(valid_output, valid_labels).item())
            #
            #     valid_loss_avg = torch.mean(torch.Tensor(validate_loss))
            #     self.scheduler.step(valid_loss_avg)
            if validate_loss < best:
                best = validate_loss
                # model_name = path + '\\' + self.model_name + 'epoch' + str(epoch) + '.pth'
                model_name = path + '\\' + self.model_name + '.pth'

        torch.save(self.model.state_dict(), model_name)
        # torch.save(self.model.state_dict(), path + '\\' + self.model_name + '.pth')
        return model_name

        print('Complete training and validation!\n')

    def validate(self, ValidateLoader):
        path = os.path.abspath(os.curdir)
        # self.model.load_state_dict(torch.load(path + '\\' + self.model_name + '.pth', map_location='cpu'))

        valid_loss = []
        predicted = []
        ground_label = []
        true = []
        self.model.eval()
        with torch.no_grad():
            for valid_seq, valid_shape, valid_labels in ValidateLoader:
                valid_output1, valid_output2 = self.model(valid_seq.unsqueeze(1).to(self.device), valid_shape.unsqueeze(1).to(self.device))
                '''print((0.5*valid_output1+0.5*valid_output2).size())
                predicted.append((0.5*valid_output1+0.5*valid_output2).squeeze(dim=1))
                true.append(valid_labels)'''
                #valid_output = self.model(valid_seq.unsqueeze(1).to(self.device))
                predicted.append((0.5*valid_output1+0.5*valid_output2).squeeze(dim=0).detach().cpu().numpy())
                valid_labels = valid_labels.float().to(self.device)
                #valid_loss.append(self.loss_function(valid_output, valid_labels))
                valid_loss.append((0.5 * self.loss_function(valid_output1, valid_labels) +
                0.5 * self.loss_function(valid_output2, valid_labels)).item())
            valid_loss_avg = torch.mean(torch.Tensor(valid_loss))
            self.scheduler.step(valid_loss_avg)
        return valid_loss_avg

        # for seq, shape, label in ValidateLoader:
        #     output1, output2 = self.model(seq.unsqueeze(1), shape.unsqueeze(1))
        #     """ To scalar"""
        #     predicted.append((0.5*output1+0.5*output2).squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())
        #     ground_label.append(label.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())

        # print('\n---Finish Inference---\n')

        # return predicted, ground_label

    def test(self, TestLoader, model_name):

        self.model.load_state_dict(torch.load(model_name))
        predicted_value = []
        true_label = []
        self.model.eval()
        for seq, shape, label in TestLoader:
            output1, output2 = self.model(seq.unsqueeze(1), shape.unsqueeze(1))  #
            output = 0.5 * output1 + 0.5 * output2
            #output=self.model(seq)#.unsqueeze(1)
            predicted_value.append(output.squeeze(dim=0).squeeze(dim=0).detach().numpy())
            true_label.append(label.squeeze(dim=0).squeeze(dim=0).detach().numpy())
        print('Complete test!\n')
        return predicted_value, true_label

    def calculate(self, predicted_value, true_label):
        accuracy = accuracy_score(y_pred=np.array(predicted_value).round(), y_true=true_label)
        roc_auc = roc_auc_score(y_score=predicted_value, y_true=true_label)

        precision, recall, _ = precision_recall_curve(probas_pred=predicted_value, y_true=true_label)
        pr_auc = auc(recall, precision)
        print(accuracy, roc_auc, pr_auc)
        return accuracy, roc_auc, pr_auc

    def run(self, file_name, ratio=0.8):



        Train_Validate_Set = Dataset_690(file_name, False)

        """divide Train samples and Validate samples"""
        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) -
                                                        math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(0))

        TrainLoader = loader.DataLoader(dataset=Train_Set, drop_last=True,
                                        batch_size=self.batch_size, shuffle=True, num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, drop_last=True,
                                           batch_size=self.batch_size, shuffle=False, num_workers=0)

        TestLoader = loader.DataLoader(dataset=Dataset_690(file_name, True),
                                       batch_size=1, shuffle=False, num_workers=0)

        model_name = self.train(TrainLoader, ValidateLoader)

        predicted_value, true_label = self.test(TestLoader, model_name)

        accuracy, roc_auc, pr_auc = self.calculate(predicted_value, true_label)


        return accuracy, roc_auc, pr_auc


from models.DSS_CTBA import dss_ctba

Train = Constructor(model=dss_ctba())
data_name = ['wgEncodeAwgTfbsHaibK562Fosl1sc183V0416101UniPk']
# data_name = ['wgEncodeAwgTfbsBroadHelas3Ezh239875UniPk',
#              'wgEncodeAwgTfbsBroadHelas3Pol2bUniPk', 'wgEncodeAwgTfbsBroadHepg2CtcfUniPk', 'wgEncodeAwgTfbsBroadHepg2Ezh239875UniPk',
#              'wgEncodeAwgTfbsBroadHmecCtcfUniPk', 'wgEncodeAwgTfbsBroadHsmmCtcfUniPk', 'wgEncodeAwgTfbsBroadHsmmEzh239875UniPk',
#              'wgEncodeAwgTfbsBroadHsmmtCtcfUniPk', 'wgEncodeAwgTfbsBroadHuvecCtcfUniPk', 'wgEncodeAwgTfbsBroadHuvecEzh239875UniPk',
#              'wgEncodeAwgTfbsBroadHuvecPol2bUniPk', 'wgEncodeAwgTfbsBroadK562CtcfUniPk', 'wgEncodeAwgTfbsBroadK562Ezh239875UniPk',
#              'wgEncodeAwgTfbsBroadK562Hdac2a300705aUniPk', 'wgEncodeAwgTfbsBroadK562Hdac6a301341aUniPk', 'wgEncodeAwgTfbsBroadNhaCtcfUniPk',
#              'wgEncodeAwgTfbsBroadNhdfadCtcfUniPk', 'wgEncodeAwgTfbsBroadNhekCtcfUniPk', 'wgEncodeAwgTfbsBroadNhekPol2bUniPk',
#              'wgEncodeAwgTfbsBroadNhlfCtcfUniPk', 'wgEncodeAwgTfbsBroadOsteoblCtcfUniPk', 'wgEncodeAwgTfbsHaibA549Elf1V0422111Etoh02UniPk',
#              'wgEncodeAwgTfbsHaibA549Fosl2V0422111Etoh02UniPk', 'wgEncodeAwgTfbsHaibA549Foxa1V0416102Dex100nmUniPk',
#              'wgEncodeAwgTfbsHaibA549GabpV0422111Etoh02UniPk', 'wgEncodeAwgTfbsHaibA549Sin3ak20V0422111Etoh02UniPk',
#              'wgEncodeAwgTfbsHaibA549Taf1V0422111Etoh02UniPk', 'wgEncodeAwgTfbsHaibEcc1Foxa1sc6553V0416102Dm002p1hUniPk',
#              'wgEncodeAwgTfbsHaibGm12878Elf1sc631V0416101UniPk', 'wgEncodeAwgTfbsHaibGm12878GabpPcr2xUniPk', 'wgEncodeAwgTfbsHaibGm12878P300Pcr1xUniPk',
#              'wgEncodeAwgTfbsHaibGm12878Pax5c20Pcr1xUniPk', 'wgEncodeAwgTfbsHaibGm12878Pax5n19Pcr1xUniPk', 'wgEncodeAwgTfbsHaibGm12878Taf1Pcr1xUniPk',
#              'wgEncodeAwgTfbsHaibGm12878Yy1sc281Pcr1xUniPk', 'wgEncodeAwgTfbsHaibGm12891Pax5c20V0416101UniPk', 'wgEncodeAwgTfbsHaibGm12892Pax5c20V0416101UniPk',
#              'wgEncodeAwgTfbsHaibGm12892Taf1V0416102UniPk', 'wgEncodeAwgTfbsHaibGm12892Yy1V0416101UniPk', 'wgEncodeAwgTfbsHaibH1hescFosl1sc183V0416102UniPk',
#              'wgEncodeAwgTfbsHaibH1hescGabpPcr1xUniPk', 'wgEncodeAwgTfbsHaibH1hescHdac2sc6296V0416102UniPk', 'wgEncodeAwgTfbsHaibH1hescSp1Pcr1xUniPk',
#              'wgEncodeAwgTfbsHaibH1hescSp2V0422111UniPk', 'wgEncodeAwgTfbsHaibH1hescSrfPcr1xUniPk', 'wgEncodeAwgTfbsHaibH1hescTead4sc101184V0422111UniPk',
#              'wgEncodeAwgTfbsHaibH1hescYy1sc281V0416102UniPk', 'wgEncodeAwgTfbsHaibHct116Yy1sc281V0416101UniPk', 'wgEncodeAwgTfbsHaibHelas3GabpPcr1xUniPk',
#              'wgEncodeAwgTfbsHaibHepg2Cebpbsc150V0416101UniPk', 'wgEncodeAwgTfbsHaibHepg2Elf1sc631V0416101UniPk', 'wgEncodeAwgTfbsHaibHepg2Fosl2V0416101UniPk',
#              'wgEncodeAwgTfbsHaibHepg2Foxa1sc101058V0416101UniPk', 'wgEncodeAwgTfbsHaibHepg2Foxa1sc6553V0416101UniPk', 'wgEncodeAwgTfbsHaibHepg2GabpPcr2xUniPk',
#              'wgEncodeAwgTfbsHaibHepg2Hdac2sc6296V0416101UniPk', 'wgEncodeAwgTfbsHaibHepg2JundPcr1xUniPk', 'wgEncodeAwgTfbsHaibHepg2P300V0416101UniPk',
#              'wgEncodeAwgTfbsHaibHepg2Sin3ak20Pcr1xUniPk', 'wgEncodeAwgTfbsHaibHepg2Sp1Pcr1xUniPk', 'wgEncodeAwgTfbsHaibHepg2Sp2V0422111UniPk',
#              'wgEncodeAwgTfbsHaibHepg2SrfV0416101UniPk', 'wgEncodeAwgTfbsHaibHepg2Taf1Pcr2xUniPk', 'wgEncodeAwgTfbsHaibHepg2Tead4sc101184V0422111UniPk',
#              'wgEncodeAwgTfbsHaibHepg2Yy1sc281V0416101UniPk', 'wgEncodeAwgTfbsHaibK562Cebpbsc150V0422111UniPk', 'wgEncodeAwgTfbsHaibK562Elf1sc631V0416102UniPk',
#              'wgEncodeAwgTfbsHaibK562Fosl1sc183V0416101UniPk', 'wgEncodeAwgTfbsHaibK562GabpV0416101UniPk', 'wgEncodeAwgTfbsHaibK562Gata2sc267Pcr1xUniPk',
#              'wgEncodeAwgTfbsHaibK562Hdac2sc6296V0416102UniPk', 'wgEncodeAwgTfbsHaibK562MaxV0416102UniPk', 'wgEncodeAwgTfbsHaibK562Sin3ak20V0416101UniPk',
#              'wgEncodeAwgTfbsHaibK562SrfV0416101UniPk', 'wgEncodeAwgTfbsHaibK562Taf1V0416101UniPk', 'wgEncodeAwgTfbsHaibK562Tead4sc101184V0422111UniPk',
#              'wgEncodeAwgTfbsHaibK562Yy1V0416101UniPk', 'wgEncodeAwgTfbsHaibK562Yy1V0416102UniPk', 'wgEncodeAwgTfbsHaibPanc1Sin3ak20V0416101UniPk',
#              'wgEncodeAwgTfbsHaibSknshraP300V0416102UniPk', 'wgEncodeAwgTfbsHaibSknshraYy1sc281V0416102UniPk', 'wgEncodeAwgTfbsHaibSknshTaf1V0416101UniPk',
#              'wgEncodeAwgTfbsHaibT47dFoxa1sc6553V0416102Dm002p1hUniPk', 'wgEncodeAwgTfbsHaibT47dGata3sc268V0416102Dm002p1hUniPk',
#              'wgEncodeAwgTfbsHaibT47dP300V0416102Dm002p1hUniPk', 'wgEncodeAwgTfbsSydhA549CebpbIggrabUniPk', 'wgEncodeAwgTfbsSydhGm10847NfkbTnfaIggrabUniPk',
#              'wgEncodeAwgTfbsSydhGm12878CfosUniPk', 'wgEncodeAwgTfbsSydhGm12878JundUniPk', 'wgEncodeAwgTfbsSydhGm12878MaxIggmusUniPk',
#              'wgEncodeAwgTfbsSydhGm12878NfkbTnfaIggrabUniPk', 'wgEncodeAwgTfbsSydhGm12878P300bUniPk', 'wgEncodeAwgTfbsSydhGm12878Stat3IggmusUniPk',
#              'wgEncodeAwgTfbsSydhGm12878TbpIggmusUniPk', 'wgEncodeAwgTfbsSydhGm12891NfkbTnfaIggrabUniPk', 'wgEncodeAwgTfbsSydhGm12892NfkbTnfaIggrabUniPk',
#              'wgEncodeAwgTfbsSydhGm18526NfkbTnfaIggrabUniPk', 'wgEncodeAwgTfbsSydhGm19099NfkbTnfaIggrabUniPk', 'wgEncodeAwgTfbsSydhH1hescCebpbIggrabUniPk',
#              'wgEncodeAwgTfbsSydhH1hescCtbp2UcdUniPk', 'wgEncodeAwgTfbsSydhH1hescJundIggrabUniPk', 'wgEncodeAwgTfbsSydhH1hescMaxUcdUniPk',
#              'wgEncodeAwgTfbsSydhH1hescSin3anb6001263IggrabUniPk', 'wgEncodeAwgTfbsSydhHelas3CebpbIggrabUniPk', 'wgEncodeAwgTfbsSydhHelas3CfosUniPk',
#              'wgEncodeAwgTfbsSydhHelas3CjunIggrabUniPk', 'wgEncodeAwgTfbsSydhHelas3Hae2f1UniPk', 'wgEncodeAwgTfbsSydhHelas3JundIggrabUniPk',
#              'wgEncodeAwgTfbsSydhHelas3MaxIggrabUniPk', 'wgEncodeAwgTfbsSydhHelas3P300sc584sc584IggrabUniPk', 'wgEncodeAwgTfbsSydhHelas3Stat3IggrabUniPk',
#              'wgEncodeAwgTfbsSydhHepg2CebpbForsklnUniPk', 'wgEncodeAwgTfbsSydhHepg2CebpbIggrabUniPk', 'wgEncodeAwgTfbsSydhHepg2CjunIggrabUniPk',
#              'wgEncodeAwgTfbsSydhHepg2Corestsc30189IggrabUniPk', 'wgEncodeAwgTfbsSydhHepg2JundIggrabUniPk', 'wgEncodeAwgTfbsSydhHepg2MaxIggrabUniPk',
#              'wgEncodeAwgTfbsSydhHepg2TbpIggrabUniPk', 'wgEncodeAwgTfbsSydhHuvecCfosUcdUniPk', 'wgEncodeAwgTfbsSydhHuvecCjunUniPk',
#              'wgEncodeAwgTfbsSydhHuvecGata2UcdUniPk', 'wgEncodeAwgTfbsSydhHuvecMaxUniPk', 'wgEncodeAwgTfbsSydhImr90CebpbIggrabUniPk',
#              'wgEncodeAwgTfbsSydhK562CebpbIggrabUniPk', 'wgEncodeAwgTfbsSydhK562CfosUniPk', 'wgEncodeAwgTfbsSydhK562CjunUniPk',
#              'wgEncodeAwgTfbsSydhK562Corestsc30189IggrabUniPk', 'wgEncodeAwgTfbsSydhK562Gata2UcdUniPk', 'wgEncodeAwgTfbsSydhK562JundIggrabUniPk',
#              'wgEncodeAwgTfbsSydhK562MaxIggrabUniPk', 'wgEncodeAwgTfbsSydhK562P300IggrabUniPk', 'wgEncodeAwgTfbsSydhK562TbpIggmusUniPk',
#              'wgEncodeAwgTfbsSydhK562Yy1UcdUniPk', 'wgEncodeAwgTfbsSydhMcf10aesCfosTam112hHvdUniPk', 'wgEncodeAwgTfbsSydhMcf10aesCfosTam14hHvdUniPk',
#              'wgEncodeAwgTfbsSydhMcf10aesCfosTamHvdUniPk', 'wgEncodeAwgTfbsSydhMcf10aesStat3Etoh01bUniPk', 'wgEncodeAwgTfbsSydhMcf10aesStat3Etoh01cUniPk',
#              'wgEncodeAwgTfbsSydhMcf10aesStat3Etoh01UniPk', 'wgEncodeAwgTfbsSydhMcf10aesStat3Tam112hHvdUniPk', 'wgEncodeAwgTfbsSydhMcf10aesStat3TamUniPk',
#              'wgEncodeAwgTfbsSydhMcf7Gata3UcdUniPk', 'wgEncodeAwgTfbsSydhMcf7Hae2f1UcdUniPk', 'wgEncodeAwgTfbsSydhNb4CmycUniPk', 'wgEncodeAwgTfbsSydhNb4MaxUniPk',
#              'wgEncodeAwgTfbsSydhNt2d1Yy1UcdUniPk', 'wgEncodeAwgTfbsSydhShsy5yGata2UcdUniPk', 'wgEncodeAwgTfbsSydhShsy5yGata3sc269sc269UcdUniPk',
#              'wgEncodeAwgTfbsUchicagoK562EfosUniPk', 'wgEncodeAwgTfbsUchicagoK562Egata2UniPk', 'wgEncodeAwgTfbsUchicagoK562Ehdac8UniPk',
#              'wgEncodeAwgTfbsUchicagoK562EjundUniPk', 'wgEncodeAwgTfbsUtaGm12878CmycUniPk', 'wgEncodeAwgTfbsUtaGm12878CtcfUniPk',
#              'wgEncodeAwgTfbsUtaH1hescCmycUniPk', 'wgEncodeAwgTfbsUtaH1hescCtcfUniPk', 'wgEncodeAwgTfbsUtaHelas3CtcfUniPk', 'wgEncodeAwgTfbsUtaHepg2CmycUniPk',
#              'wgEncodeAwgTfbsUtaK562CtcfUniPk', 'wgEncodeAwgTfbsUtaMcf7CtcfUniPk']


for name in data_name:
    print(name)
    accuracy, roc_auc, pr_auc = Train.run(file_name=name)
    with open('./results/result.txt', 'a') as f:
        f.write("{}: {} {} {}\n".format(name, accuracy, roc_auc, pr_auc))

# Train.run(file_name='1')
