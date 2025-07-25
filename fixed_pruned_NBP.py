#Import external libraries
import torch
import sys
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import os
import subprocess
from tqdm import tqdm
import random
import matplotlib.pylab as plt



class NBP_oc(nn.Module):
    def __init__(self, n: int, k: int, m: int, m1: int, m2: int, codeType: str, n_iterations: int,
                 folder_weights: bool = False, name: str = "default",
                 batch_size: int = 1):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.codeType = codeType
        self.n = n
        self.k = k
        #m_oc is the number rows of the overcomplete check matrix
        self.m_oc = m
        self.m1 = m1
        self.m2 = m2
        #m is the number of rows of the full rank check matrix
        self.m = n - k
        self.path = "./training_results/" + self.codeType + "_" + str(self.n) + "_" + str(self.k) + "_" + str(self.m_oc) +"_" + str(self.name) + "/"
        self.current_line = 0
        #If True, then all outgoing edges on the same CN has the same weight, configurable
        if self.name == 'NoWS':
            self.one_weight_per_cn = False
        else:
            self.one_weight_per_cn = True
        self.rate = self.k / self.n
        self.n_iterations = n_iterations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.xhat = torch.zeros((batch_size, self.n))
        self.zhat = torch.zeros((batch_size, self.n))
        self.load_matrices()

        if not folder_weights:
            if self.codeType == 'toric':
                self.ini_weight_as_one_toric(n_iterations)
            else:
                self.ini_weight_as_one(n_iterations)
        else:
            # load pretrained weights stored in directory "folder":
            self.load_weights(self.device)

    def fx(self, a: torch.tensor, b: torch.tensor) -> torch.Tensor:
        # ln(exp(x)+exp(y)) = max(x,y)+ln(1+exp(-|x-y|)
        return torch.max(a, b) + self.log1pexp(-1 * torch.abs(a - b))

    def log1pexp(self, x):
        # more stable version of log(1 + exp(x))
        m = nn.Softplus(beta=1, threshold=50)
        return m(x)

    def calculate_self_syn(self):
        self.synx = torch.matmul(self.Hz, torch.transpose(self.errorx, 0, 1))
        self.synz = torch.matmul(self.Hx, torch.transpose(self.errorz, 0, 1))
        self.synx = torch.remainder(torch.transpose(self.synx, 2, 0), 2)
        self.synz = torch.remainder(torch.transpose(self.synz, 2, 0), 2)
        return torch.cat((self.synz, self.synx), dim=1)

    def loss(self, Gamma) -> torch.Tensor:
        """loss functions proposed in [1] eq. 11"""

        # first row, anti-commute with X, second row, anti-commute with Z, [1] eq. 10
        prob = torch.sigmoid(-1.0 * Gamma).float()

        prob_aX = prob[:, 0, :]
        prob_aZ = prob[:, 1, :]

        assert not torch.isinf(prob_aX).any()
        assert not torch.isinf(prob_aZ).any()
        assert not torch.isnan(prob_aX).any()
        assert not torch.isnan(prob_aZ).any()

        #Depend on if the error commute with the entries in S_dual, which is denoted as G here
        #CSS constructions gives the simplification that Gx contains only X entries, and Gz contains on Z
        correctionx = torch.zeros_like(self.errorx)
        correctionz = torch.zeros_like(self.errorz)

        correctionz[self.qx == 1] = prob_aX[self.qx == 1]
        correctionz[self.qz == 1] = 1 - prob_aX[self.qz == 1]
        correctionz[self.qy == 1] = 1 - prob_aX[self.qy == 1]
        correctionz[self.qi == 1] = prob_aX[self.qi == 1]

        correctionx[self.qz == 1] = prob_aZ[self.qz == 1]
        correctionx[self.qx == 1] = 1 - prob_aZ[self.qx == 1]
        correctionx[self.qy == 1] = 1 - prob_aZ[self.qy == 1]
        correctionx[self.qi == 1] = prob_aZ[self.qi == 1]

        #first summ up the probability of anti-commute for all elements in each row of G
        synx = torch.matmul(self.Gz, torch.transpose(correctionx.float(), 0, 1))
        synz = torch.matmul(self.Gx, torch.transpose(correctionz.float(), 0, 1))
        synx = torch.transpose(synx, 2, 0)
        synz = torch.transpose(synz, 2, 0)
        syn_real = torch.cat((synz, synx), dim=1)

        #the take the sin function, then summed up for all rows of G
        loss = torch.zeros(1, self.batch_size)
        for b in range(self.batch_size):
            loss[0, b] = torch.sum(torch.abs(torch.sin(np.pi / 2 * syn_real[b, :, :])))

        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

        return loss

    def variable_node_update(self, incoming_messages, llr, weights_vn, weights_llr):
        # As we deal with CSS codes, all non-zero entries on the upper part anti-commute with Z and Y and commute with X
        # all non-zero entries on the upper part anti-commute with X and Y and commute with Z
        # Then the calculation can be done in matrices => speed up training (probably)
        incoming_messages_upper = incoming_messages[:, 0:self.m1, :]
        incoming_messages_lower = incoming_messages[:, self.m1:self.m_oc, :]
        incoming_messages_upper.to(self.device)
        incoming_messages_lower.to(self.device)

        Gammaz = llr * weights_llr + torch.sum(incoming_messages_upper, dim=1, keepdim=True)
        Gammax = llr * weights_llr + torch.sum(incoming_messages_lower, dim=1, keepdim=True)
        Gammay = llr * weights_llr + torch.sum(incoming_messages, dim=1, keepdim=True)

        Gammaz.double().to(self.device)
        Gammax.double().to(self.device)
        Gammay.double().to(self.device)

        #can be re-used for hard-decision in decoding, but not used in training as we don't check for decoding success
        #we are only interested in the loss during training
        Gamma = torch.cat((Gammay, Gammax, Gammaz), dim=1).to(self.device)

        assert not torch.isinf(Gammaz).any()
        assert not torch.isinf(Gammax).any()
        assert not torch.isinf(Gammay).any()


        outgoing_messages_upper = self.log1pexp(-1.0 * Gammax) - self.fx(-1.0 * Gammaz, -1.0 * Gammay)
        outgoing_messages_lower = self.log1pexp(-1.0 * Gammaz) - self.fx(-1.0 * Gammax, -1.0 * Gammay)
        Gamma_all = torch.cat((outgoing_messages_upper, outgoing_messages_lower), dim=1).to(self.device)

        outgoing_messages_upper = outgoing_messages_upper * self.Hx
        outgoing_messages_lower = outgoing_messages_lower * self.Hz
        outgoing_messages = torch.cat((outgoing_messages_upper, outgoing_messages_lower), dim=1)

        outgoing_messages = outgoing_messages - incoming_messages
        outgoing_messages = outgoing_messages * self.H

        assert not torch.isinf(Gammaz).any()
        assert not torch.isinf(Gammax).any()
        assert not torch.isinf(Gammay).any()

        #to avoid numerical issues
        outgoing_messages = torch.clip(outgoing_messages, -30.0, 30.0)

        return outgoing_messages.float() * weights_vn, Gamma, Gamma_all

    def check_node_update(self, incoming_messages: torch.Tensor, weights_cn: torch.Tensor) -> torch.Tensor:
        multipicator = torch.pow(-1, self.syn)
        multipicator = multipicator * self.H

        # use the simplification with the phi function to turn multipilication to addtion
        # a bit more troublesome than the usual SPA, because want to do it in matrix
        incoming_messages_sign = torch.sign(incoming_messages)
        incoming_messages_sign[incoming_messages == 0] = 1
        first_part = torch.prod(incoming_messages_sign, dim=2, keepdim=True)
        first_part = first_part * self.H
        first_part = first_part / incoming_messages_sign
        first_part = self.H * first_part
        assert not torch.isinf(first_part).any()
        assert not torch.isnan(first_part).any()

        incoming_messages_abs = torch.abs(incoming_messages).double()
        helper = torch.ones_like(incoming_messages_abs)
        helper[incoming_messages_abs == 0] = 0
        incoming_messages_abs[incoming_messages == 0] = 1.0

        phi_incoming_messages = -1.0 * torch.log(torch.tanh(incoming_messages_abs / 2.0))
        phi_incoming_messages = phi_incoming_messages * helper
        phi_incoming_messages = phi_incoming_messages * self.H

        temp = torch.sum(phi_incoming_messages, dim=2, keepdim=True)
        Aij = temp * self.H

        sum_msg = Aij - phi_incoming_messages
        helper = torch.ones_like(sum_msg)
        helper[sum_msg == 0] = 0
        sum_msg[sum_msg == 0] = 1.0

        second_part = -1 * torch.log(torch.tanh(sum_msg / 2.0))
        second_part = second_part * helper
        second_part = second_part * self.H
        assert not torch.isinf(second_part).any()
        assert not torch.isnan(second_part).any()

        outgoing_messages = first_part * second_part
        outgoing_messages = outgoing_messages * multipicator

        outgoing_messages = (outgoing_messages * weights_cn).float()
        return outgoing_messages

    
    def forward(self, errorx: torch.Tensor, errorz: torch.Tensor, ep: float, batch_size=1):
        if self.codeType == 'GB':
            return self._forward_gb(errorx, errorz, ep, batch_size)
        elif self.codeType == 'toric':
            return self._forward_toric(errorx, errorz, ep, batch_size)
        else:
            raise ValueError(f"Unknown codeType: {self.codeType}")

    def _forward_gb(self, errorx: torch.Tensor, errorz: torch.Tensor, ep: float, batch_size=1):
        """main decoding procedure"""
        loss_array = torch.zeros(self.batch_size, self.n_iterations).float().to(self.device)

        assert batch_size == self.batch_size

        self.errorx = errorx.to(self.device)
        self.errorz = errorz.to(self.device)

        self.qx = torch.zeros_like(self.errorx)
        self.qz = torch.zeros_like(self.errorx)
        self.qy = torch.zeros_like(self.errorx)
        self.qi = torch.ones_like(self.errorx)

        self.qx[self.errorx == 1] = 1
        self.qx[self.errorz == 1] = 0

        self.qz[self.errorz == 1] = 1
        self.qz[self.errorx == 1] = 0

        self.qy[self.errorz == 1] = 1
        self.qy[self.errorx != self.errorz] = 0

        self.qi[self.errorx == 1] = 0
        self.qi[self.errorz == 1] = 0

        self.syn = self.calculate_self_syn()

        #initial LLR to, first equation in [1,Sec.II-C]
        llr = np.log(3 * (1 - ep) / ep)

        messages_cn_to_vn = torch.zeros((batch_size, self.m_oc, self.n)).to(self.device)
        self.batch_size = batch_size

        # initlize VN message
        messages_vn_to_cn, _, _ = self.variable_node_update(messages_cn_to_vn, llr, self.weights_vn[0],
                                                            self.weights_llr[0])

        # iteratively decode, decode will continue till the max. iteration, even if the syndrome already matched
        for i in range(self.n_iterations):

            assert not torch.isnan(self.weights_llr[i]).any()
            assert not torch.isnan(self.weights_cn[i]).any()
            assert not torch.isnan(messages_cn_to_vn).any()

            # check node update:
            messages_cn_to_vn = self.check_node_update(messages_vn_to_cn, self.weights_cn[i])

            assert not torch.isnan(messages_cn_to_vn).any()
            assert not torch.isinf(messages_cn_to_vn).any()

            # variable node update:
            messages_vn_to_cn, Tau, Tau_all = self.variable_node_update(messages_cn_to_vn, llr, self.weights_vn[i + 1],
                                                                        self.weights_llr[i + 1])

            assert not torch.isnan(messages_vn_to_cn).any()
            assert not torch.isinf(messages_vn_to_cn).any()
            assert not torch.isnan(Tau).any()
            assert not torch.isinf(Tau).any()

            loss_array[:, i] = self.loss(Tau_all)


        _, minIdx = torch.min(loss_array, dim=1, keepdim=False)


        loss = torch.zeros(self.batch_size, ).float().to(self.device)
        #take average of the loss for the first iterations till the loss is minimized
        for b in range(batch_size):
            for idx in range(minIdx[b] + 1):
                loss[b] += loss_array[b, idx]
            loss[b] /= (minIdx[b] + 1)

        loss = torch.sum(loss, dim=0) / self.batch_size

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        return loss


    def _forward_toric(self, errorx: torch.Tensor, errorz: torch.Tensor, ep: float, batch_size=1):
        """main decoding procedure for toric"""
        loss_array = torch.zeros(self.batch_size, self.n_iterations).float().to(self.device)

        assert batch_size == self.batch_size

        self.errorx = errorx.to(self.device)
        self.errorz = errorz.to(self.device)

        self.qx = torch.zeros_like(self.errorx)
        self.qz = torch.zeros_like(self.errorx)
        self.qy = torch.zeros_like(self.errorx)
        self.qi = torch.ones_like(self.errorx)

        self.qx[self.errorx == 1] = 1
        self.qx[self.errorz == 1] = 0

        self.qz[self.errorz == 1] = 1
        self.qz[self.errorx == 1] = 0

        self.qy[self.errorz == 1] = 1
        self.qy[self.errorx != self.errorz] = 0

        self.qi[self.errorx == 1] = 0
        self.qi[self.errorz == 1] = 0

        self.syn = self.calculate_self_syn()

        # initial LLR to, first equation in [1,Sec.II-C]
        llr = np.log(3 * (1 - ep) / ep)

        messages_cn_to_vn = torch.zeros((batch_size, self.m_oc, self.n)).to(self.device)
        self.batch_size = batch_size

        messages_vn_to_cn, _, _ = self.variable_node_update(messages_cn_to_vn, llr, self.weights_vn[0], self.weights_llr[0])

        for i in range(self.n_iterations):
            assert not torch.isnan(self.weights_llr[i]).any()
            assert not torch.isnan(self.weights_cn[i]).any()
            assert not torch.isnan(messages_cn_to_vn).any()
            messages_cn_to_vn = self.check_node_update(messages_vn_to_cn, self.weights_cn[i])
            assert not torch.isnan(messages_cn_to_vn).any()
            assert not torch.isinf(messages_cn_to_vn).any()
            messages_vn_to_cn, Tau, Tau_all = self.variable_node_update(messages_cn_to_vn, llr, self.weights_vn[i + 1], self.weights_llr[i + 1])
            assert not torch.isnan(messages_vn_to_cn).any()
            assert not torch.isinf(messages_vn_to_cn).any()
            assert not torch.isnan(Tau).any()
            assert not torch.isinf(Tau).any()
            loss_array[:, i] = self.loss(Tau_all)

        _, minIdx = torch.min(loss_array, dim=1, keepdim=False)

        loss = torch.zeros(self.batch_size, ).float().to(self.device)
        loss_min = torch.zeros(self.batch_size, ).float().to(self.device)
        for b in range(batch_size):
            # for idx in range(minIdx[b] + 1):
            #     loss[b] += loss_array[b, idx]
            # loss[b] /= (minIdx[b] + 1)
            loss_min[b] = loss_array[b, minIdx[b]]
            for idx in range(self.n_iterations):
                loss[b] += loss_array[b, idx]
            loss[b] /= self.n_iterations

        loss = torch.sum(loss, dim=0) / self.batch_size
        loss_min = torch.sum(loss_min, dim=0) / self.batch_size

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        return loss, loss_min


    def check_syndrome(self, Tau):
        """performs hard decision to give the estimated error and check for decoding success.
        However, not used in the current script, as we are only performing trainig"""
        tmp = torch.zeros(self.batch_size, 1, self.n).to(self.device)
        Tau = torch.cat((tmp, Tau), dim=1)

        minVal, minIdx = torch.min(Tau, dim=1, keepdim=False)

        self.xhat = torch.zeros((self.batch_size, self.n)).to(self.device)
        self.zhat = torch.zeros((self.batch_size, self.n)).to(self.device)

        self.xhat[minIdx == 1] = 1
        self.xhat[minIdx == 2] = 1

        self.zhat[minIdx == 1] = 1
        self.zhat[minIdx == 3] = 1
        m = torch.nn.ReLU()

        synx = torch.matmul(self.Hz, torch.transpose(self.xhat, 0, 1))
        synz = torch.matmul(self.Hx, torch.transpose(self.zhat, 0, 1))
        synx = torch.transpose(synx, 2, 0)
        synz = torch.transpose(synz, 2, 0)
        synhat = torch.remainder(torch.cat((synz, synx), dim=1), 2)

        syn_match = torch.all(torch.all(torch.eq(self.syn, synhat), dim=1), dim=1)

        correctionx = torch.remainder(self.xhat + self.errorx, 2)
        correctionz = torch.remainder(self.zhat + self.errorz, 2)
        synx = torch.matmul(self.Gz, torch.transpose(correctionx, 0, 1))
        synz = torch.matmul(self.Gx, torch.transpose(correctionz, 0, 1))
        synx = torch.transpose(synx, 2, 0)
        synz = torch.transpose(synz, 2, 0)
        self.syn_real = torch.cat((synz, synx), dim=1)

        syn_real = torch.remainder(self.syn_real, 2)
        tmmp = torch.sum(syn_real, dim=1, keepdim=False)
        success = torch.all(torch.eq(torch.sum(syn_real, dim=1, keepdim=False), 0), dim=1)
        return syn_match, success



    def unsqueeze_batches(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Checks if tensor is 2D or 3D. If tensor is 2D, insert extra dimension (first dimension)
        This method can be used to allow decoding of
            batches of codewords (batch size, m, n)
            as well as single codewords (m, n)
        """
        if tensor.dim() == 3:
            return tensor
        elif tensor.dim() == 2:
            return torch.unsqueeze(tensor, dim=0)

    #continue with the NBP_oc class, some tool functions
    def load_matrices(self):
        """reads in the check matrix for decoding as well as the dual matrix for checking decoding success"""
        file_nameGx = "./PCMs/" + self.codeType + "_" + str(self.n) + "_" + str(
            self.k) + "/" + self.codeType + "_" + str(self.n) + "_" + str(self.k) + "_Gx.alist"
        file_nameGz = "./PCMs/" + self.codeType + "_" + str(self.n) + "_" + str(
            self.k) + "/" + self.codeType + "_" + str(self.n) + "_" + str(self.k) + "_Gz.alist"
        Gx = readAlist(file_nameGx)
        Gz = readAlist(file_nameGz)

        file_nameH = "./PCMs/" + self.codeType + "_" + str(self.n) + "_" + str(
            self.k) + "/" + self.codeType + "_" + str(self.n) + "_" + str(self.k) + "_H_" + str(self.m_oc) + ".alist"

        H = readAlist(file_nameH)
        self.H = H
        Hx = H[0:self.m1, :]
        Hz = H[self.m1:self.m_oc, :]
        Gx = torch.from_numpy(Gx).float()
        Gz = torch.from_numpy(Gz).float()
        Hx = torch.from_numpy(Hx).float()
        Hz = torch.from_numpy(Hz).float()


        # first dim for batches.
        self.Hx = self.unsqueeze_batches(Hx).float().to(self.device)
        self.Hz = self.unsqueeze_batches(Hz).float().to(self.device)
        self.Gx = self.unsqueeze_batches(Gx).float().to(self.device)
        self.Gz = self.unsqueeze_batches(Gz).float().to(self.device)

        self.H = torch.cat((self.Hx, self.Hz), dim=1).float().to(self.device)
        self.H_reverse = 1 - self.H


    def ini_weight_as_one_toric(self, n_iterations: int):
        """Initialize weights as trainable nn.Parameter, including VN weights."""

        import torch.nn as nn

        self.weights_llr = nn.ParameterList()
        self.weights_cn = nn.ParameterList()
        self.weights_vn = nn.ParameterList()

        if self.m_oc < self.n:
            for i in range(n_iterations):
                # CN weights
                if self.one_weight_per_cn:
                    cn_param = nn.Parameter(torch.ones((1, self.m_oc, 1), device=self.device))
                else:
                    cn_param = nn.Parameter(torch.ones((1, self.m_oc, self.n), device=self.device))
                self.weights_cn.append(cn_param)

                # LLR weights
                llr_param = nn.Parameter(torch.ones((1, 1, self.n), device=self.device))
                self.weights_llr.append(llr_param)

                # VN weights
                vn_param = nn.Parameter(torch.ones((1, self.m_oc, self.n), device=self.device))
                self.weights_vn.append(vn_param)

            self.weights_llr.append(nn.Parameter(torch.ones((1, 1, self.n), device=self.device)))
            self.weights_vn.append(nn.Parameter(torch.ones((1, self.m_oc, self.n), device=self.device)))

        else:
            self.ini_weights = np.array([1.0, 0.1])
            temp = torch.from_numpy(self.ini_weights[0] * np.ones((1, 1, self.n))).float().to(self.device)
            l1 = (self.n) // 2
            l2 = self.m_oc // 2 - l1

            if self.one_weight_per_cn:
                temp_vn = np.concatenate((np.ones((1, l1, 1)), self.ini_weights[1] * np.ones((1, l2, 1))), axis=1)
            else:
                temp_vn = np.concatenate(
                    (np.ones((1, l1, self.n)),
                     self.ini_weights[1] * np.ones((1, l2, self.n))), axis=1)
            temp_vn = np.concatenate((temp_vn, temp_vn), axis=1)
            temp_vn_tensor = torch.from_numpy(temp_vn).float().to(self.device)

            for i in range(n_iterations):
                self.weights_llr.append(nn.Parameter(temp.clone()))
                self.weights_cn.append(nn.Parameter(temp_vn_tensor.clone()))
                self.weights_vn.append(nn.Parameter(torch.ones(1, self.m_oc, self.n, device=self.device)))

            self.weights_llr.append(nn.Parameter(temp.clone()))
            self.weights_vn.append(nn.Parameter(torch.ones(1, self.m_oc, self.n, device=self.device)))

        self.save_weights()

        #TODO, make the same for toric
    def ini_weight_as_one(self, n_iterations: int):
        """Initialize weights as learnable parameters, compatible with PyTorch pruning and optimizers."""
        import torch.nn as nn

        self.weights_llr = nn.ParameterList()
        self.weights_cn = nn.ParameterList()
        self.weights_vn = []  # If not trainable, leave as tensors

        for i in range(n_iterations):
            if self.one_weight_per_cn:
                cn_param = nn.Parameter(torch.ones((1, self.m_oc, 1), device=self.device))
            else:
                cn_param = nn.Parameter(torch.ones((1, self.m_oc, self.n), device=self.device))
            self.weights_cn.append(cn_param)

            llr_param = nn.Parameter(torch.ones((1, 1, self.n), device=self.device))
            self.weights_llr.append(llr_param)

            self.weights_vn.append(torch.ones(1, self.m_oc, self.n, device=self.device))  # non-param

        # One extra for llr and vn, as in your original code
        self.weights_llr.append(nn.Parameter(torch.ones((1, 1, self.n), device=self.device)))
        self.weights_vn.append(torch.ones(1, self.m_oc, self.n, device=self.device))


    def load_weights(self, device: str):
        """
        Load pretrained weights. Parameters directory: str directory where pretrained weights are stored as "weights_vn.pt", "weights_cn.pt" or "weights_llr.pt".
        device : str 'cpu' or 'cuda'
        """
        print('continue training with previous weights')

        if device == 'cpu':
            # Safe loading for ParameterList in PyTorch >=2.6:
            with torch.serialization.safe_globals([torch.nn.modules.container.ParameterList]):
                weights_vn = torch.load(self.path + 'weights_vn.pt', map_location=torch.device('cpu'))
                weights_cn = torch.load(self.path + 'weights_cn.pt', map_location=torch.device('cpu'))
                weights_llr = torch.load(self.path + 'weights_llr.pt', map_location=torch.device('cpu'))
        else:
            with torch.serialization.safe_globals([torch.nn.modules.container.ParameterList]):
                weights_cn = torch.load(self.path + 'weights_cn.pt')
                weights_vn = torch.load(self.path + 'weights_vn.pt')
                weights_llr = torch.load(self.path + 'weights_llr.pt')

        self.weights_llr = weights_llr
        self.weights_cn = weights_cn
        self.weights_vn = weights_vn

    def save_weights(self):
        """weights are saved twice, once as .pt for python, once as .txt for c++"""
        os.makedirs(self.path, exist_ok=True)
        #some parameters may not be trained, but we save them anyway
        file_vn = "weights_vn.pt"
        file_cn = "weights_cn.pt"
        file_llr = "weights_llr.pt"

        torch.save(self.weights_vn, os.path.join(self.path, file_vn))
        torch.save(self.weights_cn, os.path.join(self.path, file_cn))
        torch.save(self.weights_llr, os.path.join(self.path, file_llr))
        print(f'  weights saved to {file_cn},{file_vn}, and {file_llr}.\n')

        # the following codes save the weights into txt files, which is used for C++ code for evaluating the trained
        # decoder. So the C++ codes don't need to mess around with python packages
        # not very elegant but will do for now
        if sys.version_info[0] == 2:
            import cStringIO
            StringIO = cStringIO.StringIO
        else:
            import io

        StringIO = io.StringIO

        # write llr weights, easy
        f = open(self.path + "weight_llr.txt", "w")
        with StringIO() as output:
            output.write('{}\n'.format(len(self.weights_llr)))
            for i in self.weights_llr:
                data = i.detach().cpu().numpy().reshape(self.n, 1)
                opt = ["%.16f" % i for i in data]
                output.write(' '.join(opt))
                output.write('\n')
            f.write(output.getvalue())
        f.close()

        # write CN weights
        H_tmp = self.H.detach().cpu().numpy().reshape(self.m_oc, self.n)
        H_tmp = np.array(H_tmp, dtype='int')
        f = open(self.path + "weight_cn.txt", "w")
        with StringIO() as output:
            output.write('{}\n'.format(len(self.weights_cn)))
            nRows, nCols = H_tmp.shape
            # first line: matrix dimensions
            output.write('{} {}\n'.format(nCols, nRows))

            # next three lines: (max) column and row degrees
            colWeights = H_tmp.sum(axis=0)
            rowWeights = H_tmp.sum(axis=1)

            maxRowWeight = max(rowWeights)

            if self.one_weight_per_cn:
                # column-wise nonzeros block
                for i in self.weights_cn:
                    matrix = i.detach().cpu().numpy().reshape(self.m_oc, 1)
                    for rowId in range(nRows):
                        opt = ["%.16f" % i for i in matrix[rowId]]
                        for i in range(rowWeights[rowId].astype('int') - 1):
                            output.write(opt[0])
                            output.write(' ')
                        output.write(opt[0])
                        # fill with zeros so that every line has maxDegree number of entries
                        output.write(' 0' * (maxRowWeight - rowWeights[rowId] - 1).astype('int'))
                        output.write('\n')
            else:
                # column-wise nonzeros block
                for i in self.weights_cn:
                    matrix = i.detach().cpu().numpy().reshape(self.m_oc, self.n)
                    matrix *= self.H[0].detach().cpu().numpy().reshape(self.m_oc, self.n)
                    for rowId in range(nRows):
                        nonzeroIndices = np.flatnonzero(matrix[rowId, :])  # AList uses 1-based indexing
                        output.write(' '.join(map(str, matrix[rowId, nonzeroIndices])))
                        # fill with zeros so that every line has maxDegree number of entries
                        output.write(' 0' * (maxRowWeight - len(nonzeroIndices)))
                        output.write('\n')
            f.write(output.getvalue())
        f.close()

        # write VN weights
        H_tmp = self.H.detach().cpu().numpy().reshape(self.m_oc, self.n)
        H_tmp = np.array(H_tmp, dtype='int')
        f = open(self.path + "weight_vn.txt", "w")
        with StringIO() as output:
            output.write('{}\n'.format(len(self.weights_vn)))
            nRows, nCols = H_tmp.shape
            # first line: matrix dimensions
            output.write('{} {}\n'.format(nCols, nRows))

            # next three lines: (max) column and row degrees
            colWeights = H_tmp.sum(axis=0)
            rowWeights = H_tmp.sum(axis=1)

            maxColWeight = max(colWeights)

            # column-wise nonzeros block
            for i in self.weights_vn:
                matrix = i.detach().cpu().numpy().reshape(self.m_oc, self.n)
                matrix *= self.H[0].detach().cpu().numpy().reshape(self.m_oc, self.n)
                for colId in range(nCols):
                    nonzeroIndices = np.flatnonzero(matrix[:, colId])  # AList uses 1-based indexing
                    output.write(' '.join(map(str, matrix[nonzeroIndices, colId])))
                    # fill with zeros so that every line has maxDegree number of entries
                    output.write(' 0' * (maxColWeight - len(nonzeroIndices)))
                    output.write('\n')
            f.write(output.getvalue())
        f.close()


    def prune_weights(self, amount=0.1):
        """
        Prunes globally across all weights in self.weights_cn ParameterList.
        """
        parameters_to_prune = [(self.weights_cn, str(i)) for i in range(len(self.weights_cn))]
        # print("Before pruning:", self.weights_cn[0].data)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        # print("After pruning:", self.weights_cn[0].data)

        print(type(self.weights_cn))
        print([type(p) for p in self.weights_cn])

        self.save_weights()

#helper functions
def readAlist(directory):
    '''
    Reads in a parity check matrix (pcm) in A-list format from text file. returns the pcm in form of a numpy array with 0/1 bits as float64.
    '''

    alist_raw = []
    with open(directory, "r") as f:
        lines = f.readlines()
        for line in lines:
            # remove trailing newline \n and split at spaces:
            line = line.rstrip().split(" ")
            # map string to int:
            line = list(map(int, line))
            alist_raw.append(line)
    alist_numpy = alistToNumpy(alist_raw)
    alist_numpy = alist_numpy.astype(float)
    return alist_numpy


def alistToNumpy(lines):
    '''Converts a parity-check matrix in AList format to a 0/1 numpy array'''
    nCols, nRows = lines[0]
    if len(lines[2]) == nCols and len(lines[3]) == nRows:
        startIndex = 4
    else:
        startIndex = 2
    matrix = np.zeros((nRows, nCols), dtype=float)
    for col, nonzeros in enumerate(lines[startIndex:startIndex + nCols]):
        for rowIndex in nonzeros:
            if rowIndex != 0:
                matrix[rowIndex - 1, col] = 1
    return matrix


def optimization_step(decoder: NBP_oc, ep0, optimizer: torch.optim.Optimizer, errorx, errorz):
   #call the forward function
   loss = decoder(errorx, errorz, ep0, decoder.batch_size)

   # delete old gradients.
   optimizer.zero_grad()
   # calculate gradient
   loss.backward(retain_graph=True)
   # update weights
   optimizer.step()

   return loss.detach()


def optimization_toric(decoder: NBP_oc, ep0, optimizer: torch.optim.Optimizer, errorx, errorz, scheduler=None):
    # call the forward function
    loss, loss_min = decoder(errorx, errorz, ep0, decoder.batch_size)

    # delete old gradients.
    optimizer.zero_grad()
    # calculate gradient
    loss_min.backward(retain_graph=True)
    clip_value = 0.001
    for p in range(decoder.n_iterations):
        # Only clamp if grad exists
        if decoder.weights_vn[p].grad is not None:
            decoder.weights_vn[p].grad.data.clamp_(-clip_value, clip_value)
        if decoder.weights_cn[p].grad is not None:
            decoder.weights_cn[p].grad.data.clamp_(-clip_value, clip_value)
        if decoder.weights_llr[p].grad is not None:
            decoder.weights_llr[p].grad.data.clamp_(-clip_value, clip_value)
    #

    # update weights
    optimizer.step()
    scheduler.step()

    return loss.detach(), loss_min.detach()


def training_loop(decoder: NBP_oc, optimizer: torch.optim.Optimizer, r1, r2, ep0, num_batch, path):
    print(f'training on random errors, weight from {r1} to {r2} ')
    loss_length = num_batch
    loss = torch.zeros(loss_length)

    # --------------------------------------------------------------

    idx = 0
    with tqdm(total=loss_length) as pbar:
        for i_batch in range(num_batch):
            errorx = torch.tensor([])
            errorz = torch.tensor([])
            for w in range(r1, r2):
                batch_subsize = decoder.batch_size // (r2 - r1 + 1)
                ex, ez = addErrorGivenWeight(decoder.n, w, batch_subsize)
                errorx = torch.cat((errorx, ex), dim=0)
                errorz = torch.cat((errorz, ez), dim=0)
            res_size = decoder.batch_size - ((decoder.batch_size // (r2 - r1 + 1)) * (r2 - r1))
            ex, ez = addErrorGivenWeight(decoder.n, r2, res_size)
            errorx = torch.cat((errorx, ex), dim=0)
            errorz = torch.cat((errorz, ez), dim=0)

            if(decoder.name == "Tick"):
                print("Training on others mistakes...")
                patterns = load_tokenized_error_lines('forTick-1.txt')
                errorx, errorz, decoder.current_line = addErrorfromEnsemble(decoder.n, patterns, decoder.batch_size, decoder.current_line)


            if(decoder.name == "Trick"):
                print("Training on others mistakes...")
                patterns = load_tokenized_error_lines('forTrick-1.txt')
                errorx, errorz, decoder.current_line = addErrorfromEnsemble(decoder.n, patterns, decoder.batch_size, decoder.current_line)

            if(decoder.name == "Track"):
                print("Training on others mistakes...")
                patterns = load_tokenized_error_lines('forTrack-1.txt')
                errorx, errorz, decoder.current_line = addErrorfromEnsemble(decoder.n, patterns, decoder.batch_size, decoder.current_line)

            loss[idx]= optimization_step(decoder, ep0, optimizer, errorx, errorz)
            pbar.update(1)
            pbar.set_description(f"loss {loss[idx]}")
            idx += 1
        decoder.save_weights()

    print('Training completed.\n')
    return loss

def addDeploarizationErrorGiveEp(n: int, ep:float, batch_size:int = 1):
    errorx = torch.zeros((batch_size, n))
    errorz = torch.ones((batch_size, n))
    np.random.seed()
    for b in range(batch_size):
        a = torch.from_numpy(np.random.rand(n,))
        # iny = (a <= ep / 3).reshape(n,)
        errorx[b, (a <= 2.0 * ep / 3.0)] = 1
        errorz[b, (a < ep / 3.0)] = 0
        errorz[b, (a > ep)] = 0
    return errorx, errorz

def training_toric(decoder: NBP_oc, optimizer, ep1, sep,num_points, ep0, num_batch, path, scheduler=None):
    print(f'training on random errors, epsilon from {ep1} to {ep1+sep*(num_points-1)} ')
    loss_length = num_batch
    loss = torch.zeros(loss_length)
    loss_min = torch.zeros(loss_length)

    idx = 0
    with tqdm(total=loss_length) as pbar:
        for i_batch in range(num_batch):
            errorx = torch.tensor([])
            errorz = torch.tensor([])
            for i in range(num_points):
                ex, ez = addDeploarizationErrorGiveEp(decoder.n, ep1+i*sep, decoder.batch_size//num_points)
                errorx = torch.cat((errorx, ex), dim=0)
                errorz = torch.cat((errorz, ez), dim=0)


            loss[idx], loss_min[idx] = optimization_toric(decoder, ep0, optimizer, errorx, errorz, scheduler)
            pbar.update(1)
            lrs = scheduler.get_last_lr()
            lr_str = ", ".join(f"{lr:.2f}" for lr in lrs)
            pbar.set_description(f"loss {loss[idx]:.2f}, loss min {loss_min[idx]:.2f}, lr(s): {lr_str}")
            idx += 1


            if ((i_batch+1)%100==0):
                decoder.save_weights()
                plot_loss(loss_min[0:idx-1], path=None)

    decoder.save_weights()
    print('Training completed.\n')
    return loss_min

def train(NBP_dec:NBP_oc):

    if(NBP_dec.codeType == 'GB'):
        lr = 0.001
        r1 = 2
        r2 = 3
        ep0 = 0.1
        # number of updates
        n_batches = 500
    elif(NBP_dec.codeType == 'toric'):
        lr = 1
        torch.autograd.set_detect_anomaly(True)
        m = 3*NBP_dec.n  # number of checks, can also use 46 or 44
        ep1=0.03
        sep=0.01
        num_points = 6
        if m==3*NBP_dec.n:
            ep0 = 0.49
            ep1=0.06

        # number of updates
        n_batches = 200


    #trainable parameters
    if(NBP_dec.codeType == 'GB'):
        optimizer = torch.optim.Adam(NBP_dec.parameters(), lr=lr)
    if(NBP_dec.codeType == 'toric'):
        optimizer = torch.optim.SGD(
            NBP_dec.parameters(),
            lr=lr
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0, end_factor=0.1, total_iters=1200)
    print('--- Training Metadata ---')
    print(f'Code: n={NBP_dec.n}, k={NBP_dec.k}, PCM rows={NBP_dec.m1},{NBP_dec.m2}')
    print(f'device: {NBP_dec.device}')
    print(f'training ep0 = {ep0}')
    print(f'Decoder: {NBP_dec.name}')
    print(f'decoding iterations = {NBP_dec.n_iterations}')
    print(f'number of batches = {n_batches}')
    print(f'error patterns per batch = {NBP_dec.batch_size}')
    print(f'learning rate = {lr}\n')

    if (NBP_dec.codeType == 'GB'): 
        #pre-training stage, basically only the parameters for the first iteration is trained
        loss_pre_train = training_loop(NBP_dec, optimizer, r1, r2, ep0, n_batches, NBP_dec.path)
        plot_loss(loss_pre_train, NBP_dec.path)


        #continue to train with higher weight errors, mostly for the later iterations
        r1 = 3
        r2 = 9

        n_batches = 200
        loss = training_loop(NBP_dec, optimizer, r1, r2, ep0, n_batches, NBP_dec.path)

        plot_loss(torch.cat((loss_pre_train, loss) , dim=0), NBP_dec.path)



    elif (NBP_dec.codeType == 'toric'):
        # training stage
        loss = torch.Tensor()
        print("Plotting loss...")
        loss_pre_train = training_toric(NBP_dec, optimizer, ep1, sep,num_points, ep0, n_batches, NBP_dec.path, scheduler=scheduler)
        loss = torch.cat((loss, loss_pre_train), dim=0)
        plot_loss(loss, NBP_dec.path) #its ok if it doesn't converge to 0
        plot_loss(loss_pre_train, NBP_dec.path)


def init_and_train(n:int, k:int, m:int, n_iterations:int, codeType:str, use_pretrained_weights:bool = False, name: str = "default"):
    # give parameters for the code and decoder
    m1 = m // 2
    m2 = m // 2

    if(codeType == 'GB'):
        #number of error patterns in each mini batch
        batch_size = 100
    elif(codeType == 'toric'):
        #number of error patterns in each mini batch
        num_points = 6
        batch_size = 20*num_points


    decoder = NBP_oc(n, k, m, m1,m2, codeType, n_iterations, use_pretrained_weights, name, batch_size)

    train(decoder)

    return decoder

def plot_loss(loss, path, myrange = 0):
    f = plt.figure(figsize=(8, 5))
    if myrange>0:
        plt.plot(range(1, myrange + 1), loss[0:myrange],marker='.')
    else:
        plt.plot(range(1, loss.size(dim=0)+1),loss,marker='.')
    plt.show()
    file_name = str(path) + "loss.pdf"
    f.savefig(file_name)
    plt.close()

def addErrorGivenWeight(n:int, w:int, batch_size:int = 1):
    errorx = torch.zeros((batch_size, n))
    errorz = torch.zeros((batch_size, n))
    li = list(range(0,n))
    for b in range(batch_size):
        pos = random.sample(li, w)
        al = torch.rand([w,])
        for p,a in zip(pos,al):
            if a<1/3:
                errorx[b,p] = 1
            elif a<2/3:
                errorz[b,p] = 1
            else:
                errorx[b,p] = 1
                errorz[b,p] = 1
    return errorx, errorz


def addErrorfromEnsemble(n: int, lines_tokenized: list, batch_size: int = 1, start_line: int = 0):
    """
    lines_tokenized: list of lists, each inner list contains tokens for a line (e.g. ['X5', 'Y7', 'Z11'])
    Returns (errorx, errorz, next_line).
    """
    errorx = torch.zeros((batch_size, n))
    errorz = torch.zeros((batch_size, n))
    end_line = min(start_line + batch_size, len(lines_tokenized))
    for b, idx in enumerate(range(start_line, end_line)):
        tokens = lines_tokenized[idx]
        for token in tokens:
            if token and token[0] in 'XYZxyz':
                try:
                    pos = int(token[1:])
                except Exception:
                    continue
                if 0 <= pos < n:
                    if token[0] in 'Xx':
                        errorx[b, pos] = 1
                    if token[0] in 'Zz':
                        errorz[b, pos] = 1
                    if token[0] in 'Yy':
                        errorx[b, pos] = 1
                        errorz[b, pos] = 1
    return errorx, errorz, end_line

def load_tokenized_error_lines(txt_path: str):
    """
    Loads lines from a txt file, removes any trailing float, splits into tokens.
    Returns a list of lists of tokens.
    """
    lines_tokenized = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Remove trailing float if present
            try:
                float(parts[-1])
                if len(parts) > 1:
                    parts = parts[:-1]
            except Exception:
                pass
            lines_tokenized.append(parts)
    return lines_tokenized


# give parameters for the code and decoder
trials = [1, 2, 3, 4, 5, 6, 7]
percentage = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.128, 0.256, 0.512]
# percentage = [0.02, 0.04, 0.08, 0.16, 0.32]

# for num in trials:
#     for percent in percentage:
#         specifier = f"{num}_{percent}"
#         print(specifier)
#         NBP_decoder = init_and_train(128, 2, 384, 18, 'toric', name = specifier)
#         for value in range(1, num+1):
#             print("Here we go again...")
#             print(num)
#             NBP_decoder.prune_weights(percent)
#             train(NBP_decoder)

Tick = init_and_train(48, 6, 2000, 6, 'GB', use_pretrained_weights=True, name="Tick")

Trick = init_and_train(48, 6, 2000, 6, 'GB', use_pretrained_weights=True, name="Trick")

Track = init_and_train(48, 6, 2000, 6, 'GB', use_pretrained_weights=True, name="Track")

print("Training and pruning completed.\n")

