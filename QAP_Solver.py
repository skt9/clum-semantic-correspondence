from typing import List
from mpopt.qap import solver
from mpopt.qap.model import Assignment, Edge
import torch
from torch.autograd import grad
import torch.nn as nn
from torch.nn import functional as F
import numpy as np 
from scipy import ndimage
from torch.nn.init import xavier_normal_
import time, os

from mpopt.qap.model import Assignment,Edge
from mpopt import qap,utils
from scipy.optimize import linear_sum_assignment
from QAP_fns import solve_qap, get_results_mask

######## first test with batch_size == 1 ############

epsilon_val=1e-6

class LAPSolver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, unaries: torch.Tensor, params:dict):
        device = unaries.device
        labelling = torch.zeros_like(unaries)
        unaries_np = unaries.cpu().detach().numpy() #   
        row_ind, col_ind = linear_sum_assignment(unaries_np)
        labelling[row_ind, col_ind] = 1.
        ctx.labels = labelling
        ctx.col_labels = col_ind
        ctx.params = params
        ctx.unaries = unaries
        ctx.device = device
        return labelling

    @staticmethod
    def backward(ctx, unary_gradients: torch.Tensor):
        """
        :param ctx: context for backpropagation. Gradient is stored here.
        :param unaries: List
        :param pwCost: dictionary
        """
        assert(ctx.unaries.shape==unary_gradients.shape)
        device=unary_gradients.device
        num_unaries,num_hypotheses=unary_gradients.shape

        #   compute lambda
        lambda_val = ctx.params["lambda"]
        unaries = ctx.unaries
        
        unaries_prime=unaries+lambda_val*unary_gradients

        unaries_prime_np=unaries_prime.detach().cpu().numpy()
        
        # print(unary_gradients)

        # print(f"unaries")
        # print(unaries.detach().cpu().numpy())
        
        # print(f"unaries_prime")
        # print(unaries_prime_np)

        # print(f"ctx.labels\n")
        # print(ctx.labels)

        bwd_labels = torch.zeros_like(unaries)
        row_ind, col_ind = linear_sum_assignment(unaries_prime_np)
        bwd_labels[row_ind, col_ind] = 1.

        # print(f"bwd_labels")
        # print(bwd_labels)

        forward_labels = ctx.labels
        unary_grad_bwd =-(forward_labels-bwd_labels) / (lambda_val + epsilon_val)

        # print(f"unary_gradients_backward")
        # print(unary_grad_bwd)
        # print(f"fwd labels: {ctx.col_labels} bwd prime_labels: {col_ind}")
        # print(f"unary_gradients: {unary_gradients}")

        # print(f"un_grads: {torch.norm(unary_gradients)} un_grad_bwd: {torch.norm(unary_grad_bwd)}")

        return unary_grad_bwd, None

class QAPSolverUnaries(torch.autograd.Function):

    @staticmethod
    def forward(ctx, unaries: torch.Tensor, pw_costs: torch.Tensor,  edges:List[tuple], params:dict):
        """
        :param ctx: context for backpropagation. Gradient is stored here.
        :param unaries: List
        :param pwCost: dictionary
        """
        device = unaries.device
        num_unaries, num_hypotheses = unaries.shape
        hypotheses = torch.Tensor([ [ j for j in range(num_hypotheses)] for i in range(num_unaries)])
        labeling = solve_qap(unaries, hypotheses, pw_costs, edges, params)
        unary_costs_paid, pw_costs_paid, un_labels, pw_labels = get_results_mask(labeling, hypotheses, pw_costs, edges)

        ctx.unaries = unaries
        ctx.pw_costs = pw_costs
        ctx.labeling = labeling
        ctx.hypotheses = hypotheses
        ctx.unary_costs_paid = unary_costs_paid
        ctx.pw_costs_paid = pw_costs_paid
        ctx.edges = edges
        ctx.params = params

        return torch.from_numpy(unary_costs_paid).to(device)
        
    @staticmethod
    def backward(ctx,unary_gradients:torch.Tensor):
        """
        :param ctx: context for backpropagation. Gradient is stored here.
        :param unaries: List
        :param pwCost: dictionary
        """
        assert(ctx.unaries.shape==unary_gradients.shape)
        device=unary_gradients.device
        num_unaries,num_hypotheses=unary_gradients.shape
        #   compute lambda
        lambda_val=ctx.params["lambda"]
        unaries_prime=ctx.unaries+lambda_val*unary_gradients
        pw_costs_prime=ctx.pw_costs
        #   solve QAP
        hypotheses = ctx.hypotheses
        labeling_bwd = solve_qap(unaries_prime,hypotheses,pw_costs_prime,ctx.edges,ctx.params)
        contains_minus_one = torch.any(labeling_bwd == -1)
        if contains_minus_one is True:
            return None, None
        unary_prime_costs_paid, _, _, _ = get_results_mask(labeling_bwd,hypotheses,pw_costs_prime,ctx.edges) 
        #   computing unary and paiewise gradients
        unary_grad_bwd =-(ctx.unary_costs_paid-unary_prime_costs_paid) / (lambda_val + epsilon_val)
        # pwCosts_grad_bwd=-(ctx.pw_costs_paid-pwCosts_prime_paid)/(lambda_val+epsilon_val)
        return torch.from_numpy(unary_grad_bwd).to(device), None, None, None

class QAPSolverWithPairwiseModule(torch.autograd.Function):

    @staticmethod
    def forward(ctx, unaries:torch.Tensor, pw_costs:torch.Tensor, edges:List[tuple], params:dict):
        """
        :param ctx: context for backpropagation. Gradient is stored here.
        :param unaries: List
        :param pwCost: dictionary
        """
        device = unaries.device
        num_unaries,num_hypotheses=unaries.shape
        hypotheses = torch.Tensor([ [ j for j in range(num_hypotheses)] for i in range(num_unaries)])
        labeling = solve_qap(unaries, hypotheses, pw_costs, edges, params)
        unary_costs_paid, pw_costs_paid, unary_labels, pw_labels\
              = get_results_mask(labeling,hypotheses,pw_costs,edges)

        ctx.unaries=unaries
        ctx.pw_costs=pw_costs
        ctx.labeling=labeling
        ctx.unary_labels = unary_labels
        ctx.pw_labels = pw_labels
        ctx.hypotheses=hypotheses
        ctx.unary_costs_paid=unary_costs_paid
        ctx.pw_costs_paid=pw_costs_paid
        ctx.edges=edges
        ctx.params=params

        return torch.from_numpy(unary_costs_paid).to(device), torch.from_numpy(pw_costs_paid).to(device)
        

    @staticmethod
    def backward(ctx,unary_gradients: torch.Tensor, pw_gradients: torch.Tensor):
        """
        :param ctx: context for backpropagation. Gradient is stored here.
        :param unaries: List
        :param pwCost: dictionary
        """
        assert(ctx.unaries.shape==unary_gradients.shape)
        device=unary_gradients.device

        #   compute lambda
        lambda_val = ctx.params["lambda"]
        unaries_prime = ctx.unaries + lambda_val * unary_gradients
        pw_costs_prime = ctx.pw_costs + lambda_val * pw_gradients
        
        #   solve QAP
        hypotheses = ctx.hypotheses
        labeling_bwd = solve_qap(unaries_prime, hypotheses, pw_costs_prime, ctx.edges, ctx.params)
        unary_prime_costs_paid, pw_costs_prime_paid, unary_labels, pw_labels\
              = get_results_mask(labeling_bwd, hypotheses, pw_costs_prime, ctx.edges)

        #   computing unary and paiewise gradients
        unary_grad_bwd =(unary_prime_costs_paid-ctx.unary_costs_paid) / (lambda_val + epsilon_val)
        pw_costs_grad_bwd=(pw_costs_prime_paid-ctx.pw_costs_paid)/(lambda_val+epsilon_val)
        unary_grad_bwd = torch.from_numpy(unary_grad_bwd).to(float).to(device)
        pw_costs_grad_bwd = torch.from_numpy(pw_costs_grad_bwd).to(float).to(device)
        return unary_grad_bwd, pw_costs_grad_bwd, None, None

class QAPSolverModule(torch.nn.Module):

    def __init__(self,params_dict):
        super(QAPSolverModule, self).__init__()
        self.params_dict=params_dict
        self.params_dict["const"]=1000   #   constant to subtract from the qap solver 
        # self.qapSolver = QAPSolverWithPairwiseModule()
        self.qap_solver = QAPSolverUnaries()

    def forward(self, unaries: torch.Tensor, pw_costs: torch.Tensor, edges:List[tuple]):
        """
            Input:
                unaries, torch.Tensor(num_unaries, num_hypotheses)
        """
        
        predicted_labeling_matrix = self.qap_solver.apply(unaries-self.params_dict["const"],pw_costs,edges,self.params_dict)
        return predicted_labeling_matrix

class LAPSolverModule(torch.nn.Module):

    def __init__(self,params_dict):
        super(LAPSolverModule, self).__init__()
        self.params_dict=params_dict
        self.params_dict["const"]=100   #   constant to subtract from the qap solver 
        self.lapSolver = LAPSolver()

    def forward(self,unaries:torch.Tensor):
        predicted_labeling_matrix=self.lapSolver.apply(unaries,self.params_dict)
        return predicted_labeling_matrix

class QAPSolverImperfect(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, unaries:torch.Tensor, pw_costs:torch.Tensor, hypotheses: torch.Tensor, \
                edges:List[tuple], params:dict):
        """
        :param ctx: context for backpropagation. Gradient is stored here.
        :param unaries: List
        :param pwCost: dictionary
        """
        device = unaries.device
        labeling = solve_qap(unaries, hypotheses, pw_costs, edges, params)
        unary_costs_paid, pw_costs_paid, unary_labels, pw_labels\
              = get_results_mask(labeling,hypotheses,pw_costs,edges)

        ctx.unaries=unaries
        ctx.pw_costs=pw_costs
        ctx.labeling=labeling
        ctx.unary_labels = unary_labels
        ctx.pw_labels = pw_labels
        ctx.hypotheses=hypotheses
        ctx.unary_costs_paid=unary_costs_paid
        ctx.pw_costs_paid=pw_costs_paid
        ctx.edges=edges
        ctx.params=params
        return torch.from_numpy(unary_costs_paid).to(device), torch.from_numpy(pw_costs_paid).to(device)
        

    @staticmethod
    def backward(ctx,unary_gradients: torch.Tensor, pw_gradients: torch.Tensor):
        """
        :param ctx: context for backpropagation. Gradient is stored here.
        :param unaries: List
        :param pwCost: dictionary
        """
        assert(ctx.unaries.shape==unary_gradients.shape)
        device=unary_gradients.device

        #   compute lambda
        lambda_val = ctx.params["lambda"]
        unaries_prime = ctx.unaries + lambda_val * unary_gradients
        pw_costs_prime = ctx.pw_costs + lambda_val * pw_gradients
        
        #   solve QAP
        hypotheses = ctx.hypotheses
        labeling_bwd = solve_qap(unaries_prime, hypotheses, pw_costs_prime, ctx.edges, ctx.params)
        unary_prime_costs_paid, pw_costs_prime_paid, unary_labels, pw_labels\
              = get_results_mask(labeling_bwd, hypotheses, pw_costs_prime, ctx.edges)

        #   computing unary and paiewise gradients
        unary_grad_bwd =(unary_prime_costs_paid-ctx.unary_costs_paid) / (lambda_val + epsilon_val)
        pw_costs_grad_bwd=(pw_costs_prime_paid-ctx.pw_costs_paid)/(lambda_val+epsilon_val)
        unary_grad_bwd = torch.from_numpy(unary_grad_bwd).to(float).to(device)
        pw_costs_grad_bwd = torch.from_numpy(pw_costs_grad_bwd).to(float).to(device)
        return unary_grad_bwd, pw_costs_grad_bwd, None, None, None

