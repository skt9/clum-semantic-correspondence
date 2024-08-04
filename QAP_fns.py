from typing import List
import torch
import numpy as np
from mpopt.qap.model import Assignment,Edge
from mpopt import qap,utils
from sklearn.neighbors import NearestNeighbors

def remapHypotheses(topKHypotheses: torch.LongTensor):
    remapToOrig=[None]*358
    max_hyp=torch.max(topKHypotheses)
    current_num=0
    for num in range(max_hyp+1):
        inds=torch.where(topKHypotheses==num)
        if(len(inds[0])>0):
            topKHypotheses[topKHypotheses==num]=current_num
            remapToOrig[current_num]=num
            current_num+=1
    return topKHypotheses,remapToOrig

#   Adds margins to winning label
def augmentUnary(unaries:torch.Tensor, gtPermMatrix:torch.Tensor,alpha:float):
    return unaries+alpha*gtPermMatrix

def computeNeighborhoodOfPoints(points:torch.Tensor,num_nbrs:int=5):

    points=points.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=num_nbrs, algorithm='ball_tree').fit(points)
    distances, neighbors = nbrs.kneighbors(points)
    return neighbors #,distances

def batchOuterProduct(vecs1:torch.Tensor, vecs2:torch.Tensor):

    return torch.einsum('bi,bj->bij', (vecs1, vecs2))

def update_pw_costs(lambda_val: float,pw_costs: dict,pw_gradients: dict):

    for ky in pw_costs.keys():
        pw_costs[ky]+=lambda_val*pw_gradients[ky]
    return pw_costs

def compute_assignments(unaries: np.array,hypotheses: np.array):
    assignments=[Assignment(i,hypotheses[i,j].item(),float(val)) for i,un_vec in enumerate(unaries) for j,val in enumerate(un_vec) ]
    return assignments

def compute_edges_with_assignments(pwCosts: torch.Tensor,edges,assignments: List[Assignment],numHypothesesPerUnary):
    """
        This function only works for pwCosts as a fully connected pwCosts. So all points are present
        We assume all unaries have the same hypotheses.
    """
    assert(len(pwCosts.shape)==3)
    assert(pwCosts.shape[0]==len(edges))

    qap_edge_costs=[]
    for i,ed in enumerate(edges):
        un0_starting_assignmentID=ed[0]*numHypothesesPerUnary
        un1_starting_assignmentID=ed[1]*numHypothesesPerUnary
        pw=pwCosts[i]
        for i in range(pw.shape[0]):
            for j in range(pw.shape[1]):
                if(assignments[un0_starting_assignmentID+i].right==assignments[un1_starting_assignmentID+j].right):
                    continue
                qap_edge_costs.append(Edge(un0_starting_assignmentID+i,un1_starting_assignmentID+j,pw[i,j]))
    return qap_edge_costs
    
def compute_edges2(edges_graph:List,pwCosts:torch.Tensor,assignments,topK:int):

    edges=[]
    for i,ed in enumerate(edges_graph):
        un0_ass_id=ed[0]*len(pwCosts.shape[0])
        un1_ass_id=ed[1]
        pw=pwCosts[i]
        for i in range(pw.shape[0]):
            for j in range(pw.shape[1]):
                if(assignments[un0_ass_id+i].right==assignments[un1_ass_id+j].right):
                    # print(f"{i} edge has some hypotheses: {assignments[un0_ass_id+i]} {assignments[un1_ass_id+j]}")
                    continue
                edges.append(Edge(un0_ass_id+i,un1_ass_id+j,pw[i,j]))
    return edges

def solve_qap(unaries:torch.Tensor,hypotheses:torch.Tensor,pw_cost:torch.Tensor,edges:List[tuple],params:dict):

    assert(len(edges)==pw_cost.shape[0])

    unaries_np=unaries.cpu().detach().numpy()-params["const"]
    hypotheses_np=hypotheses.cpu().detach().numpy().astype(int)
    pw_cost_np=pw_cost.cpu().detach().numpy()
    num_unaries=hypotheses.shape[0]
    num_hypotheses=np.max(hypotheses_np)+1
    num_hypotheses_per_unary=len(hypotheses[0])

    labeling=[]

    assignments=compute_assignments(unaries_np,hypotheses_np)

    qap_edge_costs=compute_edges_with_assignments(pw_cost_np,edges,assignments,num_hypotheses_per_unary)

    #   Constructing the models
    model=qap.Model(num_unaries,num_hypotheses,len(assignments),len(qap_edge_costs))
    #   Adding the unaries
    for i,ass in enumerate(assignments):
        model.add_assignment(i,ass.left,ass.right,ass.cost)
    #   Adding the pairwise costs
    for ed in qap_edge_costs:
        if(ed.assignment1==ed.assignment2):
            continue
        model.add_edge(ed.assignment1,ed.assignment2,ed.cost)

    deco=qap.ModelDecomposition(model,with_uniqueness=1,unary_side='left')
    solver=qap.construct_solver(deco)
    solver.run(10,1,1)        
    primals=qap.extract_primals(deco,solver)
    cost = primals.evaluate()

    print("------------------------------------------------------------")
    print(primals.labeling)
    # primals.labeling[primals.labeling==None] = -1
    primals_labeling = [-1 if x is None else x for x in primals.labeling]
    print(primals_labeling)
    labeling=np.array(primals_labeling).astype(np.uint32)
    print("------------------------------------------------------------")
    print(f"\n\n")
    
    # labeling[labeling==None]=-1
    # print(labeling)
    # print(labeling.dtype)
    return torch.from_numpy(labeling)   #   convert labels to torch

def createPermutationMatrix(labeling,num_unaries,num_hypotheses_actual,remap_to_orig, gtLabeling):
    assert(len(labeling)==num_unaries)
    result_permutation_matrix=np.zeros((num_unaries,num_hypotheses_actual))
    for i,lbl in enumerate(labeling):
        orig_hyp=remap_to_orig[lbl]
        result_permutation_matrix[i,orig_hyp]=1
    return result_permutation_matrix
    
def get_results_mask(labeling,hypotheses,pw_costs,edges):
    """

    """
    num_unaries, num_hypotheses=hypotheses.shape
    assert(len(labeling)==num_unaries)

    unary_costs_paid=np.zeros((num_unaries,num_hypotheses))
    for i,lbl in enumerate(labeling):
        unary_costs_paid[i,lbl]=1
    
    num_edges = len(edges)
    pw_costs_paid = np.zeros((num_edges,num_hypotheses,num_hypotheses))
    pw_labels = {}
    for i,ed in enumerate(edges):
        lbl_i,lbl_j=labeling[ed[0]].item(),labeling[ed[1]].item()
        # print(f"lbl_i: {lbl_i} lbl_j: {lbl_j}")
        # print(f"type(lbl_i): {type(lbl_i)} type(lbl_j): {type(lbl_j)}")
        pw_costs_paid[i,lbl_i,lbl_j]=1.
        pw_labels[(ed[0],ed[1])]= (lbl_i,lbl_j)


    return unary_costs_paid, pw_costs_paid, labeling, pw_labels