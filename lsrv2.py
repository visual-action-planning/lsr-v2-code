import numpy as np
import networkx as nx
from scipy.optimize import minimize_scalar
from scipy.cluster.hierarchy import linkage, fcluster

# Format distance type from string to correct format
def format_distance_type(distance_type):
    if distance_type=='inf' or distance_type==np.inf:
        return np.inf
    else:
        return int(distance_type)


class LSRV2:
    def __init__(self, latent_map,  distance_type, cmax,  min_edge_w=0, min_node_m = 0,
    directed_graph = False,  a_lambda_format='stacking', verbose = False,lower_b=0,upper_b=3):
        self.latent_map = latent_map
        self.distance_type = format_distance_type(distance_type)
        self.c_max = cmax

        self.min_edge_w = min_edge_w
        self.min_node_m = min_node_m
        self.directed_graph = directed_graph               
        self.verbose = verbose
        self.a_lambda_format = a_lambda_format
        self.lower_b = lower_b
        self.upper_b = upper_b  

        #Variables for lsr
        self.Z_all=[]
        self.Z_all_data=[]
        self.Z_sys_is = []
        self.G1 = []
        self.G2 = []
        self.stats_dict = []
        self.times = []
        self.h_s=-1
        self.h_c=-1
        self.eps_approximation_computed = False

    def optimize_lsr(self):
        res_opt = minimize_scalar(self.f_lsr, bounds=(self.lower_b ,self.upper_b ), method='bounded')
        return self, res_opt, self.G2, self.stats

    def f_lsr(self,x0):

        # build lsrv2 
        G2, stats=self.build_lsr(x0)
        self.G2 = G2
        self.stats = stats
       
        # calc opt criteria
        if not self.directed_graph:
            S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
        if self.directed_graph:
            S = [G2.subgraph(c).copy() for c in nx.weakly_connected_components(G2)]
            
        num_edges = 0
        num_components=len(S)
        for g in S:
            num_edges = num_edges + g.number_of_edges()
        if num_components>self.c_max:
            return np.inf
        else:
            return -num_edges


    def build_lsr(self,x0):
        # PHASE 1: Build initial graph with all the nodes and edges
        self.lsr_phase_1(self.latent_map, self.directed_graph)
        # PHASE 2: Clustering phase
        self.lsr_phase_2(x0)
        # PHASE 3: Build graph
        self.lsr_phase_3(self.a_lambda_format)
        # PHASE 4: pruning
        self.lsr_phase_4(self.min_edge_w , self.min_node_m)
        return self.G2, self.stats_dict


    def lsr_phase_1(self, latent_map, directed_graph):
        verbose = self.verbose
        distance_type = self.distance_type

        # Build the Graph
        # *** Phase 1
        if directed_graph:
            G1 = nx.DiGraph()
            G2 = nx.DiGraph()
        else:
            G1=nx.Graph()
            G2=nx.Graph()
        self.G2 = G2
        counter=0
        Z_all=set()
        Z_all_data = []
        action_count = 0
        no_action_count = 0
        for latent_pair in latent_map:
            counter+=1
            if verbose:
                print("checking " + str(counter)+ " / " + str(len_latent_map)+ " build " + str(G1.number_of_nodes()) + " so far.")

            # get the latent coordinates
            z_pos_c1=latent_pair[0]
            z_pos_c2=latent_pair[1]
            action=latent_pair[2]
            c1_class_label=latent_pair[4]
            c2_class_label=latent_pair[5]

            # action pairs
            dis=np.linalg.norm(z_pos_c1-z_pos_c2,ord=distance_type)
            if action==1:
                a_lambda=np.array(latent_pair[3])
                c_idx=G1.number_of_nodes()
                G1.add_node(c_idx,idx=c_idx,pos=z_pos_c1, idx_lsr = -1, visited = 0,class_l=c1_class_label, pair_spec = (action, c_idx + 1))
                Z_all.add(c_idx)
                c_idx=G1.number_of_nodes()
                G1.add_node(c_idx,idx=c_idx,pos=z_pos_c2, idx_lsr = -1, visited = 0,class_l=c2_class_label, pair_spec = ())
                Z_all.add(c_idx)
                G1.add_edge(c_idx-1,c_idx,l=np.round(dis,1),a_lambda=a_lambda)
                action_count = action_count + 1
                
            # no action
            if action==0:
                c_idx=G1.number_of_nodes()
                G1.add_node(c_idx,idx=c_idx,pos=z_pos_c1, idx_lsr = -1, visited = 0,class_l=c1_class_label, pair_spec = (action, c_idx + 1))
                Z_all.add(c_idx)
                c_idx=G1.number_of_nodes()
                G1.add_node(c_idx,idx=c_idx,pos=z_pos_c2, idx_lsr = -1, visited = 0,class_l=c2_class_label, pair_spec = ())
                Z_all.add(c_idx)
                no_action_count = no_action_count + 1

            Z_all_data.append(z_pos_c1)
            Z_all_data.append(z_pos_c2)
            
        if verbose:
            print("***********Phase one done*******")
            print("Num nodes: " + str(G1.number_of_nodes()))
            print("Num edges: " + str(G1.number_of_edges()))
            print("num in Z_all: " + str(len(Z_all)) )

        self.G1 = G1
        self.Z_all = Z_all
        self.Z_all_data = Z_all_data
        self.action_count = action_count
        self.no_action_count = no_action_count


    def lsr_phase_2(self,max_d):
        # *** Phase 2
        verbose = self.verbose
        Z_sys_is=[]
        Z_all_data = np.array(self.Z_all_data)
        
        # format distance types
        if self.distance_type==1:
            metric='cityblock'
        if self.distance_type==2:
            metric='euclidean'
        if self.distance_type==np.inf:
            metric='chebyshev'

        # calculate dendogram (Z)
        Z = linkage(Z_all_data, method = "average", metric = metric)

        # build cluster using max distance in the dendogram to find it 
        c_lables= fcluster(Z, max_d, criterion='distance')

        # get number of cluster
        num_c=len(set(c_lables))
        
        # prepare Z_sys_is
        Z_sys_is=[]
        for i in range(num_c):
            W_z=set()
            Z_sys_is.append(W_z)
        # add samples to the right set
        for g in self.G1:
            Z_sys_is[c_lables[g]-1].add(g)

        # print result of phase 2
        if verbose:
            print("***********Phase two done*******")
            print("Num disjoint sets: " + str(len(Z_sys_is)))
            num_z_sys_nodes=0
            w_z_min=np.Inf
            w_z_max=-np.Inf
            for W_z in Z_sys_is:
                if len(W_z)<w_z_min:
                    w_z_min=len(W_z)
                if len(W_z) > w_z_max:
                    w_z_max=len(W_z)
                num_z_sys_nodes+=len(W_z)
            print("Total number of components: " + str(num_z_sys_nodes))
            print("Max number W_z: " + str(w_z_max)+ " min number w_z: " + str(w_z_min))

        self.Z_sys_is = Z_sys_is

    def get_LSR_node_pos(self, w_pos_all, W_z, g_idx):
        W_z_c_pos = np.mean(w_pos_all,axis=0)
        return W_z_c_pos


    def lsr_phase_3p1_nodes(self):       
        # *** Phase 3 
        Z_sys_is = self.Z_sys_is
        G1 = self.G1
        G2 = self.G2

        for W_z in Z_sys_is:
            w_pos_all = []
            idx_all = []
            c_idx = G2.number_of_nodes()
            if len(W_z)>0:
                for w in W_z:
                    w_pos=G1.nodes[w]['pos']
                    w_pos_all.append(w_pos)
                    idx_all.append(G1.nodes[w]['idx'])
                    G1.nodes[w]['idx_lsr'] = c_idx
                W_z_c_pos = self.get_LSR_node_pos(w_pos_all, W_z, c_idx)
                G2.add_node(c_idx,pos=W_z_c_pos,W_z=W_z,w_pos_all=w_pos_all, idx_all = idx_all)
        self.G1 = G1
        self.G2 = G2

    def lsr_phase_3p2_edges(self,a_lambda_format):
        G1 = self.G1
        G2 = self.G2
        distance_type = self.distance_type
        verbose = self.verbose
        for g2 in G2:
            if verbose:
                print(str(g2)+ " / " + str(G2.number_of_nodes()))
            W_z=G2.nodes[g2]['W_z']
            for w in W_z:
                w_pairs=G1.neighbors(w)
                for w_pair in w_pairs:
                    neig_lsr_idx = G1.nodes[w_pair]['idx_lsr']
                    if neig_lsr_idx >= 0:
                        dis=np.linalg.norm(G2.nodes[neig_lsr_idx]['pos']-G2.nodes[g2]['pos'],ord=distance_type)
                        if not G2.has_edge(g2,neig_lsr_idx):

                            if a_lambda_format == None:
                                G2.add_edge(g2,neig_lsr_idx,l=np.round(dis,1),ew=1)
                            else:
                                a_lambda=G1.edges[w, w_pair]['a_lambda']
                                G2.add_edge(g2,neig_lsr_idx,l=np.round(dis,1),ew=1,t_lambda=a_lambda)

                            if verbose:
                                print("Num edges: "+str(G2.number_of_edges()))
                        else:
                            # update edge
                            if a_lambda_format == None:
                                ew=G2.edges[g2, neig_lsr_idx]['ew']
                                l=G2.edges[g2, neig_lsr_idx]['l']
                                G2.edges[g2, neig_lsr_idx]['l']=(ew*l+dis)/(ew+1)
                                ew+=1
                                G2.edges[g2, neig_lsr_idx]['ew']=ew
                            else:
                                a_lambda=G1.edges[w, w_pair]['a_lambda']
                                g_lambda=G2.edges[g2, neig_lsr_idx]['t_lambda']
                                ew=G2.edges[g2, neig_lsr_idx]['ew']
                                # take simple weighted average
                                t_lambda= (ew*g_lambda+a_lambda)/(ew+1)
                                G2.edges[g2, neig_lsr_idx]['ew']=ew+1
                                G2.edges[g2, neig_lsr_idx]['t_lambda']=t_lambda
        self.G2 = G2


    def lsr_phase_3(self ,a_lambda_format):
        self.lsr_phase_3p1_nodes()
        self.lsr_phase_3p2_edges(a_lambda_format)
        if self.verbose:
            print("***********Phase three done*******")
            print("Num nodes: " + str(self.G2.number_of_nodes()))
            print("Num edges: " + str(self.G2.number_of_edges()))


    def lsr_phase_4(self,  min_edge_w , min_node_m):
        # *** Phase 4 Pruning
        G2 = self.G2
        verbose = self.verbose

        if verbose:
            print("Pruning edges with ew < " + str(min_edge_w))
        num_edges=G2.number_of_edges()
        remove_edges=[]
        for edge in G2.edges:
            sidx=edge[0]
            gidx=edge[1]
            ew=G2.edges[sidx, gidx]['ew']
            if ew < min_edge_w:
                remove_edges.append((sidx,gidx))
        for re in remove_edges:
            G2.remove_edge(re[0],re[1])
        num_edges_p=G2.number_of_edges()
        if verbose:
            if num_edges > 0:
                print("pruned " + str(num_edges-num_edges_p) + " edges ( " + str(100-(num_edges_p*100.)/num_edges) + " %")
            else:
                print("pruning: num_edges = 0")

        # prune weak nodes
        if verbose:
            print("Pruning nodes with mearges < " + str(min_node_m))
        num_nodes=G2.number_of_nodes()
        remove_nodes=[]
        for g in G2.nodes:
            ngm=G2.nodes[g]['w_pos_all']
            if len(ngm) < min_node_m:
                remove_nodes.append(g)

        for re in remove_nodes:
            for idx in G2.nodes[re]['idx_all']:
                self.G1.nodes[idx]['idx_lsr'] = -1
            G2.remove_node(re)

        num_nodes_p=G2.number_of_nodes()
        if verbose:
            if num_nodes > 0:
                print("pruned " + str(num_nodes-num_nodes_p) + " nodes ( " + str(100-(num_nodes_p*100.)/num_nodes) + " %")

        # prune single nodes
        num_nodes=G2.number_of_nodes()
        remove_nodes=[]
        isolates=nx.isolates(G2)
        for iso in isolates:
            remove_nodes.append(iso)

        for re in remove_nodes:
            for idx in G2.nodes[re]['idx_all']:
                self.G1.nodes[idx]['idx_lsr'] = -1
            G2.remove_node(re)      

        if verbose:
            print("pruned " + str(num_nodes-G2.number_of_nodes()) +" isolated nodes")
            print("final graph ************************************")
            print("Num nodes: " + str(G2.number_of_nodes()))
            print("Num edges: " + str(G2.number_of_edges()))     

        self.stats_dict = {'num_nodes':num_nodes_p}
        self.G2 = G2