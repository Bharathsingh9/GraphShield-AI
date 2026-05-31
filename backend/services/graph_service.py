import os
import sys
import torch

# Add root path for GNN imports
sys.path.append("d:/fraud_detection")

from backend.services.prediction_service import PredictionService

class GraphService:
    """
    Service layer for querying GraphShield AI's heterogeneous network structures.
    Exposes graph metadata, structural density, and local neighborhood relationships.
    """
    @classmethod
    def get_graph_stats(cls) -> dict:
        predictor = PredictionService.get_predictor()
        data = predictor.data
        
        node_counts = {
            "customer": int(data["customer"].num_nodes),
            "account": int(data["account"].num_nodes),
            "merchant": int(data["merchant"].num_nodes),
            "device": int(data["device"].num_nodes)
        }
        
        edge_counts = {
            "owns (Customer -> Account)": int(data["customer", "owns", "account"].num_edges),
            "uses (Account -> Device)": int(data["account", "uses", "device"].num_edges),
            "performs (Account -> Account)": int(data["account", "performs", "account"].num_edges),
            "paid_to (Account -> Merchant)": int(data["account", "paid_to", "merchant"].num_edges)
        }
        
        total_nodes = sum(node_counts.values())
        total_edges = sum(edge_counts.values())
        
        # Calculate graph density for heterogeneous structures
        # Density = E / (V * (V - 1))
        if total_nodes > 1:
            density = float(total_edges / (total_nodes * (total_nodes - 1)))
        else:
            density = 0.0
            
        return {
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "density": round(density, 6)
        }

    @classmethod
    def get_node_neighbors(cls, node_type: str, node_id: str) -> dict:
        """
        Retrieves the immediate connected neighbors of a given node in the graph.
        """
        predictor = PredictionService.get_predictor()
        data = predictor.data
        
        # Resolve node-to-index mapping
        if node_type == "customer":
            mapping = predictor.cust_map
        elif node_type == "account":
            mapping = predictor.acc_map
        elif node_type == "merchant":
            mapping = predictor.merch_map
        elif node_type == "device":
            mapping = predictor.dev_map
        else:
            return {"status": "ERROR", "message": f"Unsupported node type: {node_type}"}
            
        if node_id not in mapping:
            return {"status": "ERROR", "message": f"Node ID {node_id} of type {node_type} not found in mappings."}
            
        node_idx = mapping[node_id]
        
        # Build lists of reverse mappings to recover string IDs from indices
        rev_acc_map = {v: k for k, v in predictor.acc_map.items()}
        rev_cust_map = {v: k for k, v in predictor.cust_map.items()}
        rev_merch_map = {v: k for k, v in predictor.merch_map.items()}
        rev_dev_map = {v: k for k, v in predictor.dev_map.items()}
        
        connections = []
        
        # 1. Check OWNS (Customer -> Account) edges
        if node_type == "customer":
            owns_idx = data["customer", "owns", "account"].edge_index
            matched_accs = owns_idx[1, owns_idx[0] == node_idx].tolist()
            for a in matched_accs:
                connections.append({"relation": "OWNS", "node_type": "account", "node_id": rev_acc_map.get(a)})
        elif node_type == "account":
            owns_idx = data["customer", "owns", "account"].edge_index
            matched_custs = owns_idx[0, owns_idx[1] == node_idx].tolist()
            for c in matched_custs:
                connections.append({"relation": "OWNED_BY", "node_type": "customer", "node_id": rev_cust_map.get(c)})
                
        # 2. Check USES (Account -> Device) edges
        if node_type == "account":
            uses_idx = data["account", "uses", "device"].edge_index
            matched_devs = uses_idx[1, uses_idx[0] == node_idx].tolist()
            for d in matched_devs:
                connections.append({"relation": "USES", "node_type": "device", "node_id": rev_dev_map.get(d)})
        elif node_type == "device":
            uses_idx = data["account", "uses", "device"].edge_index
            matched_accs = uses_idx[0, uses_idx[1] == node_idx].tolist()
            for a in matched_accs:
                connections.append({"relation": "USED_BY", "node_type": "account", "node_id": rev_acc_map.get(a)})
                
        # 3. Check PERFORMS (Account -> Account P2P) edges
        if node_type == "account":
            perf_idx = data["account", "performs", "account"].edge_index
            # Sent P2P transfers
            sent_accs = perf_idx[1, perf_idx[0] == node_idx].tolist()
            for sa in sent_accs:
                connections.append({"relation": "TRANSFERRED_TO", "node_type": "account", "node_id": rev_acc_map.get(sa)})
            # Received P2P transfers
            recv_accs = perf_idx[0, perf_idx[1] == node_idx].tolist()
            for ra in recv_accs:
                connections.append({"relation": "TRANSFERRED_FROM", "node_type": "account", "node_id": rev_acc_map.get(ra)})
                
        # 4. Check PAID_TO (Account -> Merchant) edges
        if node_type == "account":
            paid_idx = data["account", "paid_to", "merchant"].edge_index
            matched_merchs = paid_idx[1, paid_idx[0] == node_idx].tolist()
            for m in matched_merchs:
                connections.append({"relation": "PAID_TO", "node_type": "merchant", "node_id": rev_merch_map.get(m)})
        elif node_type == "merchant":
            paid_idx = data["account", "paid_to", "merchant"].edge_index
            matched_accs = paid_idx[0, paid_idx[1] == node_idx].tolist()
            for a in matched_accs:
                connections.append({"relation": "RECEIVED_PAYMENT_FROM", "node_type": "account", "node_id": rev_acc_map.get(a)})
                
        return {
            "node_type": node_type,
            "node_id": node_id,
            "connections_count": len(connections),
            "connections": connections
        }
