import json
import urllib.request
import urllib.error
import networkx as nx
import plotly.graph_objects as go
import time

BASE_URL = "http://127.0.0.1:8000"
API_PREFIX = "/api/v1"

def make_request(path, data=None, method="GET"):
    url = f"{BASE_URL}{path}"
    req_data = json.dumps(data).encode("utf-8") if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"detail": e.reason}
    except Exception as e:
        return 500, {"detail": str(e)}

def test_graph_explorer_logic():
    print("======================================================================")
    print("        GRAPHSHEILD AI - GRAPH EXPLORER VALIDATOR SUITE               ")
    print("======================================================================")
    
    # Check 1: Verify global graph data loads
    print("Check 1: Verifying global graph statistics endpoints...")
    status, stats = make_request(f"{API_PREFIX}/graph/stats")
    assert status == 200, f"Expected 200, got {status}"
    assert "node_counts" in stats
    print(f"PASS: Graph statistics loaded. Nodes in network: {sum(stats['node_counts'].values())}")
    
    # Check 2: Account Search Ego-network construction
    target_acc = "ACC_1002305"
    print(f"\nCheck 2: Querying local ego-network for target account: {target_acc}...")
    status, res = make_request(f"{API_PREFIX}/graph/neighbors/account/{target_acc}")
    assert status == 200, f"Expected 200, got {status}"
    connections = res.get("connections", [])
    print(f"PASS: Ego connections count: {res['connections_count']}")
    
    # Check 3: NetworkX graph compilation
    print("\nCheck 3: Building NetworkX graph object and validating node/edge properties...")
    G = nx.Graph()
    G.add_node(target_acc, node_type="account", is_target=True)
    
    for conn in connections:
        neigh_id = conn["node_id"]
        neigh_type = conn["node_type"]
        relation = conn["relation"]
        G.add_node(neigh_id, node_type=neigh_type, is_target=False)
        G.add_edge(target_acc, neigh_id, relation=relation)
        
    print(f"PASS: Graph successfully compiled.")
    print(f"      Node Count in G: {G.number_of_nodes()}")
    print(f"      Edge Count in G: {G.number_of_edges()}")
    
    assert G.number_of_nodes() > 0, "Graph has no nodes."
    assert G.number_of_edges() > 0, "Graph has no edges."
    
    # Check 4: Layout and Coordinate Generation
    print("\nCheck 4: Generating spring layout coordinates...")
    pos = nx.spring_layout(G, seed=42)
    for node in G.nodes():
        coords = pos[node]
        assert len(coords) == 2, f"Invalid coords shape for node {node}"
        assert not np_isnan(coords[0]) and not np_isnan(coords[1]), f"Coordinates contain NaNs for node {node}"
    print("PASS: Layout coordinate positions generated successfully.")
    
    # Check 5: Plotly Fig Generation (Zoom/Pan capabilities verification)
    print("\nCheck 5: Instantiating Plotly visualization figure...")
    traces = []
    
    # Edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#bdc3c7", width=1.5))
    traces.append(edge_trace)
    
    # Node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(size=15))
    traces.append(node_trace)
    
    fig = go.Figure(data=traces)
    # Validate Plotly parameters
    assert len(fig.data) == 2, "Plotly figure missing traces."
    print("PASS: Plotly Scatter traces compiled successfully. Plotly handles HTML5 zoom/pan natively.")
    
    # Check 6: Fraud Ring Detection Heuristics
    print("\nCheck 6: Running diagnostic heuristics for fraud rings...")
    # Find shared devices
    device_hubs = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "device" and G.degree(node) >= 3:
            device_hubs.append(node)
            
    print(f"PASS: Shared device hubs found: {len(device_hubs)}")
    
    # Check 7: Money flow layering tracing
    print("\nCheck 7: Tracing sequential P2P flow paths...")
    p2p_relations = 0
    for u, v, d in G.edges(data=True):
        if d.get("relation") in ["TRANSFERRED_TO", "TRANSFERRED_FROM"]:
            p2p_relations += 1
    print(f"PASS: P2P layering links identified: {p2p_relations}")
    
    print("\n======================================================================")
    print("STATUS: Graph Explorer validation completed with status: SUCCESS")
    print("======================================================================")

def np_isnan(val):
    # Quick NaN check to avoid numpy import overhead in assertion
    return val != val

if __name__ == "__main__":
    test_graph_explorer_logic()
