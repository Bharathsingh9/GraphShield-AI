import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from visualization.utils import apply_lbg_theme, make_api_request, render_section_header

st.set_page_config(page_title="Network Explorer - GraphShield AI", layout="wide")
apply_lbg_theme()

st.markdown('# <span class="gradient-text">🌐 Network Explorer</span>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.05rem; font-weight: 500; color: #94A3B8; margin-top: -0.6rem; margin-bottom: 1.5rem;">Interactive Heterogeneous Relationship Mapping & Path Tracing</div>', unsafe_allow_html=True)

# Cache the set of fraud accounts for rapid lookup
@st.cache_data(show_spinner="Loading fraud metadata index...")
def get_fraud_accounts():
    try:
        df = pd.read_csv(
            "d:/fraud_detection/data/processed/engineered_transactions.csv", 
            usecols=["sender_account_id", "receiver_account_id", "fraud_label"]
        )
        fraud_df = df[df["fraud_label"] == 1]
        fraud_accs = set(fraud_df["sender_account_id"].dropna().unique()) | set(fraud_df["receiver_account_id"].dropna().unique())
        return fraud_accs
    except Exception as e:
        st.warning(f"Unable to index fraud accounts: {str(e)}")
        return set()

fraud_accounts = get_fraud_accounts()

# Layout: Left control column, Right visualization canvas
col_ctrl, col_viz = st.columns([1, 2.5])

with col_ctrl:
    with st.container(border=True):
        render_section_header("Network Query Controls", "Define query entity and exploration scope", "🔍")
        
        node_type = st.selectbox(
            "Node Category", 
            ["account", "customer", "device", "merchant"],
            help="Type of entity to search."
        )
        
        node_id_input = st.text_input(
            "Enter Node ID", 
            value="ACC_1002305", 
            placeholder="e.g. ACC_1002305, DEV_H_99999"
        )
        
        expand_2hop = st.checkbox(
            "Expand to 2-Hops (Entity Sharing)", 
            value=True,
            help="Trace shared entity clusters such as devices or customer profiles."
        )
        
        st.markdown("---")
        render_section_header("Path Tracing Engine", "Trace connections between two ledger accounts", "⚡")
        
        trace_path_enabled = st.checkbox("Trace Path Between Two Accounts", value=False)
        target_path_b = st.text_input("Destination Account ID (For Tracing):", value="ACC_1001250")
        
        search_graph = st.button("Render Network Visualization", type="primary")

with col_viz:
    if search_graph and node_id_input:
        target_node = node_id_input.strip()
        
        with st.spinner(f"Compiling network relations for {target_node}..."):
            # 1. Fetch relations from API
            res_1hop = make_api_request("GET", f"/graph/neighbors/{node_type}/{target_node}")
            
            if res_1hop:
                G = nx.Graph()
                
                # Add central target node
                G.add_node(
                    target_node, 
                    node_type=node_type, 
                    is_target=True, 
                    is_fraud=(target_node in fraud_accounts)
                )
                
                connections_1hop = res_1hop.get("connections", [])
                nodes_to_expand = []
                
                for conn in connections_1hop:
                    neigh_id = conn["node_id"]
                    neigh_type = conn["node_type"]
                    relation = conn["relation"]
                    
                    G.add_node(
                        neigh_id, 
                        node_type=neigh_type, 
                        is_target=False, 
                        is_fraud=(neigh_id in fraud_accounts)
                    )
                    G.add_edge(target_node, neigh_id, relation=relation)
                    
                    if expand_2hop and neigh_type in ["device", "customer"]:
                        nodes_to_expand.append((neigh_id, neigh_type))
                
                # 2. Expand to 2-Hops if checked
                if expand_2hop:
                    for hub_id, hub_type in nodes_to_expand[:6]:
                        res_2hop = make_api_request("GET", f"/graph/neighbors/{hub_type}/{hub_id}", timeout=2.0)
                        if res_2hop:
                            connections_2hop = res_2hop.get("connections", [])
                            for conn2 in connections_2hop:
                                neigh2_id = conn2["node_id"]
                                neigh2_type = conn2["node_type"]
                                relation2 = conn2["relation"]
                                
                                if not G.has_node(neigh2_id):
                                    G.add_node(
                                        neigh2_id, 
                                        node_type=neigh2_type, 
                                        is_target=False, 
                                        is_fraud=(neigh2_id in fraud_accounts)
                                    )
                                if not G.has_edge(hub_id, neigh2_id):
                                    G.add_edge(hub_id, neigh2_id, relation=relation2)
                
                # 3. Path Tracing Logic
                highlighted_edges = set()
                highlighted_nodes = set()
                
                if trace_path_enabled and target_path_b:
                    b_node = target_path_b.strip()
                    # Add destination node if not already in graph to trace connection
                    if not G.has_node(b_node):
                        G.add_node(b_node, node_type="account", is_target=False, is_fraud=(b_node in fraud_accounts))
                        # Connect via a simulated relation or query neighbors
                        res_b = make_api_request("GET", f"/graph/neighbors/account/{b_node}", timeout=2.0)
                        if res_b:
                            for conn in res_b.get("connections", [])[:3]:
                                if G.has_node(conn["node_id"]):
                                    G.add_edge(b_node, conn["node_id"], relation=conn["relation"])
                                    
                    try:
                        shortest_path = nx.shortest_path(G, source=target_node, target=b_node)
                        for i in range(len(shortest_path) - 1):
                            highlighted_edges.add((shortest_path[i], shortest_path[i+1]))
                            highlighted_edges.add((shortest_path[i+1], shortest_path[i]))
                        highlighted_nodes = set(shortest_path)
                        st.success(f"✅ **Path Traced successfully**: Found connection path of length {len(shortest_path)-1} edges.")
                    except nx.NetworkXNoPath:
                        st.warning("⚠️ No path exists between the selected accounts in the active local subgraph.")
                
                # Render using Plotly
                pos = nx.spring_layout(G, k=1.2, seed=42)
                
                # Build Edges
                edge_traces = []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    
                    is_highlighted = (edge[0], edge[1]) in highlighted_edges
                    color = "#EF4444" if is_highlighted else "#334155"
                    width = 3 if is_highlighted else 1.2
                    
                    edge_traces.append(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        line=dict(width=width, color=color),
                        mode='lines',
                        hoverinfo='none',
                        showlegend=False
                    ))
                
                # Nodes configuration
                node_types = {
                    "account": {"color": "#3498db", "symbol": "circle", "name": "Account"},
                    "customer": {"color": "#f39c12", "symbol": "square", "name": "Customer"},
                    "merchant": {"color": "#9b59b6", "symbol": "diamond", "name": "Merchant"},
                    "device": {"color": "#7f8c8d", "symbol": "hexagon", "name": "Device"}
                }
                
                traces = edge_traces
                
                # Standard and special node groupings
                for n_type, style in node_types.items():
                    n_x, n_y, n_text, n_size, n_colors = [], [], [], [], []
                    
                    for node, data in G.nodes(data=True):
                        # Filter out fraud nodes and central path-highlighted nodes to style them separately
                        if data.get("node_type") == n_type and not data.get("is_fraud") and node not in highlighted_nodes:
                            x, y = pos[node]
                            n_x.append(x)
                            n_y.append(y)
                            deg = G.degree(node)
                            target_lbl = " (SEARCHED)" if data.get("is_target") else ""
                            n_text.append(f"Entity: {node}{target_lbl}<br>Type: {n_type.upper()}<br>Sub-Connections: {deg}")
                            n_size.append(26 if data.get("is_target") else 18)
                            n_colors.append(style["color"])
                            
                    if n_x:
                        traces.append(go.Scatter(
                            x=n_x, y=n_y, mode='markers', name=style["name"],
                            marker=dict(symbol=style["symbol"], color=n_colors, size=n_size, line=dict(width=1.5, color='#0F172A')),
                            hoverinfo='text', text=n_text
                        ))
                
                # Highlight path nodes
                if highlighted_nodes:
                    hp_x, hp_y, hp_text, hp_size = [], [], [], []
                    for node in highlighted_nodes:
                        x, y = pos[node]
                        hp_x.append(x)
                        hp_y.append(y)
                        hp_text.append(f"<b>Path Tracer Entity</b>: {node}")
                        hp_size.append(28)
                    traces.append(go.Scatter(
                        x=hp_x, y=hp_y, mode='markers+text', name="⭐ Path Highlight",
                        marker=dict(symbol='star', color='#EAB308', size=hp_size, line=dict(width=2, color='#0F172A')),
                        hoverinfo='text', text=hp_text, textposition="bottom center"
                    ))
                
                # High Risk / Fraud nodes
                f_x, f_y, f_text, f_size = [], [], [], []
                for node, data in G.nodes(data=True):
                    if data.get("is_fraud") and node not in highlighted_nodes:
                        x, y = pos[node]
                        f_x.append(x)
                        f_y.append(y)
                        deg = G.degree(node)
                        f_text.append(f"🚨 **FRAUD NODE**: {node}<br>Sub-Connections: {deg}")
                        f_size.append(28)
                        
                if f_x:
                    traces.append(go.Scatter(
                        x=f_x, y=f_y, mode='markers', name="🚨 Fraud Entity",
                        marker=dict(symbol='circle', color='#FF1744', size=f_size, line=dict(width=2.5, color='#7F1D1D')),
                        hoverinfo='text', text=f_text
                    ))
                
                # Plotly Canvas
                fig = go.Figure(data=traces, layout=go.Layout(
                    showlegend=True, hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=20),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=550,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(15, 23, 42, 0.8)")
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Topological Diagnostic checks to demonstrate GNN utility
                st.markdown("### 🕸️ Relationship Anomaly Diagnostics")
                
                device_hubs = []
                mule_chains = []
                
                for node, data in G.nodes(data=True):
                    node_t = data.get("node_type")
                    deg = G.degree(node)
                    if node_t == "device" and deg >= 3:
                        device_hubs.append(node)
                
                # Check shared devices
                if device_hubs:
                    st.error("🚨 **Shared Device Fraud Ring Flagged**")
                    for dev in device_hubs:
                        connected = [n for n in G.neighbors(dev) if G.nodes[n].get("node_type") == "account"]
                        st.markdown(f"- Device Node `{dev}` is shared by **{len(connected)} separate accounts**: {', '.join([f'`{a}`' for a in connected])}.")
                        # Highlight GNN utility
                        st.info("💡 **GNN Utility Demonstration**: Standard transactional monitoring checks isolated accounts separately. The GraphSAGE model propagates the shared device connectivity embedding, allowing the classifier to score all connected accounts as high risk simultaneously.")
                
                # Check P2P chains
                for u, v, key in G.edges(data=True):
                    rel = key.get("relation")
                    if rel in ["TRANSFERRED_TO", "TRANSFERRED_FROM"]:
                        mule_chains.append((u, v))
                
                if len(mule_chains) >= 2:
                    st.warning("⚠️ **Layering Money Mule Chain Detected**")
                    for u, v in mule_chains:
                        st.markdown(f"- P2P Money Flow: `{u}` ──> `{v}`")
                    st.info("💡 **GNN Utility Demonstration**: GNN message passing accumulates multi-hop path steps. Layered transfer chains are identified as high-risk motifs by the node embedding representation.")
                    
                if not device_hubs and len(mule_chains) < 2:
                    st.success("🟢 No abnormal device sharing or transfer chaining patterns found in this local subgraph.")
                    
            else:
                st.error("Entity not found in the graph database. Verify the ID.")
    else:
        st.info("💡 Click 'Render Network Visualization' to display the interactive graph explorer.")
