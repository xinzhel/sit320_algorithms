def BidirectionalSearch(s, t, G):
    # 's' is the source node, 't' is the target node, 'G' is the graph

    for v in G.vertices:
        v.status = "unvisited"

    n = len(G.vertices)

    Ls = [ [] for _ in range(n) ]
    Lt = [ [] for _ in range(n) ]

    Ls[0] = [s]
    Lt[0] = [t]

    s.status = "visited from source"
    t.status = "visited from target"

    for i in range(n):

        for u in Ls[i]:
            for v in u.getOutNeighbors():
                if v.status == "unvisited":
                    v.status = "visited from source"
                    Ls[i + 1].append(v)
                elif v.status == "visited from target":
                    # When a node has been visited from both source and target, a path is found
                    return True

        for u in Lt[i]:
            for v in u.getOutNeighbors():
                if v.status == "unvisited":
                    v.status = "visited from target"
                    Lt[i + 1].append(v)
                elif v.status == "visited from source":
                    # When a node has been visited from both source and target, a path is found
                    return True

    return False  # No path found after exploring all nodes

# How can I keep track of the parent of each node?
# Keeping track of the parent of each node can be accomplished by maintaining a dictionary (or other suitable data structure) that maps from each node to its parent. 
def BidirectionalSearch(s, t, G):
    # 's' is the source node, 't' is the target node, 'G' is the graph

    for v in G.vertices:
        v.status = "unvisited"

    n = len(G.vertices)

    Ls = [[] for _ in range(n)]
    Lt = [[] for _ in range(n)]

    parent_s = {s: None}  # Tracks parents for nodes visited from source
    parent_t = {t: None}  # Tracks parents for nodes visited from target

    Ls[0] = [s]
    Lt[0] = [t]

    s.status = "visited from source"
    t.status = "visited from target"

    for i in range(n):

        for u in Ls[i]:
            for v in u.getOutNeighbors():
                if v.status == "unvisited":
                    v.status = "visited from source"
                    parent_s[v] = u  # Keep track of parent
                    Ls[i + 1].append(v)
                elif v.status == "visited from target":
                    # When a node has been visited from both source and target, a path is found
                    parent_s[v] = u  # Keep track of parent
                    return parent_s, parent_t, v  # Return the parents dictionaries and the intersecting node

        for u in Lt[i]:
            for v in u.getOutNeighbors():
                if v.status == "unvisited":
                    v.status = "visited from target"
                    parent_t[v] = u  # Keep track of parent
                    Lt[i + 1].append(v)
                elif v.status == "visited from source":
                    # When a node has been visited from both source and target, a path is found
                    parent_t[v] = u  # Keep track of parent
                    return parent_s, parent_t, v  # Return the parents dictionaries and the intersecting node

    return None, None, None  # No path found after exploring all nodes

def reconstruct_path(parent_s, parent_t, intersect_node):
    # Reconstruct the path from source to target

    path = []

    # Trace path from source to intersect_node
    current = intersect_node
    while current is not None:
        path.append(current)
        current = parent_s[current]

    path = path[::-1]  # Reverse the path

    # Trace path from intersect_node to target
    current = parent_t[intersect_node]
    while current is not None:
        path.append(current)
        current = parent_t[current]

    return path


def isBipartite(w, G):
    
    for v in G.vertices:
        v.color = None  # Instead of 'status', we use 'color'
    
    n = len(G.vertices)
    
    Ls = [ [] for _ in range(n) ]
    
    Ls[0] = [w]
    w.color = "white"
    
    for i in range(n):
        
        for u in Ls[i]:
            next_color = "white" if u.color == "black" else "black"
            
            for v in u.getOutNeighbors():
                
                if v.color is None:
                    v.color = next_color
                    Ls[i + 1].append(v)
                elif v.color == u.color:
                    return False  # Two adjacent vertices have the same color. Hence, the graph is not bipartite.
                    
    return True  # No adjacent vertices with the same color were found. Hence, the graph is bipartite.


# SCCs
def DFS_helper(w, stack):
    w.status = "inprogress"
    for v in w.getOutNeighbors():
        if v.status == "unvisited":
            DFS_helper(v, stack)
    stack.append(w)
    w.status = "done"

def DFS_helper_reverse(w, visited):
    print(w)  # printing the node as part of SCC
    visited.add(w)
    for v in w.getOutNeighbors():
        if v not in visited:
            DFS_helper_reverse(v, visited)
    
def SCC(G):
    # Step 1 & 2
    stack = []
    for v in G.vertices:
        v.status = "unvisited"
    for v in G.vertices:
        if v.status == "unvisited":
            DFS_helper(v, stack)
    
    # Step 3: Reverse the graph
    GR = reverseGraph(G)  # Assuming reverseGraph function is implemented
    
    # Step 4
    visited = set()
    while stack:
        v = stack.pop()
        if v not in visited:
            print("New SCC:")
            DFS_helper_reverse(v, visited)

# Assuming a function to reverse the graph is defined as follows:
def reverseGraph(G):
    GR = Graph(len(G.vertices))
    for v in G.vertices:
        for w in v.getOutNeighbors():
            GR.addEdge(w, v)
    return GR
