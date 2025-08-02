import heapq
import math

class RouteFinder:
    
    def __init__(self, graph):
        self.graph = graph
    
    def optimized_dijkstra(self, start_airports, end_airports):
        dist = {n: math.inf for n in self.graph}
        prev = {n: None for n in self.graph}
        pq = []
        
        for s in start_airports:
            if s in dist:
                dist[s] = 0
                heapq.heappush(pq, (0, s))
        
        visited = set()
        best_dist = math.inf
        best_node = None
        
        while pq:
            current_dist, node = heapq.heappop(pq)
            
            if node in visited:
                continue
                
            visited.add(node)
            
            if node in end_airports:
                best_dist = current_dist
                best_node = node
                break
            
            for neighbor, weight in self.graph.get(node, []):
                if neighbor not in dist:
                    continue
                    
                new_dist = current_dist + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = node
                    heapq.heappush(pq, (new_dist, neighbor))
        
        if math.isinf(best_dist):
            return math.inf, []
        
        path = []
        cur = best_node
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        
        return best_dist, path[::-1] 
