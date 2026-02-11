from __future__ import annotations



import heapq

import math

from dataclasses import dataclass

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple





@dataclass(frozen=True)

class PathResult:

    """A route candidate.

    cost: routing objective (distance + penalties)
    distance_km: pure great-circle distance (no penalties)
    """



    cost: float

    path: List[str]

    distance_km: float



class RouteFinder:

    """
    Routing utilities:
      - Dijkstra shortest path (supports banned nodes/edges)
      - Yen's algorithm for K-shortest loopless paths
      - Multi-source/multi-target K-shortest paths via a temporary super-source/sink
    """



    def __init__(self, graph: Dict[str, List[Tuple[str, float]]], edge_distance_km: Optional[Dict[Tuple[str, str], float]] = None):

        self.graph = graph



        if edge_distance_km is None:

            d = {}

            for u, nbrs in graph.items():

                for v, w in nbrs:

                    d[(u, v)] = min(w, d.get((u, v), float("inf")))

            self.edge_distance_km = d

        else:

            self.edge_distance_km = edge_distance_km





    def shortest_path(self, start: str, goal: str) -> Optional[PathResult]:

        """Distance-optimal path (Dijkstra). Compatibility helper for CLI/tests."""

        return self.dijkstra(start, goal)

    def path_distance_km(self, path: Sequence[str]) -> float:

        dist = 0.0

        for u, v in zip(path, path[1:]):

            w = self.edge_distance_km.get((u, v))

            if w is None:

                return float("inf")

            dist += w

        return dist



    def shortest_path(self, start: str, goal: str, *, per_leg_penalty_km: float = 0.0) -> Optional[PathResult]:

        return self.dijkstra(start, goal, per_leg_penalty_km=per_leg_penalty_km)



    def dijkstra(

        self,

        start: str,

        goal: str,

        *,

        per_leg_penalty_km: float = 0.0,

        banned_nodes: Optional[Set[str]] = None,

        banned_edges: Optional[Set[Tuple[str, str]]] = None,

    ) -> Optional[PathResult]:

        banned_nodes = banned_nodes or set()

        banned_edges = banned_edges or set()



        if start in banned_nodes or goal in banned_nodes:

            return None



        dist = {start: 0.0}

        prev: Dict[str, Optional[str]] = {start: None}

        pq = [(0.0, start)]



        visited: Set[str] = set()



        while pq:

            cur_cost, u = heapq.heappop(pq)

            if u in visited:

                continue

            visited.add(u)



            if u == goal:



                path: List[str] = []

                x: Optional[str] = goal

                while x is not None:

                    path.append(x)

                    x = prev.get(x)

                path.reverse()

                return PathResult(cost=cur_cost, path=path, distance_km=self.path_distance_km(path))



            for v, w in self.graph.get(u, []):

                if v in banned_nodes:

                    continue

                if (u, v) in banned_edges:

                    continue



                new_cost = cur_cost + w + per_leg_penalty_km

                if new_cost < dist.get(v, float("inf")):

                    dist[v] = new_cost

                    prev[v] = u

                    heapq.heappush(pq, (new_cost, v))



        return None



    def k_shortest_paths(

        self,

        start: str,

        goal: str,

        *,

        k: int = 5,

        per_leg_penalty_km: float = 0.0,

    ) -> List[PathResult]:

        """
        Yen's algorithm: K shortest loopless paths. Uses Dijkstra as the subroutine.
        """

        if k <= 0:

            return []



        first = self.dijkstra(start, goal, per_leg_penalty_km=per_leg_penalty_km)

        if first is None:

            return []



        A: List[PathResult] = [first]

        B: List[Tuple[float, List[str]]] = []

        seen: Set[Tuple[str, ...]] = {tuple(first.path)}



        for k_i in range(1, k):

            prev_path = A[-1].path

            for i in range(len(prev_path) - 1):

                spur_node = prev_path[i]

                root_path = prev_path[: i + 1]



                banned_edges: Set[Tuple[str, str]] = set()

                banned_nodes: Set[str] = set(root_path[:-1])





                for p in A:

                    if len(p.path) > i and p.path[: i + 1] == root_path:

                        banned_edges.add((p.path[i], p.path[i + 1]))



                spur = self.dijkstra(

                    spur_node,

                    goal,

                    per_leg_penalty_km=per_leg_penalty_km,

                    banned_nodes=banned_nodes,

                    banned_edges=banned_edges,

                )

                if spur is None:

                    continue



                total_path = root_path[:-1] + spur.path

                t = tuple(total_path)

                if t in seen:

                    continue

                seen.add(t)







                total_cost = self._cost_with_penalty(total_path, per_leg_penalty_km)

                heapq.heappush(B, (total_cost, total_path))



            if not B:

                break



            cost, path = heapq.heappop(B)

            A.append(PathResult(cost=cost, path=path, distance_km=self.path_distance_km(path)))



        return A



    def k_shortest_paths_multi(

        self,

        starts: Iterable[str],

        goals: Iterable[str],

        *,

        k: int = 10,

        per_leg_penalty_km: float = 0.0,

    ) -> List[PathResult]:

        """
        Multi-source/multi-target K-shortest paths by temporarily adding a super-source and super-sink.
        """

        starts = [s for s in starts if s in self.graph]

        goals = [g for g in goals if g in self.graph]

        if not starts or not goals:

            return []



        super_s = "__SUPER_SOURCE__"

        super_t = "__SUPER_SINK__"





        augmented: Dict[str, List[Tuple[str, float]]] = {u: list(nbrs) for u, nbrs in self.graph.items()}

        augmented[super_s] = [(s, 0.0) for s in starts]

        for g in goals:

            augmented.setdefault(g, [])

            augmented[g] = augmented[g] + [(super_t, 0.0)]

        augmented[super_t] = []



        rf = RouteFinder(augmented, edge_distance_km=self._augmented_edge_distance(augmented))

        paths = rf.k_shortest_paths(super_s, super_t, k=k, per_leg_penalty_km=per_leg_penalty_km)



        cleaned: List[PathResult] = []

        for pr in paths:

            p = pr.path

            if len(p) < 3:

                continue



            p2 = p[1:-1]



            cleaned.append(PathResult(cost=pr.cost, path=p2, distance_km=self.path_distance_km(p2)))

        return cleaned



    def shortest_path(self, start: str, goal: str, *, per_leg_penalty_km: float = 0.0) -> Optional[PathResult]:

        """Compatibility wrapper used by the CLI."""

        return self.dijkstra(start, goal, per_leg_penalty_km=per_leg_penalty_km)



    def _cost_with_penalty(self, path: Sequence[str], per_leg_penalty_km: float) -> float:

        cost = 0.0

        for u, v in zip(path, path[1:]):

            w = self.edge_distance_km.get((u, v))

            if w is None:

                return float("inf")

            cost += w + per_leg_penalty_km

        return cost



    def _augmented_edge_distance(self, augmented: Dict[str, List[Tuple[str, float]]]) -> Dict[Tuple[str, str], float]:

        d = dict(self.edge_distance_km)

        for u, nbrs in augmented.items():

            for v, w in nbrs:

                d[(u, v)] = min(w, d.get((u, v), float("inf")))

        return d

