#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

void parallelBFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;
    vector<int> traversalOrder; // To store order of traversal
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int sz = q.size();
        vector<int> currentLevel;

        #pragma omp parallel for shared(q, visited)
        for (int i = 0; i < sz; ++i) {
            int u;
            #pragma omp critical
            {
                if (!q.empty()) {
                    u = q.front(); q.pop();
                    cout << "Visited: " << u << endl;
                    traversalOrder.push_back(u); // Record visit
                }
            }

            for (int v : graph[u]) {
                #pragma omp critical
                {
                    if (!visited[v]) {
                        visited[v] = true;
                        currentLevel.push_back(v);
                    }
                }
            }
        }

        for (int v : currentLevel) q.push(v);
    }

    // Print traversal order at the end
    cout << "Traversal Order: ";
    for (int node : traversalOrder) {
        cout << node << " ";
    }
    cout << endl;
}

int main() {
    int n, e;
    cout << "Enter number of nodes and edges: ";
    cin >> n >> e;
    vector<vector<int>> graph(n);
    cout << "Enter edges (u v):\n";
    for (int i = 0; i < e; ++i) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    int start;
    cout << "Enter starting node: ";
    cin >> start;
    parallelBFS(graph, start);
    return 0;
}
