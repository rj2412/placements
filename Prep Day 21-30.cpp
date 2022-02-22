#include<bits/stdc++.h>
using namespace std;
int_fast32_t main()
{
    return 0;
}

// DAY 21 * BINARY SEARCH TREE
//1. floor and ceil in BST
int floorBST(Node* root, int key)
{
    int floor=-1;
    while (root)
    {
        if(root->val==key) 
        {
            floor=key->val;
            return floor;
        }
        if(key > root->val) 
        {
            floor=root->val;
            root=root->right;
        }
        else root=root->right;
    }
    return floor;
    
} //tc-O(log n) sc-O(1)

int findceil(Node* root,int key)
{
    int ceil=-1;
    while(root)
    {
        if(root->data==key) return root->data;
        else if(root->data > key) 
        {
            ceil=root->data;
            root=root->left;
        }
        else root=root->right;
    }
    return ceil;
} //tc-O(log n) sc-O(1)

//2.kth smallest/largest element in a BST
 void solve(Node*root,int k,int &a,int &c)
    {
        if(root==NULL)
        {
            return;
        }
        solve(root->left,k,a,c);
        c++;
        if(c==k)
        {
            a=root->data;
            return;
        }
        solve(root->right,k,a,c);
    }
    int KthSmallestElement(Node *root, int K) {
       
        if(root==NULL)
        {
            return -1;
        }
        int a=-1;
        int c=0;
        solve(root,K,a,c);
        return a;
    } //tc-O(n) sc-O(1)

    //3. two sum IV (two sum in BST)
    class BSTIterator {
    stack<TreeNode *> myStack;
    bool reverse = true; 
public:
    BSTIterator(TreeNode *root, bool isReverse) {
        reverse = isReverse; 
        pushAll(root);
    }

    /** @return whether we have a next smallest number */
    bool hasNext() {
        return !myStack.empty();
    }

    /** @return the next smallest number */
    int next() {
        TreeNode *tmpNode = myStack.top();
        myStack.pop();
        if(!reverse) pushAll(tmpNode->right);
        else pushAll(tmpNode->left);
        return tmpNode->val;
    }

private:
    void pushAll(TreeNode *node) {
        for(;node != NULL; ) {
             myStack.push(node);
             if(reverse == true) {
                 node = node->right; 
             } else {
                 node = node->left; 
             }
        }
    }
};
class Solution {
public:
    bool findTarget(TreeNode* root, int k) {
        if(!root) return false; 
        BSTIterator l(root, false); 
        BSTIterator r(root, true); 
        
        int i = l.next(); 
        int j = r.next(); 
        while(i<j) {
            if(i + j == k) return true; 
            else if(i + j < k) i = l.next(); 
            else j = r.next(); 
        }
        return false; 
    }
}; //tc-O(n) sc-O(h)*2

//4. BST iterator
 
 class BSTIterator {
    stack<TreeNode *> myStack;
public:
    BSTIterator(TreeNode *root) {
        pushAll(root);
    }

    /** @return whether we have a next smallest number */
    bool hasNext() {
        return !myStack.empty();
    }

    /** @return the next smallest number */
    int next() {
        TreeNode *tmpNode = myStack.top();
        myStack.pop();
        pushAll(tmpNode->right);
        return tmpNode->val;
    }

private:
    void pushAll(TreeNode *node) {
        for (; node != NULL; myStack.push(node), node = node->left);
    }
}; //tc-O(1) sc-O(h)

 //5. Largest BST in a BT
 class NodeValue {
public:
    int maxNode, minNode, maxSize;
    
    NodeValue(int minNode, int maxNode, int maxSize) {
        this->maxNode = maxNode;
        this->minNode = minNode;
        this->maxSize = maxSize;
    }
};

class Solution {
private:
    NodeValue largestBSTSubtreeHelper(TreeNode* root) {
        // An empty tree is a BST of size 0.
        if (!root) {
            return NodeValue(INT_MAX, INT_MIN, 0);
        }
        
        // Get values from left and right subtree of current tree.
        auto left = largestBSTSubtreeHelper(root->left);
        auto right = largestBSTSubtreeHelper(root->right);
        
        // Current node is greater than max in left AND smaller than min in right, it is a BST.
        if (left.maxNode < root->val && root->val < right.minNode) {
            // It is a BST.
            return NodeValue(min(root->val, left.minNode), max(root->val, right.maxNode), 
                            left.maxSize + right.maxSize + 1);
        }
        
        // Otherwise, return [-inf, inf] so that parent can't be valid BST
        return NodeValue(INT_MIN, INT_MAX, max(left.maxSize, right.maxSize));
    }
    public:
    int largestBSTSubtree(TreeNode* root) {
        return largestBSTSubtreeHelper(root).maxSize;
    } //TC-O(n) sc-O(1)


}; 

 //6.serialize and deserialize binary tree
  string serialize(TreeNode* root) {
        if(!root) return "";
        
        string s ="";
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()) {
           TreeNode* curNode = q.front();
           q.pop();
           if(curNode==NULL) s.append("#,");
           else s.append(to_string(curNode->val)+',');
           if(curNode != NULL){
               q.push(curNode->left);
               q.push(curNode->right);            
           }
        }
        return s;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        if(data.size() == 0) return NULL; 
        stringstream s(data);
        string str; 
        getline(s, str, ',');
        TreeNode *root = new TreeNode(stoi(str));
        queue<TreeNode*> q; 
        q.push(root); 
        while(!q.empty()) {
            
            TreeNode *node = q.front(); 
            q.pop(); 
            
            getline(s, str, ',');
            if(str == "#") {
                node->left = NULL; 
            }
            else {
                TreeNode* leftNode = new TreeNode(stoi(str)); 
                node->left = leftNode; 
                q.push(leftNode); 
            }
            
            getline(s, str, ',');
            if(str == "#") {
                node->right = NULL; 
            }
            else {
                TreeNode* rightNode = new TreeNode(stoi(str)); 
                node->right = rightNode;
                q.push(rightNode); 
            }
        }
        return root; 
    } //tc-O(n) sc-O(n)

    //DAY 22 
    // * MIXED QUESTIONS *
    //1. Find Kth largest in stream of numbers
    priority_queue<int,vector<int>,greater<int>>q;
    int solve(int num,int k)
    {
        if(q.size()<k) 
        {
            q.push(num);
            return q.size()==k ? q.peek() : -1;
        }
        if(num>q.peek())
        {
            q.pop();
            q.push(num);

        }
        return q.peek();
    }
    int kth_ele_instream(vector<int>&nums,int k)
    {
        for(int i=0;i<nums.size();i++)
        {
            int ans=solve(nums[i],k);
        }
        return ans;

    } //tc-O(klogk)+ (n-k)(logk)

    //2. count distinct element in window of size k
    vector <int> countDistinct (int A[], int n, int k)
    {
        
        unordered_map<int,int>mp;
        vector<int>ans;
        for(int i=0;i<k;i++)
        {
            mp[A[i]]++;
        }
        ans.push_back(mp.size());
        for(int i=1;i<=n-k;i++)
        {
            mp[A[i-1]]--;
            if(mp[A[i-1]]==0) mp.erase(A[i-1]);
            mp[A[i+k-1]]++;
            ans.push_back(mp.size());
        }
        return ans;
    } // tc-O(n) sc-O(n)

    //3. kth largest in an array
    int partition(vector<int>& nums,int l,int h)
    {
        int pivot=l;
        int i=l,j=h;
        while(i<j)
        {
           while(i<=j && nums[i]<=nums[pivot])
               i++;
           while(i<=j && nums[j]>nums[pivot])
               j--;
            
            if(i>j)
            break;
            
            swap(nums[i],nums[j]);
        }
        
        swap(nums[pivot],nums[j]);
        return j;
    }
    
    int findKthLargest(vector<int>& nums, int k) {
        k=nums.size()-k;
        int l=0,h=nums.size()-1;
        int res;
        while(l<=h)
        {
            res=partition(nums,l,h);
            if(res==k)
                return nums[res];
            else if(res>k)
                h=res-1;
            else
                l=res+1;
        }
        return -1;
    } //tc-O(n) avg sc-O(1)

    //4. flood fill algo
     int dx[4]={0,-1,0,1};
    int dy[4]={1,0,-1,0};
    int color=-1;
         
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
        
       
       if(color==-1)  color=image[sr][sc];
        int n=image.size();
        int m=image[0].size();
        
        if(sr==-1|| sc==-1 || sr==n || sc==m||image[sr][sc]!=color || color==newColor||image.empty())
            return image;
        image[sr][sc]=newColor;
        for(int p=0;p<4;p++)
        {
            floodFill(image,sr+dx[p],sc+dy[p],newColor);
        }
     
        
        return image;
    } // tc-O(m*n) sc-O(1)

    //DAY 23
    // *GRAPH *
    //1. CLONE  a graph
    class Node{
        int val;
        vector<Node*>neighbors;
    }
    unordered_map<Node*,Node*>mp;
    Node* clonegraph(Node* root)
    {
        if(!root) return NULL;
        if(mp.find(root)==mp.end())
        {
            mp[root]=new Node(root->val,{});
            for(auto x: neighbors)
            {
                mp[root]->neighbors.push_back(clonegraph((x)));
            }
        }
        return mp[root];
    } // tc-O(E*N) sc-O(N) n- vertices e- edges

    //2. DFS
    void dfsrec(vector<int>adj[],bitset<n>&vis,int u)
    {
        vis[i]=1;
        cout<<u<<" ";
        for(int i=0;i<adj[u].size();i++)
        {
            if(vis(adj[u][i]==0)) dfsrec(adj,vis,adj[u][i]);
        }
    }
     
    void dfs(vector<int>adj[],int n)
    {
        bitset<n> vis; // vis array with all 0 intitially
        for(int i=0;i<n;i++)
        {
            if(vis[i]!=1)
            {
                dfsrec(adj,vis,i);
            }
        }

    } //tc-O(n+e) sc-O(n+e)+O(n)+O(n)

    //3. BFS
    vector<int>bfs(vector<int>adj[],int V)
    {
        vector<int>bfs;
        vector<int>vis(V+1,0)
        for(int i=1;i<=V;i++)
        {
            if(!vis[i])
            {
                queue<int>q;
                q.push(i);
                vis[i]=1;
                while(!q.empty())
                {
                    int node=q.front();
                    q.pop();
                    bfs.push_back(node);
                    for(auto x : adj[node])
                    {
                        if(!vis[x]) 
                        {
                            vis[x]=1;
                            q.push(x);
                        }
                    }

                }

            }


        }
        return bfs;
    } // tc-O(n+e) sc-O(n) + O(n)

    //4. Cycle detection 

      //* in undirected graph*

      bool checkcycledfs(vector<int>adj[],vector<int>&vis,int node,int parent)
      {
          vis[node]=1;
          for(auto it: adj[node])
          {
              if(vis[it]==0)
               {
                   if(checkcycledfs(adj,vis,it,node)) return true;
               }
               else if(it!=parent) return true;
          }
          return false;

      }
      bool iscycle(vector<int>adj[],int V)
      {
          vector<int>vis(V+1,0);
          for(int i=1;i<=V;i++)
          {
              if(!vis[i]) 
              {
               if(checkcycledfs(adj,vis,i,-1)) return true;

              }
          }
          return false;
      } //tc-O(n+e) sc-O(n)

      //* directed graph*
      bool check(int node,vector<int>adj[],vector<int>&vis,vector<int>&dfsvis)
      {
          vis[node]=1;
          dfsvis[node]=1;
          for(auto it: adj[node])
          {
              if(!vis[it])
              {
                  if(check(it,adj,vis,dfsvis)) return true;
              }
              else if(dfsvis[it]) return true;
          }
          dfsvis[node]=0;
          return false;
      }

      bool iscyclic(vector<int>adj[],int V)
      {
        vector<int>vis(V,0);
        vector<int>dfsvis(V,0);
        for(int i=0;i<V;i++)
        {
            if(!vis[i])
            {
                if(check(i,adj,vis,dfsvis)) return true;
            }
        }
        return false;

      } // tc-O(n+e) sc-O(n)+O(n)

      //5. Topo sort
      // using dfs

      void dfs(vector<int>&vis,vector<int>adj[],int node,stack<int>&st)
      {
          vis[node]=1;
          for(auto x:adj[node])
          {
              if(!vis[x]) dfs(vis,adj,x,st);
          }
          st.push(node);
      }
      vector<int>toposort(vector<int>adj[],int N)
      {
          vector<int>vis(N,0);
          stack<int>st;
          for(int i=0;i<N;i++)
          {
              if(!vis[i]) dfs(vis,adj,i,st);
          }
          vector<int>topo;
          while(!st.empty())
          {
              int p=st.top();
              st.pop();
              topo.push_back((p));
          }
          return topo;
      } // tc-O(N+E) sc-O(n)+O(n)

      //6. find no of islands
        int dx[8]={-1, -1, -1, 0, 1, 0, 1, 1};
  int dy[8]={-1, 1, 0, -1, -1, 1, 0, 1};
  
  
  void bfs(int i,int j,vector<vector<char>>&grid,vector<vector<bool>> &vis,int n,int m)
  { 
      queue<pair<int,int>> q;
  q.push({i,j});
  while(!q.empty()){
      
      int x=q.front().first;
      int y=q.front().second;
      q.pop();
  
      for(int p=0;p<8;p++)
      {
          int nx=x+dx[p];
          int ny=y+dy[p];
          if(nx>=0 && nx<n && ny>=0 && ny<m && grid[nx][ny]=='1' && vis[nx][ny]==false)
          {
          vis[nx][ny]=true;
          q.push({nx,ny});
          }
      }
  }
  }
    // Function to find the number of islands.
    int numIslands(vector<vector<char>>& grid) {
        // Code here
        int n=grid.size();
        int m=grid[0].size();
       vector<vector<bool>> vis(n,vector<bool>(m,false));
        int ans=0;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<m;j++)
            {
                if(grid[i][j]=='1' && vis[i][j]==false)
                {
                    bfs(i,j,grid,vis,n,m);
                    ans++;
                }
            }
        }
        return ans;
        
    } // tc-O(n*m) sc-O(n*m)

    //7. bipartite check
    // works for both connected as well as unconnected graph
    bool checkifbi(vector<int>adj[],int N)
    {
        vector<int>col(N,-1);
        queue<pair<int,int>>q; // for vertex,color
         //loop incase graph is not connected
    for (int i = 0; i < V; i++) {
       
      //if not coloured
        if (col[i] == -1) {
           
          //colouring with 0 
            q.push({ i, 0 });
            col[i] = 0;
           
            while (!q.empty()) {
                pair<int, int> p = q.front();
                q.pop();
               
                  //current vertex
                int v = p.first;
                  //colour of current vertex
                int c = p.second;
                 
                  //traversing vertexes connected to current vertex
                for (int j : adj[v]) {
                   
                      //if already coloured with parent vertex color
                      //then bipartite graph is not possible
                    if (col[j] == c)
                        return 0;
                   
                      //if uncooloured
                    if (col[j] == -1) {
                      //colouring with opposite color to that of parent
                        col[j] = (c) ? 0 : 1;
                        q.push({ j, col[j] });
                    }
                }
            }
        }
    }
   
    return true;
    }
    bool ifbigraph(vector<int>adj[],int N)
    {
        bool ans=checkifbi(adj,N);
        return ans;
    } // tc-O(n+e) sc-O(n)+O(n)
    

    //DAY 24 
    // *GRAPH*

    //1. Kosaraju algo for SCC
    
void dfs(int node,stack<int>&st, vector<int> &vis, vector<int>adj[]){
    vis[node]=1;
    for(auto it: adj[node]){
        if(!vis[it]){
            dfs(it,st,vis,adj);
        }
    }
    st.push(node);
    
}

 void revDFS(int node,vector<int>&vis,vector<int> transpose[]){
     cout<<node<<" ";
     vis[node]=1;
     for(auto it : transpose[node]){
         if(!vis[it]){
             revDFS(it,vis,transpose);
         }
     }
     
 }

int main() {
	int n,m;
	cin>>n>>m;
	vector<int> adj[n];
	for(int i=0;i<m;i++){
	    int u,v;
	    cin>>u>>v;
	    adj[u].push_back(v);
	    
	}
	stack<int> st;
	vector<int> vis(n,0);
	for(int i=0;i <n;i++){
	    if(!vis[i]){
	        dfs(i,st,vis,adj);
	    }
	}
	
	vector<int>transpose[n];
	for(int i=0;i<n;i++){
	    vis[i]=0;
	    for(auto it : adj[i]){
	        transpose[it].push_back(i);
	        
	    }
	}
	
	while(!st.empty()){
	    int node=st.top();
	    st.pop();
	    if(!vis[node]){
	        cout<<" SCC :";
	        revDFS(node,vis,transpose);
	        cout<<endl;
	        
	    }
	}
	return 0;
} // tc-2*O(n+e)+O(n) sc-O(n+e)+O(n)

//2. Djikstra's algo for shortest distance bw source and nodes
vector<int>Djikstra(vector<pair<int,int>>g[n+1],int source){
	
	
	
	//dijdktra algo begin here
	priority_queue<pair<int,int>,vector<pair<int,int>>, graeter<pair<int,int>>>pq;// min heap in pair(dist,node)
	vector<int>dist(n=1,INT_MAX);// array to store distance
	dist[source]=0;
	pq.push(make_pair(0,source));
	
	while(!pq.empty()){
	    int dis = pq.top().first;
	    int prev= pq.top().second;
	    pq.pop();
	    
	    vector<pair<int,int>> :: iterator it;
	    for(it=g[prev].begin();it!=g[prev].end();it++){
	        int next= it->first;
	        int nextDist=it->second;
	        if(dist[next]>dist[prev]+nextDist){
	            dist[next]= dist[prev]+ nextDist;
	            pq.push(make_pair(dist[next],next));
	        }
	    }
	    
	}
	return dist;	
	
} //tc-O((n+e)logn) sc-O(n)+o(n)

//3. Bellman ford algo for detection of negative cycles
	int isNegativeWeightCycle(int n, vector<vector<int>>adj){
	    
	    vector<int>weight(n,INT_MAX);
    weight[0]=0;
    for(int k=0;k<n;k++){
        for(auto x:adj){
            int u=x[0];
            int v=x[1];
            int w=x[2];
            if(weight[u]!=INT_MAX && weight[u]+w<weight[v]){
                weight[v]=weight[u]+w;
            }
        }
    }
    for(auto x:adj){
        int u=x[0];
        int v=x[1];
        int w=x[2];
        if(weight[u]!=INT_MAX && weight[u]+w<weight[v]){
            return 1;
        }
    }
    return 0;
	}//tc-O(n+e) sc-O(n)

    //4. floyd warshall algo for shortest path bw all pairs of nodes
    #define V 6		//No of vertices

  void floyd_warshall(int graph[V][V])
   {
	int dist[V][V];

	//Assign all values of graph to allPairs_SP
	for(int i=0;i<V;++i)
		for(int j=0;j<V;++j)
			dist[i][j] = graph[i][j];

	//Find all pairs shortest path by trying all possible paths
	for(int k=0;k<V;++k)	//Try all intermediate nodes
		for(int i=0;i<V;++i)	//Try for all possible starting position
			for(int j=0;j<V;++j)	//Try for all possible ending position
			{
				if(dist[i][k]==INT_MAX || dist[k][j]==INT_MAX)	//SKIP if K is unreachable from i or j is unreachable from k
					continue;
				else if(dist[i][k]+dist[k][j] < dist[i][j])		//Check if new distance is shorter via vertex K
					dist[i][j] = dist[i][k] + dist[k][j];
			}

	//Check for negative edge weight cycle
	for(int i=0;i<V;++i)
		if(dist[i][i] < 0)
		{
			cout<<"Negative edge weight cycle is present\n";
			return;
		}

	//Print Shortest Path Graph
	//(Values printed as INT_MAX defines there is no path)
	for(int i=1;i<V;++i)
	{
		for(int j=0;j<V;++j)
			cout<<i<<" to "<<j<<" distance is "<<dist[i][j]<<"\n";
		cout<<"=================================\n";
	}
    }

   void solve()
  {
	int graph[V][V] = { {0, 1, 4, INT_MAX, INT_MAX, INT_MAX},
						{INT_MAX, 0, 4, 2, 7, INT_MAX},
						{INT_MAX, INT_MAX, 0, 3, 4, INT_MAX},
						{INT_MAX, INT_MAX, INT_MAX, 0, INT_MAX, 4},
						{INT_MAX, INT_MAX, INT_MAX, 3, 0, INT_MAX},
						{INT_MAX, INT_MAX, INT_MAX, INT_MAX, 5, 0} };

	floyd_warshall(graph);
	
   } //tc-O(V^3) sc-O(V^2)

   //5. Prims algo for MST
   void prims-algo(vector<pair<int,int>>adj[],int n)
   {
       // adj is like (a->b weight) 
       vector<int>key(n,INT_MAX);
       vector<int>par(n,-1);
       vector<int>MST(n,false);
       priority_queue<pair<int,int>,vector<pair<int,int>>,graeter<pair<int,int>>>pq;

       key[0]=0;
       par[0]=-1;
       pq.push({0,0}); // key[i] - index
       for(int ct=0;ct<n;ct++)
       {
           int u=pq.top().second;
           pq.pop();
           mst[u]=true;
           for(auto it:adj[u])
           {
               int v=it.first;
               int wt=it.second;
               if(mst[v]==false && wt<key[v])
               {
                   par[v]=u;
                   pq.push({wt,v});
               }
           }
       }
       // par[] will have our MST 
   } // brute- tc_O(n*n) optimal tc-O(nlogn) sc-O(n)

   //6. kruskals algo using disjoint set DS
   struct node{
       int u,v, wt;
       node(int first,int second,int weight)
       {
           u=first,v=second,wt=weight;
       }
   };
   bool comp(node a ,node b)
   {
       return a.wt<b.wt;
   }
   int find_par(int u,vector<int>&parent)
   {
       if(u==parent[u]) return u;

       return (find_par(parent[u],parent));

   }
   void union(int u,int v,vector<int>&parent,vector<int>&rank)
   {
       u=find_par(u,parent);
       v=find_par(v,parent);
       if(rank[u]<rank[v]) parent[u]=v;
       else if(rank[u]>rank[v]) parent[v]=u;
       else {
           parent[v]=u;
           rank[u]++;
       }
   }
   int kruskals_algo(vector<node>edges,int n,int m) // n vertex , m edges 
   {
       sort(edges.begin(),edges.end(),comp);
       vector<int>parent(n);
       for(int i=0;i<n;i++) parent[i]=i;
       vector<int>rank(n,0);
       int cost=0;
       vector<pair<int,int>>mst; // storing edges
       for(auto it : edges)
       {
           if(find_par(it.v,parent!=find_par(it.u,parent)))
           {
               cost+=it.wt;
               mst.push_back({it.u,it.v});
               union(it.u,it.v,parent,rank);
           }
       }
       cout<<cost<<endl;
       for(auto it: mst)
       {
           cout<<it.first<<"-"<<it.second<<endl;
       }
       return cost;

   } // tc-O(mlogm) + O(m*const) ..for sorting m edges and using disjoint set DS for m edges
   //sc-O(m)+O(n)+o(n)

   //DAY 25  *Dynamic Programming*

   //1. maxm product subarray
   int maxproductsubarray(vector<int>&nums)
   {
       int ans=nums[0];
       int tmax=nums[0];
       int tmin=nums[0];
       int n=nums.size();
       for(int i=1;i<n;i++)
       {
           if(nums[i]<0) swap(tmax,tmin);
           tmax=max(nums[i],nums[i]*tmax);
           tmin=min(nums[i],tmin*nums[i]);
           ans=max(ans,tmax);
       }
       return ans;
   }
   //tc-O(n) sc-O(1)

   //2. Longest increasing  subsequece
   int LIS(vector<int>arr,int n)
   {
       vector<int>lis(n,1);
       for(int i=1;i<n;i++)
       {
           lis[i]=1;
           for(int j=0;j<i;j++)
           {
               if(arr[i]>arr[j] && lis[i]<=lis[j]) lis[i]=1+lis[j];
           }
       }
       return *max_element(lis,lis+n);
   } // tc-O(n) sc-O(n)

   //3. Longest common subsequence
   int t[m+1][n+1];
   int lcs(string s,string y,int m,int n)
   {
       for(int i=0;i<m;i++)
       {
           for(int j=0;j<n;j++)
           {
               if(i==0 || j==0) t[i][j]=0;
           }
       }
       for(int i=1;i<=m;i++)
       {
           for(int j=1;j<=n;j++)
           {
               if(x[i-1]==y[j-1]) t[i][j]=1+t[i-1][j-1];
               else t[i][j]=max(t[i-1][j],t[i][j-1]);
           }

       }
       return t[m][n];
   } //tc-O(m*n) sc-O(m*n)

   //4.0-1 knapsack
   int 0-1_knapsacak(vector<int>&wt,vctor<int>&val,int w,int n)
   {
       int t[n+1][w+1];
       for(int i=0;i<=n;i++)
       {
           for(int j=0;j<=w;j++)
           {
               if(i==0||j==0) t[i][j]=0;
           }
       }

       for(int i=1;i<=n;i++)
       {
           for(int j=1;j<=w;j++)
           {
               if(wt[i-1]<=j)
               { 
                   t[i][j]=max(val[i-1]+t[i-1][j-wt[i-1]], t[i-1][j]);

               }
               else t[i][j]=t[i-1][j];
           }
       }
       return t[n][w];
   } //tc-(n*w) sc-O(n*w)



   //5. edit distance
   int minDistance(string word1, string word2) {
        int m=word1.length(),n=word2.length();
        vector<vector<int>>t(m+1,vector<int>(n+1));
        for(int i=1;i<=m;i++) t[i][0]=i;
        for(int j=1;j<=n;j++) t[0][j]=j;
         for(int i=1;i<=m;i++)
         {
             for(int j=1;j<=n;j++)
             {
                 if(word1[i-1]==word2[j-1]) t[i][j]=t[i-1][j-1];
                 else t[i][j]=1+min({t[i-1][j-1],t[i][j-1],t[i-1][j]});
             }
         }
        return t[m][n];
        
    } // tc-O(m*n) sc-O(m*n)

    //6. Maximum  sum increasing subsequence (MSIS)
    int MSIS(vector<int>&arr,int n)
    {
        int msis[n];
        for(int i=0;i<n;i++) msis[i]=arr[i];
        for(int i=1;i<n;i++)
        {
            for(int j=0;j<i;j++)
            {
                if(arr[i]>arr[j] && msis[i]<arr[i]+msis[j]) msis[i]=arr[i]+msis[j];
            }
        }
        int mx=0;
        for(int i=0;i<n;i++) mx=max(msis[i],mx);
        return mx;
    } //tc-O(n*n) sc-O(n)

    //7. matric chain multiplication
    int t[101][101];
    int solve(int arr[],int i,int j)
    {
        if(i>=j) return 0;
        if(t[i][j]!=-1) return t[i][j];
        int res=INT_MAX:
        for(int k=i;k<j;k++)
        {
            int tempans=solve(arr,i,k)+solve(arr,k+1,j) + arr[i-1]*arr[k]*arr[j];
            res=min(res,tempans);

        }
        return res;

    }
    int matrixchainMultiplication(int arr[],int n)
    {
        memset(t,-1,sizeof(t));
        return solve(arr,1,n-1);
    } //tc-O(n*n*n) sc-O(n*n)

    // DAY 26 *DP*

    //1. maximum path usm in matrix
    int maxpathsum(int m[][],int r.int c)
    {
        for(int i=1;i<r;i++)
        {
            for(int j=0;j<c;j++)
            {
                if(j>0 && j<c-1) m[i][j]+=max({m[i-1][j],m[i-1][j-1],m[i-1][j+1]});
                else if(j>0) m[i][j]+=max(m[i-1][j],m[i-1][j-1]);
                else if(j<c-1) m[i][j]+=max(m[i-1][j],m[i-1][j+1]);
            }
        }
        int ans=0;
        for(int i=0;i<c;i++)
        ans=max(ans,m[r-1][i]);
        return ans;
    } //tc-O(m*n) sc-O(1)
    

    //2.coin change I - find maxm no of ways to have sum S if we have unlimited suplly of given coins
    long long coinchange1(int C[],int m,int S)
    {
        vector<int>t(S+1);
        t[0]=1;
        for(int i=0;i<m;i++)
        {
            for(int j=C[i];j<=S;j++)
            {
                t[j]+=t[j-C[i]];
            }
        }
        return t[n];
    } //tc-O(m*S) sc_O(S)

    //3.subset sum
    bool t[n+1][sum+1];
    bool subsetsum(vector<int>arr,int n,int sum)
    {
        for(int i=0;i<=n;i++)
        {
            for(int j=0;j<=sum;j++) 
            {
                if(i==0) t[i][j]=false;
                if(j==0) t[i][j]=true;
            }
        }
        for(int i=1;i<=n;i++)
        {
            for(int j=01j<=sum;j++)
            {
                if(arr[i-1]<=j) t[i][j]=t[i-1][j-arr[i-1]] || t[i-1][j];
                else t[i][j]=t[i-1][j];
            }
        }
        return t[n][sum];
    } // tc-O(n*sum) sc-O(n*sum)

    //4.Rod cutting problem (unbounded knapsack)
       int cutRod(int price[], int n) {
       
        int len[n];
        for(int i=0;i<n;i++) len[i]=i+1;
        int t[n+1][n+1];
        for(int i=0;i<=n;i++)
        {
            for(int j=0;j<=n;j++)
            {
                if(i==0) t[i][j]=0;
                else if(j==0) t[i][j]=0;
            }
        }
        for(int i=1;i<=n;i++)
        {
            for(int j=1;j<=n;j++)
            {
                if(len[i-1]<=j)
                  t[i][j]=max(price[i-1]+t[i][j-len[i-1]],t[i-1][j]);
                  else t[i][j]=t[i-1][j];
            }
        }
        return t[n][n];
    } //tc-O(n*n) sc-O(n*n)

    //5.egg dropping prblm
      int t[201][201];
    int eggDrop1(int e, int f) 
    {
       
        if(f==0||f==1) return f;
        if(e==1) return f;
        if(t[e][f]!=-1) return t[e][f];
        int ans=INT_MAX;
        for(int k=1;k<=f;k++)
        {
            int temp=1+max(eggDrop1(e-1,k-1),eggDrop1(e,f-k));
            ans=min(ans,temp);
        }
        return t[e][f]=ans;
    }
    int eggDrop(int e,int f)
    {
        memset(t,-1,sizeof(t));
        return eggDrop1(e,f);
    } //tc-O(e*(f^2)) osc-O(e*f)

    //6. word break problem
        bool wordBreak(string s, vector<string>& wordDict) {
        int n=s.length();
        unordered_set<string>wrdset(wordDict.begin(),wordDict.end());
        vector<bool>t(n+1);
        t[0]=true;
        for(int i=1;i<=n;i++)
        {
            for(int j=0;j<i;j++)
            {
                if(t[j]&& wrdset.count(s.substr(j,i-j))) {
                    t[i]=true;
                    break;
                }
            }
        }
        return t[n];
      
    } // tc-O(n*n) sc-O(n)

    //7.palindrome partitioning
    // bottom up DP
    int t[1001][1001];
    int solve(string s,int i,int j)
    {
        if(i>=j) return 0;
        if(ispalindrome(s,i,j)==true) return 0;
        if(t[i][j]!=-1) return t[i][j];
        int ans=INT_MAX;
        for(int k=i;k<=j-1;k++) 
        {
            int temp=1+solve(s,i,k)+solve(s,k+1,j);
            ans=min(temp,ans);
        }
        return t[i][j]=ans;
    }
    int palindromepartitioning(string s)
    {
        memset(t,-1,sizeof(t));
        int ans=solve(s,0,n-1);
        return ans;
    } //tc-O(n*n) sc-O(n)

    //8. maximum profit in job scheduling
   static bool comp(vector<int>a,vector<int>b)
{
    return a[1]<b[1];
}
    int jobScheduling(vector<int>& startTime, vector<int>& endTime, vector<int>& profit) {
        int n=endTime.size();
        vector<vector<int>>v(n);
        for(int i=0;i<n;i++)
        {
            int x=startTime[i];
            int y=endTime[i];
            int p=profit[i];
            v[i]={x,y,p};
        }
        sort(v.begin(),v.end(),comp);
        int t[n];
        t[0]=v[0][2];
        for(int i=1;i<n;i++)
        {
            int inc=v[i][2];
            int last=-1;
            int low=0;
            int high=i-1;
            while(low<=high)
            {
                int mid=(low+high)/2;
                if(v[mid][1]<=v[i][0])
                {
                    last =mid;
                    low=mid+1;
                }
                else high=mid-1;
            }
            if(last!=-1) inc+=t[last];
            int exc=t[i-1];
            t[i]=max(exc,inc);
        }
        return t[n-1];
    } //tc-O(nlogn) sc-O(n)

    //* End of placement series*
