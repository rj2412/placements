// DAY-11
// BINARY SEARCH
//1. Nth root of no using binary search ,m is no

double multiply(double num,int n)
{
    double ans=1.0;
    for(int i=1;i<=n;i++)
    {
        ans=ans*num;
    }
    return ans;
}
double getNthroot(int n, int m)
{
    double low=1;
    double high=m;
    double eps=1e-6;
    while(high-low>eps)
    {
        double mid=(high - low)/2.0;
        if(muitiply(mid,n)<m) low=mid;
        else high=mid;
    }
    return pow(m,(double)(1.0/(double)n));

} //  tc -O(n*logn) sc-O(1)

//2. median of row wise sorted matrix


int findmedian(vector<vector<int>>&mat)
{
    int n=mat.size();
    int m=mat[0].size;
    int low=1,high=1e9;
    while(low<=high)
    [
        int mid=low+high/2;
        int cnt=0;// counting how many elements are <=mid
        for(int i=0;i<n;i++)
        {
            int l=0,r=mat[i].size()-1;
            while(l<=r)
            {
                int md=(l+r)/2;
                if(mat[i][md]<=mid) l=md+1;
                else r=md-1;

            }
            cnt+=l;
        }
        if(cnt >= (n*m)/2) high=mid-1;
        else low=mid+1;
    ]
    return low;
} // tc-O(32 * n* logm) sc-O(1)

//3.single element in sorted array
  int singleNonDuplicate(vector<int>& nums) {
        int l=0,h=nums.size()-2;
        while(l<=h) 
        {
            int mid=(l+h)/2;
            if(nums[mid]==nums[mid^1]) l=mid+1;
            else h=mid-1;
        }
        return nums[l];
    } // tc-O(logn) sc-O(1)

 //4. search in rotated array
 int search(vector<int>nums,int target)
 {
     int l=0,r=nums.size()-1;
     while(l<=r)
     {
         int m=(l+r)/2;
         if(nums[m]==target) return m;
         if(nums[l]<=nums[m]) // if left half sorted
         {
             if(target>nums[l]&& target<=nums[r]) r=mid-1;
             else l=mid+1; 
         }
         else //r8 half sorted
         {
            if(target>=nums[m] && target<=nums[r]) l=m+1;
            else r=m-1;
         }
     }
     return -1;
 } // tc-O(log n) sc-O(1)

 //5. median of two sorted array
 double findmed(vector<int>nums1,vector<int>nums2) //taking nums1 as smaller array

 {
    int n1=nums1.size();
    int n2=nums2.size();
    int l=0,r=n1;
    while(l<=r)
    {
        int cut1=(l+r)/2;
        int cut2=(n1+n2+1)/2- cut1;
        int l1=cut1==0 ? INT_MIN : nums1[cut1-1];
        int l2=cut2==0 ? INT_MIN : nums2[cut2-1];
        int r1=cut1==n1 ? INT_MAX : nums1[cut1];
        int r2=cut2==n2 ? INT_MAX : nums2[cut2];
        if(l1<=r2 && l2<=r1)
        {
            if((n1+n2)%2==0) return max(l1,l2)+min(r1,r2)/2;
            else return max(l1,l2);
        }
        else if(l1>r2) r=cut1-1;
        else l=cut1+1;
    }
    return 0.0;
 } // tc-O(log(min(n1,n2))) sc-O(1)

 // 6. kth element of two sorted arrays
int kthelement(int array1[],int array2[],int m,int n,int k) {
    int p1=0,p2=0,counter=0,answer=0;
    
    while(p1<m && p2<n) {
        if(counter == k) break;
        else if(array1[p1]<array2[p2]) {
            answer = array1[p1];
            ++p1;
        }
        else {
            answer = array2[p2];
            ++p2;
        }
        ++counter;
    }
    if(counter != k) {
        if(p1 != m-1) 
            answer = array1[k-counter];
        else 
            answer = array2[k-counter];
    }
    return answer;
} // tc-O(log(min(n1,n2))) sc-O(1)

 //7. allocate min pages
 bool valid(int a[],int n,int m,int barrier)
 {
     int alot_stu=1,pages=0;
     for(int i=0;i<n;i++)
     {
         if(a[i]>barrier) return false;
         if(pages + a[i]> barrier) alot_stu+=1,pages+=a[i];
         else pages+=a[i];
     }
     if(alot_stu > m) return false;
     return true;
 }
  int findPages(int a[], int n, int m) 
    {
        //code here
        int l,h,mid;
        int sum=0,mx=0;
        for(int i=0;i<n;i++){
            sum+=a[i];
            mx=max(mx,a[i]);
        }
        l=mx,h=sum;
        int res=0;
        while(l<=h){
            mid=(l+h)/2;
            if(valid(a,n,m,mid))
            {
                res=mid;
                h=mid-1;
            }
            else l=mid+1;
        }
        return res;
    } // tc-O(n*logn) sc-O(1)

    //8.Agressive cows
    bool isplacecows(vector<int>&arr,int n,int cows,int dist)
    {
        int cnt=1,cord=arr[0];
        for(int i=1;i<n;i++)
        {
           if(arr[i]-cord >=dist)
           {
               cnt++;
               cord=arr[i];
           }
           if(cnt==cows) return true;
        }
        return false;
    }
    int mindist(vector<int>arr,int cows)
    {
        int l=arr[0];
        int n=arr.size();
        int h=arr[n-1]-arr[0];
        while(l<=h)
        {
            int mid=(l+h)>>1;
            if(isplacecows(arr,n,cows,mid)) 
            {
                res=mid;
                l=mid+1;
            }
            else 
            {
                h=mid-1;
            }
        }
        return h;
    } // tc -O(n*logn) sc-O(1)

    // DAY 12
    // BITS
    //1. power set
    vector<string>powerset(string s) // generating all subsets of an array or string using bits
    {
        int n=s.length();
        vector<string>ans;
        for(int i=0;i<(1<<n-1);i++)
        {
            string temp="";
            for(int j=0;j<n;j++)
            {
                if(i && (j<<i)) // jth bit is set/1 ..means picking up that bit
                 temp+=s[j];
            }
            ans.push_back(temp);

        }
        return ans;
    } // tc-O(2^n * n) sc-O(1)

    // DAY 13
    // STACKS AND QUEUES

    //1. implement stack using arrays
    int arr[3];
     int top=-1
     void push(int x)
     {
         arr[++top]=x;
     }

     int pop()
     {
         top--;
         return arr[top];
     }

     int top()
     {
         return arr[top];

     }

     int size()
     {
         return top+1;
     }

     bool isempty()
     {
         return(top==-1);
     }

    //2. implement queue using arrays
    int arr[],n;
    int count;
    void push(x)
    {
        if(count==n) return; // full
        arr[rear%n]=x;
        rear++,count++;
    }

    void pop()
    {
        if(count==0) return ;// empty
        int temp=arr[front%n];
        arr[front%n]=-1; // dummy no
        front++,count--;

    }
    int top()
    {
        if(count ==0) return -1;
        return arr[front%n];
    }
    int size()
    {
        return count;
    }


    //3. implement stack using queue - single queue
    queue<int>q;
    void push(int x)
    {
        q.push(x);
        for(int i=0;i<q.size()-1;i++)
        {
            q.push(q.front());
            q.pop();
        }

  
    } // tc-O(n), sc-O(n) fr queue used 
    void pop()
    {
        q1.pop();
    }
    int top()
    {
        return q.front();
    }

    //4.queue implementation using stack O(1) push & pop
    stack<int>ip,op;

    void push(int x)
    {
        ip.push(x); // O(1)
    }
    void pop()
    {
        if(!op.empty)
          op.pop();
        else 
          {
              while(!ip.empty())
              {
                  op.push(ip.top());
              }
              op.pop();
          }
    } // O(1) amortised 

    int top()
    {
        if(!op.empty) op.top();
        else 
        {
            while(!ip.empty())
            {
                op.push(ip.top());
            }
            return op.top();
        }
    } // O(1) amortised

    //5. valid paranthesis
    bool isvalid_par(string s)
    {
        int n=s.length();
        if(n%2!=0) return false;
        stack<int>st;
        for(int i=0;i<n;i++)
        {
            if(s[i]=='['  ||  s[i]=='{'|| s[i]=='(' ) 
             st.push(s[i]);
            else 
            {
                if(st.empty()) return false;
                char c=st.top();
                st.pop();
                if(s[i]==')' && c=='(' || s[i]==']' && c=='[' || s[i]=='}' && c=='{'   ) continue;
                else return false;
            }
        }
        if(st.empty()) return true;
        else return false;
    } //tc-O(n) sc-O(n) 

    //6. // next greater element II-  circular array variant
      vector<int> nextGreaterElements(vector<int>& nums) {
        int n=nums.size();
        vector<int>nge(n);
        stack<int>st;
        for(int i=2*n-1;i>=0;i--) // for first variant just keep n-1
        {
            while(!st.empty()&& st.top()<=nums[i%n])
            {
                st.pop();
            }
           if(i<n){
            if(!st.empty()) nge[i]=st.top();
            else nge[i]=-1;
           }
            st.push(nums[i%n]);
        }
        return nge;
    } // tc-O(2n) sc-O(n) 

    // 7. sort a stack
 void SortedStack :: sort()
{
  
   int n=s.size();
  multiset<int>a;
  while(n--)
  {
      a.insert(s.top());
      s.pop();
  }
  for(auto i : a)
  {
      s.push(i);
  }
} // tc -O(n) sc-O(n)

// DAY 8
// STACKS AND QUEUES

 //1. next smaller element
 // same as next greater element ..just pop if(st.top()>=arr[i]) till we remove all elements greater than arr[i]
 vector<int> nextSmallerElement(vector<int> &arr, int n)
{
    // Write your code here.
    vector<int>nse(n);
    stack<int>st;
    for(int i=n-1;i>=0;i--)
    {
        while(!st.empty() && st.top()>=arr[i])
        {
            st.pop();
        }
        if(!st.empty()) nse[i]=st.top();
        else nse[i]=-1;
        st.push(arr[i]);
    }
    return nse;
    
} // tc -o(n) sc-O(n)

// 2. LRU cache
class LRUCache{
    public:
    class node {
        public:
            int key;
            int val;
            node* next;
            node* prev;
        node(int _key, int _val) {
            key = _key;
            val = _val; 
        }
    };
    
    node* head = new node(-1,-1);
    node* tail = new node(-1,-1);
    
    int cap;
    unordered_map<int, node*>m;
    
    LRUCache(int capacity) {
        cap = capacity;    
        head->next = tail;
        tail->prev = head;
    }
    
    void addnode(node* newnode) {
        node* temp = head->next;
        newnode->next = temp;
        newnode->prev = head;
        head->next = newnode;
        temp->prev = newnode;
    }
    
    void deletenode(node* delnode) {
        node* delprev = delnode->prev;
        node* delnext = delnode->next;
        delprev->next = delnext;
        delnext->prev = delprev;
    }
    
    int get(int key_) {
        if (m.find(key_) != m.end()) {
            node* resnode = m[key_];
            int res = resnode->val;
            m.erase(key_);
            deletenode(resnode);
            addnode(resnode);
            m[key_] = head->next;
            return res; 
        }
    
        return -1;
    }
    
    void put(int key_, int value) {
        if(m.find(key_) != m.end()) {
            node* existingnode = m[key_];
            m.erase(key_);
            deletenode(existingnode);
        }
        if(m.size() == cap) {
          m.erase(tail->prev->key);
          deletenode(tail->prev);
        }
        
        addnode(new node(key_, value));
        m[key_] = head->next; 
    }
    
    
};

//3. largest rectangle in histogram
  // two pass solution
  int largestarea(vector<int>arr,int n)
  {
      stack<int>s;
      int ls[n];
      int rs[n];
      // for left smaller ele
      for(int i=0;i<n;i++)
      {
          while(!st.empty() && arr[st.top()]>=arr[i]) st.pop();
          if(st.empty()) ls[i]=0;
          else ls[i]=st.top()+1;
          st.push(i);
      }
      // ls done..now clear the stack to reuse for r8 smaller ele
      st.clear();
      for(int i=n-1;i>=0;i--)
      {
          while(!st.empty() && arr[st.top()]>arr[i]) st.pop();
          if(st.empty()) rs[i]=n-1;
          else rs[i]=st.top()-1;
          st.push(i);
      } // rs done
      int maxa=INT_MIN;
      for(int i=0;i<n;i++)
      {
          maxa=max(maxa,(rs[i]-ls[i]+1)*arr[i]);
      }
      return maxa;

  } // tc-O(4n) sc-O(3n)

   //one pass solution
     int largestRectangleArea(vector<int>& heights) {
       
         int ans = 0;
    stack<int> stack;

    for (int i = 0; i <= heights.size(); ++i) {
      while (!stack.empty() &&
             (i == heights.size() || heights[stack.top()] > heights[i])) {
        const int h = heights[stack.top()];
        stack.pop();
        const int w = stack.empty() ? i : i - stack.top() - 1;
        ans = max(ans, h * w);
      }
      stack.push(i);
    }

    return ans;
    } // tc-O(n) sc-O(n)
    //4,lfu cache ..ignored

    //5. sliding window maximum
    vector<int> maxSlidingWindow(vector<int>& nums, int k)
    {
        vector<int>ans;
        int n=nums.size();
        deque<int>dq;
        for(int i=0;i<n;i++)
        {
            if(!dq.empty() && dq.front()==i-k) dq.pop_front(); // removing all index which is out of bound
            while(nums[i]>nums[dq.back()] && !dq.empty()) dq.pop_back(); // maintaining decreasing order .. index having maxm value at front and minimum at back
            dq.push_back(i);
            if(i>=k-1) ans.push_back(nums[i]); 


        }
        return ans;
    } // tc-O(n) + O(n) sc-O(n)

    // 6. implementing minm stack
    #define ll long long
    stack<ll>st;
    ll mini;
     MinStack() {
        while(st.empty()==false) st.pop();
        mini=INT_MAX;

    }
    
    void push(int val) {
       ll value=val;
       if((st.empty()))
       {
           st.push(value);
           mini=value;
       }
       else if(mini<value)
       {
           st.push(value);
       }
       else 
       {
           st.push(2*value - mini);
           mini=value;
       }
    }
    
    void pop() {
       if(st.empty()) return;
      int ele=st.top();
      st.pop();
      if(ele<mini) mini=2*mini - ele;
      
    }
    
    int top() {
       if(st.empty()) return -1;
       ll ele=st.top();
       if(ele < mini) return mini;
       return ele;
    }
    
    int getMin() {
        return mini;
    } // tc-O(n) sc-O(n)

    //7. rotten oranges
     int orangesRotting(vector<vector<int>>& grid) {
        if(grid.size()==0) return 0;
        int cnt=0;
        int m=grid.size(),n=grid[0].size();
        int days=0,tot_oran=0;
        queue<pair<int,int>>q;
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++) 
            {
                if(grid[i][j]!=0) tot_oran++;
                if(grid[i][j]==2) q.push({i,j});
                
            }
        }
        int dx[]={0,0,1,-1};
        int dy[]={1,-1,0,0};
        while(!q.empty())
        {
            int k=q.size();
            cnt+=k;
            while(k--)
            {
                int x=q.front().first;
                int y=q.front().second;
                q.pop();
                for(int i=0;i<4;i++)
                {
                    int nx=x+dx[i],ny=y+dy[i];
                    if(nx<0 || ny<0 || nx>=m || ny>=n|| grid[nx][ny]!=1) continue;
                    grid[nx][ny]=2;
                    q.push({nx,ny});
                }
            }
            if(!q.empty()) days++;
            
        }
        return tot_oran==cnt ? days :-1;
    } // tc-O(n*n)*4 sc-O(n*n)

    //8. stock span prblm
       vector <int> calculateSpan(int price[], int n)
    {
       
       vector<int>ans;
       stack<pair<int,int>>st;
       for(int i=0;i<n;i++)
       {
           int d=1;
           while(!st.empty() && st.top().first <=price[i])
           {
           d+=st.top().second;
           st.pop();
           }
           st.push({price[i],d});
           ans.push_back(d);
       }
       return ans;
       
    }// tc-O(n) sc-O(n)

    // 9 . maximum of minimum for every window size
     vector <int> maxOfMin(int arr[], int n)
    {
        // Your code here
           stack<int> st;
       vector<int>ans(n);
       for(int i=0;i<n;i++)
       {
           while(!st.empty()&&arr[i]<arr[st.top()])
           {
               int tp = st.top();st.pop();
               int window= i-(st.empty()?-1:st.top())-1-1;// -1 again because 0 based indexing in result.
               ans[window]=max(ans[window],arr[tp]);
           }
           st.push(i);
       }
       while(!st.empty())
       {
           int tp = st.top();st.pop();
           int window = n-(st.empty()?-1:st.top())-1-1;
           ans[window]=max(ans[window],arr[tp]);
       }
       for(int i=n-2;i>=0;i--){ans[i]=max(ans[i],ans[i+1]);}
       return ans;
    } // tc-O(n)+O(n) sc-O(n)

    //10. celebrity problem
      int celebrity(vector<vector<int> >& M, int n) 
    {
       
     stack<int>s;
     for(int i=0;i<n;i++) s.push(i);
     while(s.size()>1)
     {
         int a=s.top();
         s.pop();
         int b=s.top();
         s.pop();
         if(M[a][b]==0) s.push(a); // a doesnt know b so b cant be celeb
         else s.push(b);
     }
     int x=s.top();
     for(int i=0;i<n;i++) // checking if x is celeb or not
     {
         if(M[x][i]==1) return -1;
         if(x!=i && M[i][x]==0) return -1;
     }
     return x;
    }// tc-O(n) sc-O(n)

   int celebrity(vector<vector<int> >& M, int n) 
   {
       int c = 0;
       for(int i = 1; i<n; i++){
           if(M[c][i] == 1){
               c = i;
           }
       }
       for(int i = 0; i<n; i++){
           if(i != c){
               if(M[c][i] != 0 || M[i][c] != 1){
                   return -1;
               }
           }
       }
       return c;
   } // tc-0(n) sc-O(1)


// DAY 15
// STRINGS

  //1. reverse words in a string
   string reverseWords(string s) {
        
         reverse(s.begin(), s.end());
    int storeIndex = 0;
    for (int i = 0; i < s.size(); i++) {
        if (s[i] != ' ') {
            if (storeIndex != 0) storeIndex++;
            int j = i;
            while (j < s.size() && s[j] != ' ')  
                j++;  
            reverse(s.begin() + storeIndex, s.begin() + j);
            storeIndex += (j - i);
            i = j;
        }
    }
    s.erase(s.begin() + storeIndex, s.end());
        return s;
    } // tc-O(n) sc-O(1)

    //2. longest palindromic substring
    string l_p_substring(string s)
    {
        int n=size();
        if(n==0) return "";
        vector<vector<bool>>t(n,vector<bool>(n,false));
        for(int i=0;i<n;i++)
        {
            t[i][i]=true;
            if(i>n-1) break;
            t[i][i+1]=(s[i]==s[i+1]);

        }
        //dp starts
        for(int i=n-3;i>=0;i--)
        {
            for(int j=i+2;j<n;j++)
            {
                t[i][j]=(t[i+1][j-1] && (s[i]==s[j]));
            }
        }
        // calc ans from table
        int ans=0;
        string maxstr="";
        for(int i=0;i<n;i++)
        {
            for(int j=i;j<n;j++)
            {
                if(t[i][j]==true && j-i+1>ans)
                {
                    ans=j-i+1;
                    maxstr=s.substr(i,j-i+1);
                }
            }
        }
        return maxstr;
    } // tc-O(n*n) sc-O(n*n)


    //3.integer to roman
    int int_to_roman(string s)
    {
        if(s.length()==0) return -1;
        int n=s.length();
        unordered_map<char,int>mp;
        mp.insert('I',1);
        mp.insert('V',5);
        mp.insert('X',10);
        mp.insert('L',50);
        mp.insert('C',100);
        mp.insert('D',500);
        mp.insert('M',1000);
        int res=mp[s[n-1]];
        for(int i=n-2;i>=0;i--)
        {
            if(mp[s[i]] < mp[s[i+1]]) res-=mps[i]];
            else res+=mp[s[i]];
        }
        return res;
    } // tc-O(n) sc-O(n)

    // roman to int
     vector<string> rom({"I","IV","V","IX","X","XL","L","XC","C","CD","D","CM","M"});
        vector<int>val({1,4,5,9,10,40,50,90,100,400,500,900,1000});
        int n=rom.size();
        int index=n-1;
        string s="";
        while(num>0){
            while(val[index]<=num){
                s =s + rom[index];
                num-=val[index];
                
            }
            index--;
            
        }
        return s;
        
    } // tc-O(n) sc-O(2n)
    

    //4. atoi//strstr()

      int myAtoi(string s) {
   if(s.length()==0)
       return 0;
        int i=0;
        while(s[i]==' ')
            i++;
        bool flag=true;
        if(s[i]=='+' || s[i]=='-'){
            flag= (s[i]=='+'? true:false);
            i++;
            
        }
        if(s[i]-'0'<0 || s[i]-'9'>9)
            return 0;
        int n=0;
        while(s[i]>='0' && s[i]<='9'){
             if(n>INT_MAX/10 ||( n==INT_MAX/10 && s[i]-'0'>7)){
                 
             
                 return flag?INT_MAX : INT_MIN;
             }
            n=n*10 + (s[i]-'0');
            i++;
            
        }
        return flag?n : n*-1;   
    } // tc-O(n) sc-O(1)

     int strStr(string haystack, string needle) {
        if(needle.length()==0) return 0;
        int p= haystack.find(needle);
        return p;
        
    } // tc-O(1) sc-O(1)

    //5. longest common prefix
    string longest_commonprefix(vector<string>&s)
    {
        int ans=INT_MAX;
        string temp=s[0];
        for(int i=1;i<s.size();i++)
        {
            int j=0,a=0,k=0;
            while(j<temp.size() && k<s[i].size()) 
            {
                if(temp[j]==s[i][k]) a++;
                else break;
                j++,k++;
            }
            ans=min(ans,a);
        }
        return temp.substr(0,ans);
    } // tc-O(n) sc-O(1)

    //6. Rabin karp algo
    // Rabin-Karp algorithm in C++

#include <string.h>

#include <iostream>
using namespace std;

#define d 10

void rabinKarp(char pattern[], char text[], int q) {
  int m = strlen(pattern);
  int n = strlen(text);
  int i, j;
  int p = 0;
  int t = 0;
  int h = 1;

  for (i = 0; i < m - 1; i++)
    h = (h * d) % q;

  // Calculate hash value for pattern and text
  for (i = 0; i < m; i++) {
    p = (d * p + pattern[i]) % q;
    t = (d * t + text[i]) % q;
  }

  // Find the match
  for (i = 0; i <= n - m; i++) {
    if (p == t) {
      for (j = 0; j < m; j++) {
        if (text[i + j] != pattern[j])
          break;
      }

      if (j == m)
        cout << "Pattern is found at position: " << i + 1 << endl;
    }

    if (i < n - m) {
      t = (d * (t - text[i] * h) + text[i + m]) % q;

      if (t < 0)
        t = (t + q);
    }
  }
}

int main() {
  char text[] = "ABCCDDAEFG";
  char pattern[] = "CDD";
  int q = 13;
  rabinKarp(pattern, text, q);
} // tc-O(m+n) avg 

  
// DAY 16 
//*STRINGS*

  //1. prefix function
  vector<int> prefix_function(string s) {
    int n = (int)s.length();
    vector<int> pi(n);
    for (int i = 1; i < n; i++) {
        int j = pi[i-1];
        while (j > 0 && s[i] != s[j])
            j = pi[j-1];
        if (s[i] == s[j])
            j++;
        pi[i] = j;
    }
    return pi;
} //tc-O(n) osc-O(1)

 //2.  KMP algo
 	int lps(string s) {
	    // Your code goes here
	     int n=s.size();
	    
	    int arr[n]={0};
	    int i=0,j=1;
	    arr[0]=0;
	    while(j<n){
	        if(s[i]==s[j]){
	            arr[j++]=i+1;
	            i++;
	            
	            
	        }
	        else {
	            if(i!=0) i=arr[i-1];
	            else arr[j++]=0;
	        }
	    }
	    return arr[n-1];
	}// tc-O(n) Osc-O(n)

    //3.minm no of chars to be added in front of string to make it a palindrome
    //using KMP algo
    // returns vector lps for given string str
vector<int> computeLPSArray(string str)
{
    int M = str.length();
    vector<int> lps(M);
 
    int len = 0;
    lps[0] = 0; // lps[0] is always 0
 
    // the loop calculates lps[i] for i = 1 to M-1
    int i = 1;
    while (i < M)
    {
        if (str[i] == str[len])
        {
            len++;
            lps[i] = len;
            i++;
        }
        else // (str[i] != str[len])
        {
            // This is tricky. Consider the example.
            // AAACAAAA and i = 7. The idea is similar
            // to search step.
            if (len != 0)
            {
                len = lps[len-1];
 
                // Also, note that we do not increment
                // i here
            }
            else // if (len == 0)
            {
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}
 
// Method returns minimum character to be added at
// front to make string palindrome
int getMinCharToAddedToMakeStringPalin(string str)
{
    string revStr = str;
    reverse(revStr.begin(), revStr.end());
 
    // Get concatenation of string, special character
    // and reverse string
    string concat = str + "$" + revStr;
 
    //  Get LPS array of this concatenated string
    vector<int> lps = computeLPSArray(concat);
 
    // By subtracting last entry of lps vector from
    // string length, we will get our result
    return (str.length() - lps.back());
} // tc-O(n) sc-O(n)

  //4. check for anagrams
   bool isAnagram(string s, string t) {
       if(s.length()!=t.length()) return false;
        vector<int>cnt(128);
        for(auto c: s) ++cnt[c];
        for(auto x:t) 
        {
            if(--cnt[x]<0) return false;
        }
        return true;
    } // tc-O(n) sc-O(256)

    //5. count and say
       string countAndSay(int n) {
        if(n==1) return "1";
        if(n==2) return "11";
        string s="11";
        for(int i=3;i<=n;i++){
            string t="";
            s=s+"@";
            int c=1;
            for(int j=1;j<s.length();j++){
                if(s[j]!=s[j-1]){
                    t=t+to_string(c);
                    t=t+s[j-1];
                    c=1;
                }
                else c++;
            }
            s=t;
        }
        return s;
    } // tc-O(n*n) sc-o(n)

//DAY 17
//* BINARY TRESS*

 //1. INORDER traversal reccursive and iterative

 void inorder(Node* root)
 {
     if(root==NULL) return;
     inorder(root->left);
     cout<<root->data;
     inorder(root->right);
 } //tc-O(n) sc-o(1)

 vector<int> inorder(Node* root)
 {
     if(!root) return {};
     stack<Node*>st;
     Node* node=root;
     vector<int>inorder;
     while(true)
     {
         if(node!=NULL) st.push(node),node=node->left;
         else 
         {
             if(st.empty()) break;
             node=st.top();
             st.pop();
             inorder.push_back(node->data);
             node=node->right;
         }  
     }
     return inorder;
 } //tc-O(n) osc-O(n)

  //2. Preorder - with and without recuursion
   void preorder(Node* root)
   {
       if(!root) return;
       cout<<root->data;
       preorder(root->left);
       preorder(root->right);
   } //tc-O(n) sc-O(1)

   vector<int>preorder(Node* root)
   {
       if(!root) return {};
       vector<int>pre;
       stack<Node*>st;
       Node* node=root;
       st.push(node);
       while (!st.empty())
       {
          Node* temp=st.top();
          st.pop();
          pre.push_back(temp->data);
          if(temp->right!=NULL) st.push(temp->right);
          if(temp->left!-NULL) st.push(temp->left);

       }
       return pre;
       
   } //tc-O(n) sc-O(n)

   //3.postorder
  void postorder(Node* root)
  {
      if(!root) return;
      postorder(root->left);
      postorder(root->right);
      cout<<root->data;
  }

  vector<int>postorder(Node* root)
  {
      if(!root) return{};
      vector<int>post;
      stack<Node*>st;
      Node* curr=root;
      Node* temp=NULL;
      st.push(curr);
      while(curr!=NULL || !st.empty())
      {
          if(curr!=NULL) st.push(curr),curr=curr->left;
          else temp=st.top()->right;
          if(temp==NULL) 
          {
              temp=st.top();
              st.pop();
              post.push_back(temp->data);
              while(!st.empty() && temp=st.top()->right)
              {
                  temp=st.top();
                  st.pop();
                  post.push_back(temp->data);

              }
              
          }
          else curr=temp;
      }
      return post;
  } // O(2n) sc-O(n)

  //4. right /left view of binary tree - here we doin right view
  vector<int>rightview(Node* root)
  {
      vector<int>ans;
      solve(root,ans,0); // 0 is level
      return ans;
  } // tc-O(n) sc-O(h)
  void solve(Node* root, vector<int>&ans, int &level)
  {
      if(!root) return;
      if(ans.size()==level) ans.push_back(root->data);
      solve(root->right,ans.level+1);
      solve(root->left,ans,level+1); // we are doin reverse pre order for right side view..for left side view...just do pre-order
  }

  //vertical order traversal
  vector<vector<int>>Verticalorder(Node* root)
  {
      map<int,map<int,multiset<int>>>mp;
      queue<pair<Node*,pair<int,int>>>q;
      q.push({root,{0,0}});
      while(!q.empty())
    {
        auto p=q.front();
        q.pop();
        Node* node=p.first;
        int x=p.second.first,y=p.second.second;
        mp[x][y].insert(node->val);
        if(node->left)
        {
            q.push({node->left,{x-1,y+1}});
        }
        if(node->right) q.push({node->right,{x+1,y+1}});
    }
    vector<vector<int>>ans;
    for(auto x: mp)
    {
        vector<int>temp;
        for(auto q: x.second)
        {
            temp.inset(temp.end(),q.second.begin(),q.second.end());

        }
        ans.push_bacK(temp);
    }
    return ans;


  } // tc-O(nlogn) sc-o(n)

  //5.Top view of binary tree
  vector<int>topview(Node* root)
  {
      vector<int>ans;
      if(!root) return ans;
      queue<pair<Node*,int>>q; // node,line
      map<int,int>mp; // line ->first node
      q.push({root,0});
      while(!q.empty())
      {
          auto it=q,front();
          q.pop();
          Node* node=it.first;
          int line=it.second;
          if(mp.find(line)==mp.end()) mp[line]=node->data;
          if(node->left!-NULL) q.push({node->left,line-1});
          if(node->right!= NULL) q.push({node->right,line+1});
      }
      for(auto x : mp)
      {
          ans.push_back(it.second);
      }
      return ans;
  } // tc-O(n) sc-O(n)

   //6. Bottom view of BT
   vector<int>bootomview(Node* root)
   {
       vector<int>ans;
       if(!root) return ans;
       queue<pair<Node*,int>>q; // node - line
       map<int,int>mp;
       q.push({root,0});
       while(!q.empty())
       {
           auto it=q.front();
           q.pop();
           Node* node=it.first;
           int line=it.second;
           mp[line]=node->data;
           if(node->left!=NULL) q.push({node->left,line-1});
           if(node->right!=NULL) q.push({node->right,line+1});
       }
       for(auto x:mp)
       ans.push_back(x.second);
       return ans;
   } // tc-O(n) sc-O(n)

   //DAY18 BINARY TRESS
    
    //1. level order traversal
    vector<vector<int>>levelorder(Node* root)
    {
        vector<vector<int>>ans;
        queue<Node*>q;
        if(!root) return ans;
        q.push(root);
        while(!q.empty())
        {
            int k=q.size();
            vector<int>temp;
            while(k--)
            {
                Node* node=q.front();
                q.pop();
                
                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);
                temp.push_back(node->data);

            }
            ans.push_back(temp);
        }
        return ans;
    } // tc-O(n) sc-O(n)

    //2. height of a binary tree
    int heightofBT(Node* root)
    {
        if(root==NULL) return 0;
        int l=heightofBT(root->left);
        int r=heightofBT(root->right);
        return 1+max(l,r);

    } // tc-O(n) sc-O(n)

    //3. diameter of binary tree

    int solve(Node* root,int d)
    {
        if(!root) return 0;
        int lh=solve(node->left,d);
        int rh=solve(node->right,d);
        d=max(d,lh+rh);
        return 1+max(h,rh);
    }
    int diameter(Node* root) // we are using height of the tree concept and slight modification in code
    {
        
        int maxi=0;
        solve(root,maxi);
        return maxi;
    } //tc-O(n) osc-O(n) 

  //4.check for balanced tree
  bool isbalanced(Node* root)
  {
      return ht(root)!=-1;
  } // tc-O(n) osc-o(n)

  int ht(Node* root)
  {
      if(!root) return 0;
      int lh=ht(root->left);
      if(lh==-1) return -1;
      int rh=ht(root->right);
      if(rh==-1) rteurn -1;
      if(abs(lh-rh)>1) return -1;
      return max(lh,rh)+1;
  }
  
  //5. LCA in BT
  Node* lca(Node* root,Node* a,Node* b)
  {
      if(root==NULL || root==a || root==b) return root;
      Node* l=lca(root->left,a,b);
      Node* r=lca(root->right,a,b);
      if(!l) return r;
      else 
      if(!r) return l;
      else return root;
  } //tc-o(n) sc-O(n)

  //6. check if two trees are identical or not
  bool isidentical(Node* p, Node* q)
  {
      if(!p || !q) return p==q;
      return (p->val==q->val) && isidentical(p->left,q->left) && isidentical(p->right,q->right);
  } //tc-O(n) sc-O(1)

  //DAY 19 
  // * BINARY TRESS*

  //1. Maxm path sum
  int maxpathsum(Node* root)
  {
      int res=INT_MIN;
      solve(root,res);
      return res;
  } //tc-O()
  int solve(Node* root,int &maxi)
  {
      if(!root) return 0;
      int lsum=solve(root->left,maxi);
      int rsum=solve(root->right,maxi);
      maxi=max(maxi,root-val + lsum+rsum);
      return node->val+max(lsum,rsum);
  }
   
   //2. cerating BT from inorder and preorder traversal
   Node* buildtree(vector<int>&preorder, vector<int>&inorder)
   {
       map<int,int>inmp;
       for(int i=0;i<inorder.size();i++) inmp[inorder[i]]=i;
       Node* root= build(preorder,0,preorder.size()-1,inorder,0,inorder.size()-1,inmp);
       return root;
   }  // tc-O(nlogn) sc-O(n)

   Node* build(vector<int>&preorder, int pstart,int pend,vector<int>&inorder,int istart,int iend,map<int,int>&inmp)
   {
       if(pstart > pend) return NULL;
       Node* root=new Node(preorder[pstart]);
       int inroot=inmp[root->val];
       int numleft=inroot-istart;
       root->left=build(preorder,pstart+1,pstart+numleft,inorder,istart,inroot-1,inmp);
       root->right=build(preorder,pstart+numleft+1,pend,inorder,inroot+1,iend,inmp);
       return root;
   }

  //3. creating BT from inorder and postorder traversal
    Node* buildtree(vector<int>&postorder, vector<int>&inorder)
   {
       map<int,int>inmp;
       for(int i=0;i<inorder.size();i++) inmp[inorder[i]]=i;
       Node* root= build(postorder,0,postorder.size()-1,inorder,0,inorder.size()-1,inmp);
       return root;
   }  // tc-O(nlogn) sc-O(n)

   Node* build(vector<int>&postorder, int pstart,int pend,vector<int>&inorder,int istart,int iend,map<int,int>&inmp)
   {
       if(pstart > pend || istart > iend) return NULL;
       Node* root=new Node(postorder[pend]);
       int inroot=inmp[root->val];
       int numleft=inroot-istart;
       root->left=build(postorder,pstart,pstart+numleft-1,inorder,istart,inroot-1,inmp);
       root->right=build(postorder,pstart+numleft,pend-1,inorder,inroot+1,iend,inmp);
       return root;
   }

   //4.check if BT is symmetric
    bool solve(TreeNode* p,TreeNode* q)
     {
         if(!p || !q) return p==q;
         return p->val==q->val && (solve(p->left,q->right)&& solve(p->right,q->left));
     }
    bool isSymmetric(TreeNode* root) {
        return solve(root,root);
        
    } //tc-O(n) sc-O(n)

    //5. flatten binary tree to linked list
     void flatten(TreeNode* root) {
        if(!root) return;
        while(root)
        {
            if(root->left)
            {
                TreeNode* r8most=root->left;
                while(r8most->right) r8most=r8most->right;
                r8most->right=root->right;
                root->right=root->left;
                root->left=NULL;
            }
            root=root->right;
        }
    } //tc-O(n) sc-O(1)

    //6. print all nodes at a distance of k
     void markParents(TreeNode* root, unordered_map<TreeNode*, TreeNode*> &parent_track, TreeNode* target) {
        queue<TreeNode*> queue;
        queue.push(root);
        while(!queue.empty()) { 
            TreeNode* current = queue.front(); 
            queue.pop();
            if(current->left) {
                parent_track[current->left] = current;
                queue.push(current->left);
            }
            if(current->right) {
                parent_track[current->right] = current;
                queue.push(current->right);
            }
        }
    }

    vector<int> distanceK(TreeNode* root, TreeNode* target, int k) {
        unordered_map<TreeNode*, TreeNode*> parent_track; // node -> parent
        markParents(root, parent_track, target); 
        
        unordered_map<TreeNode*, bool> visited; 
        queue<TreeNode*> queue;
        queue.push(target);
        visited[target] = true;
        int curr_level = 0;
        while(!queue.empty()) { /*Second BFS to go upto K level from target node and using our hashtable info*/
            int size = queue.size();
            if(curr_level++ == k) break;
            for(int i=0; i<size; i++) {
                TreeNode* current = queue.front(); queue.pop();
                if(current->left && !visited[current->left]) {
                    queue.push(current->left);
                    visited[current->left] = true;
                }
                if(current->right && !visited[current->right]) {
                    queue.push(current->right);
                    visited[current->right] = true;
                }
                if(parent_track[current] && !visited[parent_track[current]]) {
                    queue.push(parent_track[current]);
                    visited[parent_track[current]] = true;
                }
            }
        }
        vector<int> result;
        while(!queue.empty()) {
            TreeNode* current = queue.front(); queue.pop();
            result.push_back(current->val);
        }
        return result;
    } // tc-O(n) sc-O(n)


// DAY 20 *BINARY SEARCH TREE*

 // 1. Populate next right pointers in each node
 Node* connect(Node* root)
 {
     if(!root || !root->left || !root->right) return NULL;
     root->left->next=root->right;
     if(root->next!=NULL)
     {
         root->right->next=root->next->left;
     }
     connect(root->left);
     connect(root->right);
     return root;

 } // tc-O(n) sc-O(1)

   //2. search in a BST
   Node* find(Node* root, int val)
   {
       while(root!=NULL && root->data!=val)
       {
           root= root->data < val ? root->left : root->right;
       }
       return root;
   } // tc-O(logn) sc-O(1)

   //3. creating balanced bst from given keys

   Node* BST( vector<int>&arr,int low,int high)
   { 
       if(low>high) return NULL;
       int mid=(l+r)>>1;
       Node* root=new Node(arr[mid]);
       root->left=BST(arr,low,mid-1);

       root->right=BST(arr,mid+1,high);
       return root;

   }
   Node* createBST(vector<int>&arr)
   {
       int n=arr.size();
       sort(arr.begin(),arr.end());
       Node* root=BST(arr,0,n-1);
       return root;
   } //tc-O(nlogn) sc-O(n)

   //4.check if BT is BST or not

   bool check(Node* root, int minval, int maxval)
   {
       if(!root) return true;
       if(root->val >= maxval || root->val <=minval) return false;
       return check(root->left,minval,root->val) && check(root->right,root->val,maxval);

   }
   bool isBST(Node* root)
   {
       return check(root,INT_MIN,INT_MAX);
   } //tc-O(n) sc-O(1)

   // 5. LCA in a BST
   Node* lcaBST(Node* root, Node* p, Node* q)
   {
       if(!root) return root;
       int curr=root->val;
       if(curr< p-val && curr<q->val) return lcaBST(root->right,p,q);
       if(curr> p-val && curr>q->val) return lcaBST(root->left,p,q);
       return root;

   } //tc-O(h) sc-O(1)

   //6.inorder successor in BST
   Node* inorder_succ(Node* root, Node* a)
   {
       Node* succ=NULL;
       while(root!=NULL)
       {
           if(a->val > root->val) root=root->right;
           else{
               succ=root;
               root=root->left;
           }
       }
       return succ;
   } // tc-O(h) sc-O(1)


