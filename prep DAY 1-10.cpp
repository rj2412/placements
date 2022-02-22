#include <iostream>
using namespace std;

// placements preparations
// ARRAYS
// DAY 1
// 1. sort array of 0 1 and 2 without sorting and extra space
void solve(vector<int>arr)
{
    //1st method to sort 2nd is to linear traverse and use counting sort
    int n=arr.size();
    int low=0,mid=0;
    int high=n-1;
    while(mid<=high)
    {
        if(arr[mid]==0) {
            swap(arr[low++],arr[mid++]);
            
        }
        else if(arr[mid]==1) mid++;
        else swap(arr[mid],arr[high--]);
        
    } // TC- O(logn) SC- O(1)

}
//2. find missing and repeating no
vector<int>find (vector<int>arr)
{
    int n=arr.size();
 vector<int>freq(n+1,0);
 freq[0]=1;
 int miss,rep;

for(int i=0;i<n;i++) freq[arr[i]]++;
for(int i=1;i<=n;i++) {
if(freq[i]>1) rep=freq[i];
else if(freq[i]=0) miss=freq[i];
}
 return {miss,rep};
}

//3. Merge two sorted array in O(1) SPACE
 void merge(long long arr1[], long long arr2[], int n, int m) 
        { 
            
            for(long long i=0;i<n;i++) // traversing the 1st array 
            {
                if(arr1[i]>arr2[0])
                {
                    swap(arr1[i],arr2[0]);
                    long long fst=arr2[0];
                    // now we have to maintain the order of arr2
                    long long k;
                    for(k=1;k<m && arr2[k]<fst;k++)
                    {
                        arr2[k-1]=arr2[k];
                    }
                    arr2[k-1]=fst;
                }
            }
        } 

//4. kADANE'S algo ..maximum subarray sum
int maxsubarr(vector<int>nums)
{
    int sum=0;
    int ans=INT_MIN;
    for(auto it:nums)
    {
        sum+=it;
        ans=max(ans,sum);
        if(sum<0) sum=0;
    }
    return ans; // TC-O(n) sc-O(1)

}

//5. Merge intervals
vector<vector<int>>merge(vector<vector<int>>nums)
{    
    if(nums.size()==0) return nums;
    vector<vector<int>>ans;
    sort(nums.becgin(),nums.end());
    ans.push_back(nums[0]);
    for(auto it : nums)
    {
      if(ans.back()[1]>=it[1]) ans.back()[1]=max(ans.back()[1],it[1]);
      else ans.push_back(it);
    }
    return ans;
} // TC-O(n) SC-O(n)

//6. Find duplicate no leetcode 287
// brute approach is to sort array and check if(a[i]==a[i+1]) return a[i]; tc-O(nlogn) sc-O(1)
// better approach- do counting sort and if any count==2 return that ; tc-O(n)+O(n), sc-O(n)
// optimal approach- cycle method
int findDuplicate(vector<int>& nums) {
       int s=nums[0];
        int f=nums[0];
        do{
            s=nums[s];
            f=nums[nums[f]];
        }while(s!=f);
        f=nums[0];
        while(s!=f)
        {
            s=nums[s];
            f=nums[f];
        }
        return s;
    } // TC-O(n) SC-O(1)
     

    // DAY 2 
    // ARRAYS
    //1.Set matrix zeroes
    vector<vector<int>>setzeroes(vector<vector<int>>&mat)
    {
        int r=mat.size(),c=mat[0].size(),col0=1;
        for(int i=0;i<r;i++)
        {
            if(mat[i][0]==0) col0=0;
            for(int j=1;j<m;j++)
            {
                if(mat[i][j]==0) 
                {
                    mat[i][0]=mat[0][i]=0;
                }
            }
        }
        for(int i=r-1;i>=0;i--)
        {
            for(int j=c-1;j>=1;j--)
            {
                if(mat[i][0]==0||mat[0][j]==0) mat[i][j]=0;
            }
            if(col0=0) mat[i][0]=0;
        }
// TC- 2*R*C sc- O(1)
    }

//2. PASCALS TRIANGLE
// 3 types of questions can be asked
vector<vector<int>>genPascaltriangle(int nrows) // no of rows as input and we have to generate the triangle
{
    vector<vector<int>>r(nrows);
    for(int i=0;i<nrows;i++)
    {
        r.resize(i+1);
        r[i][0]=r[i][i]=1;
        for(int j=1;j<i;j++)
        {
            r[i][j]=r[i-1][j-1]+r[i-1][j];
        }


    }
    return r;
}

//3. next permutation
void nextpermutation(vector<int>&nums)
{
    int n=nums.size();
    int k,l;
    for(k=n-2;k>=0;k--) 
    {
        if(nums[k]>nums[k+1]) break;
    }
    if(k<0) reverse(nums.begin(),nums.end())//edge case eg. nums{5,4,3,2,1}
    else {
        for(l=n-1;l>k;l--) {
            if(nums[l]>nums[k])
            break;

        }

    }
    swap(nums[k],nums[l]);
    reverse(nums.begin()+k+1,nums.end());

}// TC-O(n) sc- O(1)

//4. count inversions
int merge(vector<int>&arr,vector<int>&temp,int left,int mid, int right)
{
    int i,j,k;
    i=left,j=mid,k=left;
    while(i<=mid-1&& j<=right)
    {
        if(arr[i]<=arr[j]) temp[k++]=arr[i++];
        else 
        {
            temp[k++]=arr[j++];
            inv_cnt=inv_cnt + (mid-i);

        }
    }
    while(i<=mid-1) temp[k++]=arr[i++];
    while(j<=right) temp[k++]=arr[j++];
    for(i=left;i<=right;i++)
    {
        arr[i]=temp[i];
    }
    return inv_cnt;

}
int mergesort(vector<int>&arr,vector<int>&temp,int left, int right)
{
    int mid,inv_cnt=0;
    if(right>left)
    {
        mid=(left+right)/2;
        inv_cnt+=mergesort(arr,temp,left,mid);
        inv_cnt+=mergesort(arr,temp,mid+1,right);
        inv_cnt+=merge(arr,temp,left,mid+1,right);
    }
    return inv_cnt;
    
}

//5.Buy and sell stocks all variations
   //* buy and sell stocks I - one transaction max profit
int maxp(vector<int>prices)
{
    int ans=0;
    int minp=prices[0];
    for(auto it : prices)
    {
        ans=max(ans,it-minp);
        minp=min(minp,it);

    }
    return ans;
}
// buy and sell stocks II - as many transactions possible
int maxp(vector<int>prices)
{
    int ans=0;
    for(int i=1;i<prices.size();i++)
    {
        if(prices[i]>prices[i-1]) ans+=prices[i]-prices[i-1];
    }
    return ans;
} //tc- O(n) sc-O(1)

// buy and sell stocks III - at most two transactions allowed
int maxProfit(vector<int>& price) {
       int b1=INT_MAX,b2=INT_MAX;
        int s1=0,s2=0;
        for(auto it : price)
        {
            b1=min(b1,it);
            s1=max(s1,it-b1);
            b2=min(b2,it-s1);
            s2=max(s2,it-b2);
            
        }
        return s2;
    }// tc -O(n) sc-O(1)

// buy and sell stocks IV - at most k transactions
int maxprofit(int k,vector<int>prices)
{
    if(k>=prices.size()/2)
    {
        // case of as many transactions possible - buy and sell stocks II
         int ans=0;
    for(int i=1;i<prices.size();i++)
    {
        if(prices[i]>prices[i-1]) ans+=prices[i]-prices[i-1];
    }
    return ans;
    }
    vector<int>sell(k+1);
    vector<int>hold(k+1,INT_MIN);
    for(auto it : prices)
    {
        for(int i=k;i>0;i--)
        {
            sell[i]=max(sell[i],hold[i]+it);
            hold[i]=max(hold,sell[i-1]-it);
        }
    }
    return sell[k];

}// tc- O(n*k) sc-O(n)

//6. rotate matrix
  void rotate(vector<vector<int>>& matrix) {
       int n=matrix.size();
        for(int i=0;i<n;i++) // creating transpose of matrix
        {
            for(int j=0;j<i;j++)
            {
                swap (matrix[i][j],matrix[j][i]);
            }
        }
        for(int i=0;i<n;i++) // reversing the rows of transpose of matrix
        {
            reverse(matrix[i].begin(),matrix[i].end());
        }
    } // TC -O(n^2) sc-O(1)


//DAY -3 ARRAYS/MATHS
//1. SEARCH in 2-D matrix - gfg variation
// here first integer in a row need not to be greater than last int of previous row
	int matSearch (vector <vector <int>> &mat, int N, int M, int X)
	{
	    
	    int i=0,j=M-1;
	    while(i<N && j>=0)
	    {
	        if(mat[i][j]==X)
	        {
	            return 1;
	        }
	        if(mat[i][j]>X) j--;
	        else i++;
	    }
	    return 0;
    } // TC-O(logn) sc-O(1)

    // leetcode version - 1st int of ith row > last int of (i-1)th row
      bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row=matrix.size();
        int col=matrix[0].size();
        int l=0,r=row*col-1;
        while(l<=r){
            int mid=l+(r-l)/2;
            int val=matrix[mid/col][mid%col];
            if(val==target) return true;
            if(val>target){
                r=mid-1;
            }
            else {
                l=mid+1;
            }
        }
        return false;
    } // tc-O(logn) sc-O(1)

   //2 Pow(x,n) we are using binary exponentiation 
    #define ll long long
    double myPow(double x, int n) {
     double ans=1.0;
        ll nn=n;
        if(nn<0) nn=-1*nn;
        while(nn)
        {
            if(nn%2==0) 
            {
                x=x*x;
                nn=nn/2;
            }
            else 
            {
                ans=ans*x;
                nn=nn-1;
            }
            
        }
        if(n<0) ans=double(1.0)/double(ans);
        return ans;
        
    } // tc-O(logn) sc-O(1)

    //3. majority element (N/2) times 
    // one approach is to use hashmap tc-O(nlogn) sc-O(n)
    // further optimised solution is to sort array and return mid element tc-O(nlogn) sc-O(1)
    // further optimisation is to use moore voting algorithm
     int majorityElement(vector<int>& nums) {
        int count=0;
        int ele=0;
        for(auto it : nums)
        {
            if(count ==0) ele=it;
            if(it==ele) count+=1;
            else count-=1;
        }
        return ele; // moore voting algo

    } // tc -O(n) sc-O(1)

    //4.majority element II (N/3) times
    // moore voting algo for 2 candidates as maxm 2 candidates can be possible for having count > (N/3)
     vector<int>maj_ele(vector<int>&nums)
     {
         int n=nums.size();
         int num1=-1,num2=-1,count1=0,count2=0,i;
         for(auto ele : nums)
         { 
             if(ele==num1) count1++;
             else if(ele==num2) count2++;
             else if(count1==0)
             {
                 num1=ele;
                 count1=1;
             }
             else if(count2==0)
             {
                 num2=ele;
                 count2=1;

             }
             else count1--,count2--;

         }
         count1=count2=0;
         for(auto x : nums)
         {
             if(x==num1) count1++;
             else if(x==num2) count2++;
         }
         int n1,n2;
         if(count1>(n/3)) n1=num1;
         if(count2 > (n/3)) n2=num2;
         return {n1,n2};
     }

     //5.unique paths (grid based prblm)
     // first approach - reccursion
     // further can optimise with memoization 
     int countpaths(int i,int j,int m,int n,vector<vector<int>>&t)
     {
         if(i==m-1 && j==n-1) return 1;
         if(i>m || j>n) return 0;
         if(t[i][j]!=-1) return t[i][j];
         return t[i][j]=countpaths(i+1,j,m,n,t)+countpaths(i,j+1,m,m,t); // memoized dp
     }
     // 2-d DP
     int uniquepaths(int m ,int n)
     {
         vector<vector<int>>t(m,vector<int>(n,1)); // 2-d dp initialised with 1 as there are no obstacles so there must be at least 1 path from every index
         for(int i=1;i<m;i++)
         {
             for(int j=1;j<n;j++)
             {
                 t[i][j]=t[i-1][j]+t[i][j-1];

             }
         }
         return t[m-1][n-1];
          // TC- O(m*n) SC-O(m*n)
     }
     // 1-d dp
     int uniquepaths(int m,int n)
     {
         vector<int>t(n,1); // n is  no of columns
         for(int i=1;i<m;i++)
         {
             for(int j=1;j<n;j++)
             {
                 t[j]+=t[j-1];
             }
         }
         return t[n-1];
     } // TC-O(m*n) SC-O(n)

     // best optimal solution using P&C
     int uniquepaths(int m,int n)
     {
         int N=m+n-2;
         int r=n-1;
         double ans=1;
         for(int i=1;i<=r;i++)
         {
             res=res*(N-r-i)/i;
         }
         return (int)res;
     } //TC- O(n) SC-O(1) best intuition

    //6 . reverse pairs little tough
        int merge(vector<int>&nums,int low,int mid, int high)
{
   int cnt=0;
        int j=mid+1;
        for(int i=low;i<=mid;i++)
        {
            while(j<=high && nums[i]>2*nums[j])
            {
                j++;
            }
            cnt+=j-(mid+1);
        }
        vector<int>temp;
        int left=low,right=mid+1;
        while(left<=mid && right<=high)
        {
            if(nums[left]<=nums[right])
            {
                temp.push_back(nums[left++]);
            }
            else temp.push_back(nums[right++]);
        }
        while(left<=mid) temp.push_back(nums[left++]);
        while(right<=high ) temp.push_back(nums[right++]);
        return cnt;


}
    
    int mergesort(vector<int>&nums,int low,int high)
    {
        if(low>high) return 0;
        int mid=(low + high)/2;
        int inv=mergesort(nums,low,mid);
        inv+=mergesort(nums,mid+1,high);
        inv+=merge(nums,low,mid,high);
        return inv;
            
    }
    
    int reversePairs(vector<int>& nums) {
        return mergesort(nums,0,nums.size()-1);
    }

// DAY 4
// HASHING
//1. two sum
vector<int>twosum(vector<int>&nums,int target)
{
    unordered_map<int,int>mp;
    for(int i=0;i<nums.size();i++)
    {
        int p=target-nums[i];
        if(mp.find(p)!=mp.end()) return{i,mp[p]};
        mp[nums[i]]=i;

    }
    return {};
} //TC-O(n) SC-O(n)

//2. 4sum leetcode
    vector<vector<int>>res;
        if(nums.empty()) return res;
        int n=nums.size();
        sort(begin(nums),end(nums));
        for(int i=0;i<n;i++)
        {
            for(int j=i+1;j<n;j++)
            {
                int t2=target-nums[i]-nums[j];
                int left=j+1,right=n-1;
                while(left<right)
                {
                    int twosum=nums[left]+nums[right];
                    if(twosum < t2) left++;
                    else if(twosum >t2) right++;
                    else 
                    {
                        vector<int>temp(4,0);
                        temp[0]=i,temp[1]=j,temp[2]=left,temp[3]=right;
                        res.push_back(temp);
                        while(left < right && nums[left]==temp[2]) left++;
                        while(left < right && nums[right]==temp[3]) right--;
                        
                    }
                }
                while(j+1 < n && nums[j+1]==nums[j]) ++j;
            }
            while(i+1 < n && nums[i+1]==nums[i]) ++i;
        }
        return res;
    } //tc- O(n^3) sc-O(1)

    // 3. Longest consecutive subsequence
    int functn(vector<int>nums)
    {
        int ans=0;
       unordered_set<int>s{begin(nums),begin(end));
       for(auto it : nums)
       {
           if(s.count(it-1)) continue;
           int len=1;
           while(s.count(++it)) ++len;
           ans=max(ans,len);
       }
       return ans;


    } // TC-O(n) sc-O(n)

  //4. largest subarray with 0 sum
   int maxLen(vector<int>&A, int n)
    {   
        // Your code here
        unordered_map<int,int>mp;
        int sum=0;
        int maxl=0;
        for(int i=0;i<A.size();i++)
        {
            sum+=A[i];
            if(sum==0) maxl=i+1;
            else 
            {
                if(mp.find(sum)!=mp.end()) maxl=max(maxl,i-mp[sum]);
                else mp[sum]=i;
            }
        }
        return maxl;
    } // TC-O(n) SC-O(n)

    //5. count subarrays as XOR m
    int solve(vector<int>arr,int m)
    {
        map<int,int>mp;
        int ctr=0;
        int xorr=0;
        for(auto it : arr)
        {
            xorr=xorr^it;
            if(xorr==m) ctr++;
            if(mp.find(xorr^m)!=mp.end()) ctr+=mp[xorr^m];
            mp[xorr]+=1;

        }
        return ctr;
    } //TC-O(n) sc-O(n)

    // 6. longest substring without repeating character
     int lengthOfLongestSubstring(string s) {
        vector<int> m(256,-1);
        int l=0,r=0;
        int n=s.size();
        int len=0;
        while(r<n){
            if(m[s[r]]!=-1)
                l=max( m[s[r]]+1,l);
           m[s[r]]=r;
           len=max(len,r-l+1);
           r++;
        }
        return len;
        
        
    } // TC- O(n) sc-O(N)

    //DAY 6
    // LINKED LIST
    //1. reverse link list
     ListNode* reverseList(ListNode* head) {
        ListNode* prev=NULL;
        while(head)
        {
            ListNode* temp=head->next;
            head->next=prev;
            prev=head;
            head=temp;
        }
        return prev;
    } // tc- O(n) sc-o(1)

    //2.find middle of linked list
    listnode* middlenode( Listnode* head  )
    {
        if(!head) return NULL;
        ListNode* s=head;
        ListNode* f=head;
        while(fast!=NULL && fast->next!=NULL) s=s->next,f=f->next->next;
        return s;
    } // TC-O(n/2) sc-O(1)

    // 3. merge two sorted linked lists
    ListNode* merge2lists(ListNode* l1,ListNode*l2)
    {
        ListNode* dummy=new ListNode(0);
        ListNode* temp=dummy;
        while(true)
        {
            if(!l1) 
            {
                temp->next=l2;
                break;
            }
            if(!l2) temp->next=l1,break;
            if(l1->val < l2->val)
            {
                temp->next=l1;
                l1=l1->next;
            }
            else 
            {
                temp->next=l2;
                l2=l2->next;
            }
            temp=temp->next;

        
            
        }
        return dummy->next;
    } // TC-O(N1+N2) SC-O(1)

    //4. remove nth node from end of linked list
    ListNode* removenthendnode (ListNode* head,int n)
    {
        if(!head) return nullptr;
        ListNode* dummy=new ListNode(0);
        dummy->next=head;
        ListNode* s=dummy;
        ListNode* f=dummy;
        for(int i=1;i<=n;i++) f=f->next;
        while(f->next!=nullptr)
        {
            s=s->next,f=f->next;
        }
        s-next=s->next->next;
        return dummy->next;
    } // TC-O(n) sc-O(1)

    //5.delete a given node (no head given)
     void deleteNode(ListNode* node) {
        node->val=node->next->val;
        node->next=node->next->next;
        
    } // tc-O(1) sc-O(1)

    //6. add two nos given as linked list and return its sum as linked list
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2)
    {
      ListNode* dummy=new ListNode(0);
      ListNode* temp=dummy;
      int carry=0;
      while(l1!=NULL || l2!=NULL || carry)
      {
          int sum=0;
          if(l1) sum+=l1->val,l1=l1->next;
          if(l2) sum+=l2->val,l2=l2->next;
          sum+=carry;
          carry=sum/10;
          ListNode* node=new ListNode(sum%10);
          temp->next=mode;
          temp=temp->next;
      }  
      return dummy->next;
    }// TC-O(max(m,n)) sc-O(n)

// DAY 6 Linked List
// 1. find intersection point of two linked list
ListNode* intersection(ListNode* head1, ListNode* head2)
{
    ListNode* d1=head1;
    ListNode* d2=head2;
    while(d1 != d2)
    {
        d1= d1? d1->next : head2;
        d2=d2? d2->next : head1;
    }
    return d1;
} // TC-O(2m) sc-O(1)

//2. detect cycle in linked list
bool iscycle(ListNode* head)
{
    ListNode*s =head;
    ListNode* f=head;
    while(f && f->next)
    {
        s=s->next,f=f->next->next;
        if(s==f) return true;

    }
    return false;
} // tc-O(n) sc-O(1)

 //3. reverse link list in group of k
   ListNode* reverseKGroup(ListNode* head, int k) {
        if(head==NULL || k==1)
            return head;
        ListNode *dummy=new ListNode(0);
        dummy->next=head;
        int ctr=0;
        
        ListNode *cur=dummy,*pre=dummy,*nex=dummy;
        while(cur->next!=NULL){
            cur=cur->next;
            ctr++;
        }
        while(ctr>=k){
            cur=pre->next;
            nex=cur->next;
            for(int i=1;i<k;i++){
                cur->next=nex->next;
                nex->next=pre->next;
                pre->next=nex;
                nex=cur->next;
            }
            pre=cur;
            ctr-=k;
            
            
        }
        return dummy->next;
        
       
        
    } // tc-O(n) sc-O(1)

    //4.check for palindromic linked list
     bool isPalindrome(ListNode* head) {
        ListNode* slow = head;
    ListNode* fast = head;

    while (fast && fast->next) {
      slow = slow->next;
      fast = fast->next->next;
    }

    if (fast)
      slow = slow->next;
    slow = reverseList(slow);

    while (slow) {
      if (slow->val != head->val)
        return false;
      slow = slow->next;
      head = head->next;
    }

    return true;
    } // tc-O(3n/2) sc-O(1)
    
  ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;

    while (head) {
      ListNode* next = head->next;
      head->next = prev;
      prev = head;
      head = next;
    }

    return prev;
  }
 
  //5.find starting point of linked list
   ListNode *detectCycle(ListNode *head) {
          ListNode* s=head;
        ListNode* f=head;
        ListNode*temp=NULL;
        while(f && f->next)
        {
            s=s->next;
            f=f->next->next;
            if(s==f){
                s=head;
                while(s!=f)
                {
                    s=s->next;
                    f=f->next;
                }
                return s;
            }
        }
        return nullptr;   
  
    } // tc-O(n) sc-O(1)

    //6.flatten the linked list
      Node* merge(Node *a,Node *b)
    {
        Node * temp=new Node(0);
        Node *res=temp;
        while(a!=NULL && b!=NULL)
        {
            if(a->data < b->data){
                temp->bottom=a;
                temp=temp->bottom;
                a=a->bottom;
            }
            else 
            {
                 temp->bottom=b;
                temp=temp->bottom;
                b=b->bottom;
            }
        }
        if(a) temp->bottom=a;
        else temp->bottom =b;
        return res->bottom;
    }
Node *flatten(Node *root)
{
  
  if(root==NULL || root->next==NULL) return root;
  root->next=flatten(root->next);
  root =merge(root,root->next);
  return root;
} // tc=O(no of total nodes) sc-O(1)

//7. rotate a linked list k times to right
ListNode* rotateRight(ListNode* head, int k) {
        if(!head || !head->next|| !k) return head;
        ListNode* temp=head;
        int length=1;
        while(temp->next!=NULL)
        {
            temp=temp->next; // temp will be pointing to tail of list
            ++length;
        }
        temp->next=head;
        
        int n=length- (k%length);
        for(int i=0;i<n;i++)
        {
            temp=temp->next;
        }
        ListNode* newh=temp->next;
        temp->next=NULL;
        return newh;
       
        
    } // TC-O(n) sc-O(1)


// DAY 7 TWO POINTER
  //1.clone the linked list with next and random pointer
    unordered_map<Node*,Node*>mp;// mapping orig nodes to copied nodes
    Node* copyRandomList(Node* head) {
        if(!head) return NULL;
        if(mp.count(head)) return mp[head];
        Node* newnode=new Node(head->val);
        mp[head]=newnode;
        newnode->next=copyRandomList(head->next);
        newnode->random=copyRandomList(head->random);
     
    return newnode;
    } // TC-o(2N) SC-o(1)

    //2.3 sum problem
    vector<vector<int>>3sum(vector<int>&nums)
    {
        sort(nums.begin(),nums.end());
        vector<vector<int>>ans;
        int n=nums.size();
        for(int i=0;i<n;i++)
        {
            if(i==0 || (i>0 && nums[i]!=nums[i-1]))
            {
                int l=i+1,r=n-1,sum=0-nums[i];
                while(l<r)
                {
                    if(nums[l]+nums[r]==sum)
                    {
                        ans.push_back({nums[i],nums[l],nums[r]});
                        while(l<r && nums[l]==nums[l+1]) l++;
                        while(l<r && nums[r-1]==nums[r]) r--;
                        l++,r--;
                        
                    }
                    else if(nums[l]+nums[r]<sum) l++;
                    else r--;
                }
            }
        }
        return ans;
        
    } // tc-O(n*n) sc-O(m) osc-O(1) m = no of unique triplets 

    //3. trapping rain water
    int traprainwater(vector<int>&ht)
    {
        int n=ht.size();
        int l=0,r=n-1;
        int water=0;
        int lmax=0,rmax=0;
        while(l<r)
        {
            if(ht[l]>=hr[r]) // checking for ht[l]
            {
              if(ht[l]>=lmax) lmax=ht[l]; // cant store water in this case so updating lmax to ht [l]
              else water+=lmax-ht[l]; // storing water
              l++;
            }
            else 
            {
                if(ht[r]>=rmax) rmax=ht[r];
                else water+=rmax-ht[r];
                r--;
            }
        }
        return water;
    } // tc-O(n) sc-O(1)

    //4. remove duplicates from sorted array
    int removeduplicates(vector<int>&arr)
    {
        int n=arr.size();
        if(n==0) return;
        int i=0,j=1;
        for(int j=1;j<n;j++)
        {
            if(arr[j]!=arr[i])
            {
                i++;
                arr[i]=arr[j];
            }
        }
        return i+1;
    } // tc-O(n) sc-O(1)

    //5. max consecutive ones
     int findMaxConsecutiveOnes(vector<int>& nums) {
        int cnt=0,ans=0;
        for(int i=0;i<nums.size();i++)
        {
            if(nums[i]==1)
            {
                cnt++;
                ans=max(ans,cnt);
            }
            else cnt=0;
        }
        return ans;
    } // tc-O(n) sc-O(1)


// DAY 8
// GREEDY
 // 1. maxm meetings in a room
 static bool check(pair<int,int>a,pair<int>b)
 {
     return a.second < b.second;
 }

 int maxmmeetings(int start[],int end[],int n) 
 {
     vector<pair<int,int>>p;
     int ans=1;
     for(int i=0;i<n;i++)
     {
         p.push_back({start[i],end[i]});
     }
     sort(p.begin(),p.end(),check);
     int i=0;
     for(int j=1;j<n;j++)
     {
         if(p[j].first> p[i].seond) ans++,i=j;
     }
     return ans;



 } //tc-O(n) + O(nlogn) + O(n) sc-O(n)

 //2. minimum platforms
 int minplatforms(int arr[],int dep[],int n)
 {
     sort(arr,arr+n);
     sort(dep,dep+n);
     int i=1,j=0;
     int plat=1,ans=1;
     while(i< n && j<n)
     {
         if(arr[i]<=dep[j]) plat++,i++;

         else if(arr[i]>dep[j]) plat--,j++;
         ans=max(ans,plat); 
     }
     return ans;
 } //tc-O(2n*logn) + O(n) sc- O(1)

 //3. job scheduling prblm
// 
/* struct Job { 
    int id;	 // Job Id 
    int dead; // Deadline of job 
    int profit; // Profit if job is over before or on deadline 
};
*/
static bool comp(Job a , Job b)
{
    return a.profit > b.profit;
}
 vector<int>maxprofit(Job arr[],int n)
 {
    sort(arr,arr+n,comp);
    int maxi=arr[0].dead;
    for(int i=1;i<n;i++) maxi=max(maxi,arr[i].dead);
    vector<int>slot(maxi,-1);
    int ctr=0,p=0;
    for(int i=0;i<n;i++)
    {
        for(int j=arr[i].dead,j>=0;j--)
        {
            if(slot[j]==-1)
            {
                slot[j]=i;
                ctr++;
                p+=arr[i].profit;
                break;
            }
        }
    }
   return {ctr,p};
 } // tc-O(nlogn) + O(n*m) sc-O(m) where m is max of all deadlines

 //4. Fractional knapsack
 /*
struct Item{
    int value;
    int weight;
};
*/
static bool comp(Item a, Iteam b)
{
    double r1=(double)a.value / (double)a.weight;
    duble r2=(double)b.value/(double)b.weight;
    return r1>r2;
}

 int maxmprofit(Item arr[],int n,int W)
 { 
   sort(arr,arr+n,comp);
   double ans=0;
   int curwt=0;
   for(int i=0;i<n;i++)
   {
       if(curwt + arr[i].weight<=W) curwt+=arr[i].weight, ans+=arr[i].value;
       else 
       {
           int rem=W- curwt;
           ans+=(arr[i].value/(double)arr[i].weight)*(double)rem;
           break;
       }
   }
   return ans;

   

 } // tc-O(nlogn) + O(n) sc-o(1)

 //5. minm coins
  vector<int> minPartition(int N)
    {
        // code here
        vector<int>ans;
        vector<int>coins{1,2,5,10,20,50,100,200,500,2000};
        int n=10;
        for(int i=n-1;i>=0;i--)
        {
            while(N >=coins[i]) 
            {
                N-=coins[i];
                ans.push_back(coins[i]);
            }
        }
        return ans;
    } // tc-O(N) sc-O(1)

//DAY9
// RECCURSION
 //1. print subset sum in increasing order
 vector<int>ans;

 void solve(vector<int>arr,int n,int indx,int sum)
 {
     if(indx==n) return;
     for(int i=indx;i<n;i++)
     { 
         ans.push_back(sum+arr[i]);
         solve(arr,n,i+1,sum+arr[i]);
         
     }
 }
    vector<int> subsetSums(vector<int> arr, int N)
    {
      
        
        ans.push_back(0);
        solve(arr,N,0,0);
        return ans;
    } // tc-O(2^N) SC-O(2^N) 

    //2.subset  II print subsets (result must not contains duplicate subsets )
    vector<vector<int>>ans;
    void solve(vector<int>&nums,bector<int>&temp,int n,int indx)
    {
        ans.push_back(temp);
        for(int i=indx;i<n;i++)
        {
            if(i!=indx && nums[i]==nums[i-1]) continue;
            temp.push_back(nums[i]);
            solve(nums,temp,n,i+1);
            temp.pop_back();
        }

    }
    vector<vector<int>>subset2(vetcor<int>&nums,int n)
    {
        vector<int>temp;
        sort(nums.begin(),nums.end());
        solve(nums,temp,n,0);
        return ans;

        
    } // TC-O(nlogn * 2^n) sc-O(n)

    //3. combination sum
    vector<vector<int>>ans;
    void solve(vector<int>&arr,vector<int>&temp,int &target,int indx)
    {
        if(indx==arr.size())
        {
            if(target==0)
            {
            ans.push_back(temp);
            }
            return;
        }
        if(arr[indx]<=target) // staying at same index and checking for valid set
        {
            temp.push_back(arr[indx]);
            solve(arr,temp,target-arr[indx],indx);
        }
        solve(arr,temp,target,indx+1); // after staying part is done we are moving to next of index and checking

    }


    vector<vector<int>>combsum(vector<int>&arr,int target)
    {
        vector<int>temp;
        solve(arr,temp,target,0);
        return ans;
    } // tc-O(k* 2^k) sc-O(k*x) k is avg length of combinations , x = assumed no of combinations

    //4. combination sum II
     vector<vector<int>>ans;
    void solve(vector<int>&arr,vector<int>&temp,int &target,int indx)
    {
        if(indx==arr.size())
        {
            if(target==0)
            {
            ans.push_back(temp);
            }
            return;
        }
       for(int i=0;i<arr.size(),i++)
       {
           if(i>indx && arr[i]==arr[i-1]) continue;
           if(arr[i]>target) break;
           temp.push_back(arr[i]);
           solve(arr,temp,target-arr[i],i+1);
           temp.pop_back();
       }

    }


    vector<vector<int>>combsum2(vector<int>&arr,int target)
    {
        vector<int>temp;
        sort(arr.begin(),arr.end());
        solve(arr,temp,target,0);
        return ans;
    } // tc-O(k* 2^k) sc-O(k*x) k is avg length of combinations , x = assumed no of combinations

    //5.palindrome partitioning
    vector<vector<string>> partition(string s) {
        vector<vector<string>>ans;
        
        solve(s,ans,{},0);
        return ans;
        
        
    } // tc~O(n^3) sc-O(k*x)
    void solve(string &s,vector<vector<string>>&ans,vector<string>temp,int indx) // dfs kinda
    {
        if(indx==s.length()) 
        {
            ans.push_back(temp);
            return;
        }
        for(int i=indx;i<s.length();i++)
        {
            if(ispalindrome(s,indx,i))
            {
                temp.push_back(s.substr(indx,i-indx+1));
                solve(s,ans,temp,i+1);
                temp.pop_back();
            }
        }
    }
    bool ispalindrome(string &s,int l,int r)
    {
        while(l<r)
        {
            if(s[l++]!=s[r--]) return false;
        }
        return true;
    }

    //6. kth permutational sequence
     string getPermutation(int n, int k) {
        int fact=1;
        vector<int>num;
        for(int i=1;i<n;i++)
        {
            fact=fact*i;
            num.push_back(i);
        } // calc n-1 fact
        num.push_back(n);
        k=k-1;
        string ans="";
        while(true)
        {
            ans=ans+to_string(num[k/fact]);
            num.erase(num.begin()+ k/fact);
            if(num.size()==0)
            {
                break;
            }
            k=k%fact;
            fact=fact/num.size();
        }
        return ans;
        
    } // tc-O(n^2) sc-O(n)

// DAY 10
// Reccursion and Backtracking
//1. print all permutations of array/string
  void solve(vector<int>& nums,vector<vector<int>>&ans,int indx)
    {
        if(indx==nums.size())
        {
            ans.push_back(nums);
            return;
            
        }
        for(int i=indx;i<nums.size();i++)
        {
            swap(nums[indx],nums[i]);
            solve(nums,ans,indx+1);
            swap(nums[indx],nums[i]);
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>>ans;
        solve(nums,ans,0);
        return ans;
    } // TC-O(n! * n) osc-O(1)

    // 2. N queens
 void dfs(int n, int i, vector<bool>& cols, vector<bool>& diag1, vector<bool>& diag2, vector<string>& board,vector<vector<string>>& ans)
  {
    if (i == n) {
      ans.push_back(board);
      return;
    }

    for (int j = 0; j < n; ++j) {
      if (cols[j] || diag1[i + j] || diag2[j - i + n - 1])
        continue;
      board[i][j] = 'Q';
      cols[j] = diag1[i + j] = diag2[j - i + n - 1] = true;
      dfs(n, i + 1, cols, diag1, diag2, board, ans);
      cols[j] = diag1[i + j] = diag2[j - i + n - 1] = false;
      board[i][j] = '.';
    }
  }
    vector<vector<string>> solveNQueens(int n) {
           vector<vector<string>> ans;
    vector<string> board(n, string(n, '.'));
    vector<bool> cols(n);
    vector<bool> diag1(2 * n - 1);
    vector<bool> diag2(2 * n - 1);

    dfs(n, 0, cols, diag1, diag2, board, ans);

    return ans;
    }

    //3. sudoku solver
     bool isvalid(vector<vector<char>>&board,int row,int col,char c)
    {
        for(int i=0;i<9;i++) 
        {
            if(board[i][col]==c) return false;
            if(board[row][i]==c) return false;
            if(board[3*(row/3) + i/3][3*(col/3)+ i%3]==c) return false;
            
         }
        return true;
    }
    bool solve(vector<vector<char>>&board)
    {
        for(int i=0;i<board.size();i++)
        {
            for(int j=0;j<board[0].size();j++)
            {
                if(board[i][j]=='.')
                {
                    for(char c='1';c<='9';c++)
                    {
                        if(isvalid(board,i,j,c))
                        {
                            board[i][j]=c;
                            if(solve(board)==true) return true;
                            else board[i][j]='.';

                        }
                    }
                    return false;

                }
            }
        }
        return true;

    }
    void solveSudoku(vector<vector<char>>& board) {
        solve(board);
        
    }

    //4. m coloring prblm
    bool isvalid(int indx,int col[],bool graph[101][101],int V,int i)
    {
        for(int j=0;j<V;J++)
        {
            if(j!=indx && graph[j][indx]==1 && col[j]==i) return false;
            return true;
        }
    }
    bool solve(int indx,bool graph[101][101],int col[],int V,int m)
    {
        if(indx==V) return true;
        for(int i=0;i<m;i++)
        {
            if(isvalid(indx,col,graph,V,i))
            {
                col[indx]=i;
               if(solve(indx+1,graph,col,V,m)) return true;
                col[indx]=0;

            }
        }
        return false;
    }
    bool mcolor(bool graph[101][101],int m,int V) // V is no of vertices
    {
        int col[V]={0};
        if(solve(0,graph,col,V,m)) return true;
        return false;
    } // TC-O(V^m), sc-O(n) fr colr array, osc-O(n) for recccursion tree depth

    //5. rat in a maze
    
    vector<string>ans;
    int dx[4]={0,1,0,-1};
    int dy[4]={-1,0,1,0};
    char adddirect(int i,int j)
    {
        if(i==1 && j==0) return 'D';
        if(i==-1 && j==0) return 'U';
        if(i==0 && j==1) return 'R';
        if(i==0 && j==-1) return 'L';
    }
    
    void solve(vector<vector<int>> &m,int i,int j,int n,string s)
    {
        if(i<0 || j<0||i>=n || j>=n||m[i][j]!=1) return;
        if(i==n-1 && j==n-1)
        {
            ans.push_back(s);
            return;
        }
        m[i][j]=2;
        for(int k=0;k<4;k++)
        {
            int nx=i + dx[k];
            int ny=j+dy[k];
            s.push_back(adddirect(dx[k],dy[k]));
            solve(m,nx,ny,n,s);
            s.pop_back();
        }
        m[i][j]=1;
    }
    vector<string> findPath(vector<vector<int>> &m, int n) {
      
        solve(m,0,0,n,"");
        sort(ans.begin(),ans.end());
        return ans;
    } // tc-O(4^(n^2)) sc-O(n*n);

    //6. word break
      int wordBreak(string s, unordered_set<string> &dict) {
        //code here
         if(dict.size()==0) return false;
        
        vector<bool> dp(s.size()+1,false);
        dp[0]=true;
        
        for(int i=1;i<=s.size();i++)
        {
            for(int j=i-1;j>=0;j--)
            {
                if(dp[j])
                {
                    string word = s.substr(j,i-j);
                    if(dict.find(word)!= dict.end())
                    {
                        dp[i]=true;
                        break; //next i
                    }
                }
            }
        }
        
        return dp[s.size()];
    } // TC-O(n*n) sc-O(n)
