# 排序、贪心

## 归并排序代码模板

### Golang模板

```go
func mergeSort(array []int, left, right int) {
  if right <= left {
    return
  }
  mid := (left + right) >> 1
  mergeSort(array, left, mid)
  mergeSort(array, mid + 1, right)
  merge(array, left, mid, right)
}

func merge(arr []int, left, mid, right int) {
  temp := make([]int, right - left + 1)
  i, j, k := left, mid + 1, 0
  for ;i <= mid && j <= right; k++ {
    if arr[i] <= arr[j] {
      temp[k] = arr[i]
      i++
    } else {
      temp[k] = arr[j]
      j++
    }
  }
  for ;i <= mid; k++ {
    temp[k] = arr[i]
    i++
  }
  for ;j <= right; k++ {
    temp[k] = arr[j]
    j++
  }
  for i := range temp {
    arr[left + i] = temp[i]
  }
}
```



### C++模板

```c++
void mergeSort(vector<int> &nums, int left, int right) {
  if (left >= right) return;
  int mid = left + (right - left) / 2;
  mergeSort(nums, left, mid);
  mergeSort(nums, mid+1, right);
  merge(nums, left, mid, right);
}

void merge(vector<int> &nums, int left, int mid, int right) {
  vector<int> tmp(right-left+1);
  int i = left, j = mid+1, k = 0;
  while (i <= mid && j <= right) {
    tmp[k++] = nums[i] < nums[j] ? nums[i++] : nums[j++];
  }
  while (i <= mid) tmp[k++] = nums[i++];
  while (j <= right) tmp[k++] = nums[j++];
  for (i = left, k = 0; i <= right;) nums[i++] = tmp[k++];
}
```



### java模板

```java
// Java
public static void mergeSort(int[] array, int left, int right) {
  if (right <= left) return;
  int mid = (left + right) >> 1; // (left + right) / 2
  mergeSort(array, left, mid);
  mergeSort(array, mid + 1, right);
  merge(array, left, mid, right);
}

public static void merge(int[] arr, int left, int mid, int right) {
  int[] temp = new int[right - left + 1]; // 中间数组
  int i = left, j = mid + 1, k = 0;
  while (i <= mid && j <= right) {
    temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
  }
  while (i <= mid)   temp[k++] = arr[i++];
  while (j <= right) temp[k++] = arr[j++];
  for (int p = 0; p < temp.length; p++) {
    arr[left + p] = temp[p];
  }
  // 也可以用 System.arraycopy(a, start1, b, start2, length)
}
```



### python模板

```python
def mergesort(nums, left, right):
  if right <= left:
    return
  mid = (left+right) >> 1
  mergesort(nums, left, mid)
  mergesort(nums, mid+1, right)
  merge(nums, left, mid, right)
  
def merge(nums, left, mid, right):
  temp = []
  i = left
  j = mid+1
  while i <= mid and j <= right:
    if nums[i] <= nums[j]:
      temp.append(nums[i])
      i +=1
    else:
      temp.append(nums[j])
      j +=1
  while i<=mid:
        temp.append(nums[i])
        i +=1
  while j<=right:
    temp.append(nums[j])
    j +=1
  nums[left:right+1] = temp
```



## 快速排序代码模板

### Golang模板

```go
// 待实现
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
  return nil
}

func sortArray (nums []int) []int {
  quickSort(nums, 0, len(nums) - 1)
  return nums
}

func quickSort(arr []int, l, r int) {
  if (l >= r) {
    return
  }
  pivot := partition(arr, l, r)
  quickSort(arr, l, pivot)
  quickSort(arr, pivot + 1, r)
}

func partition(a []int, l, r int) int {
  pivot := l + rand.Intn(r - l + 1)
  pivotVal := a[pivot]
  for l <= r {
    for a[l] < pivotVal {
      l++ 
    }
    for (a[r] > pivotVal) {
      r-- 
    }
    if (l == r) {
      break 
    }
    if (l < r) {
      a[l], a[r] = a[r], a[l]
      l++
      r--
    } 
  }
  return r
}
```



### C++模板

```c++
class Solution {
  public:
  	vector<int> sortArray(vector<int>& nums) {
      quickSort(nums, 0, nums.size() - 1);
      return nums;
    }
  
  void quickSort(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int pivot = partition(arr, l, r);
    quickSort(arr, l, pivot);
    quickSort(arr, pivot + 1, r);
  }
  
  int partition(vector<int>& a, int l, int r) {
    int pivot = l + rand() % (r - l + 1);
    int pivotVal = a[pivot];
    while (l <= r) { 
      while (a[l] < pivotVal) l++;
      while (a[r] > pivotVal) r--;
      if (l == r) break;
      if (l < r) {
        int temp = a[l]; a[l] = a[r]; a[r] = temp;
        l++; r--;
      }
    }
    return r;
  }
};
```



### java模板

```java
class Solution {
  public int[] sortArray(int[] nums) {
    quickSort(nums, 0, nums.length - 1);
    return nums;
  }
  
  public static void quickSort(int[] arr, int l, int r) {
    if (l >= r) return;
    int pivot = partition(arr, l, r);
    quickSort(arr, l, pivot);
    quickSort(arr, pivot + 1, r);
  } 
  
  static int partition(int[] a, int l, int r) {
    int pivot = l + (int)(Math.random() * (r - l + 1));
    int pivotVal = a[pivot];
    while (l <= r) {
      while (a[l] < pivotVal) l++;
      while (a[r] > pivotVal) r--;
      if (l == r) break;
      if (l < r) {
        int temp = a[l]; a[l] = a[r]; a[r] = temp;
        l++; r--;
      }
    }
    return r;
  }
}
```



### python模板

```python
class Solution:
  def sortArray(self, nums: List[int]) -> List[int]:
    self.quickSort(nums, 0, len(nums) - 1)
    return nums
  def quickSort(self, arr, l, r):
    if l >= r:
      return
    pivot = self.partition(arr, l, r)
    self.quickSort(arr, l, pivot)
    self.quickSort(arr, pivot + 1, r)
  def partition(self, a, l, r):
    pivot = random.randint(l, r)
    pivotVal = a[pivot]
    while l <= r:
      while a[l] < pivotVal:
        l += 1
      while a[r] > pivotVal: 
        r -= 1
      if l == r:
        break 
      if l < r:
        a[l], a[r] = a[r], a[l]
        l += 1
        r -= 1
        
    return r
```



## 堆排序代码模板

### Golang模板

```go
func heapify(array []int, length, i int) {
  left, right := 2 * i + 1, 2 * i + 2
  largest := i
  if left < length && array[left] > array[largest] {
    largest = left
  }
  if right < length && array[right] > array[largest] {
    largest = right
  }
  if largest != i {
    array[i], array[largest] = array[largest], array[i]
    heapify(array, length, largest)
  }
}

func heapSort(array []int) {
  if len(array) == 0 {
    return
  } 
  length := len(array)
  for i := length / 2-1; i >= 0; i-- {
    heapify(array, length, i)
  }
  for i := length - 1; i >= 0; i-- {
    array[0], array[i] = array[i], array[0]
  }
}
```



### C++模板

```c++
void heapify(vector<int> &array, int length, int i) {
  int left = 2 * i + 1, right = 2 * i + 2;
  int largest = i;
  if (left < length && array[left] > array[largest]) {
    largest = left;
  }
  if (right < length && array[right] > array[largest]) {
    largest = right;
  }
  if (largest != i) {
    int temp = array[i];
    array[i] = array[largest];
    array[largest] = temp;
    heapify(array, length, largest);
  }
  return ;
}

void heapSort(vector<int> &array) {
  if (array.size() == 0) return ;
  int length = array.size();
  for (int i = length / 2 - 1; i >= 0; i--)
    heapify(array, length, i);
  for (int i = length - 1; i >= 0; i--) {
    int temp = array[0]; array[0] = array[i]; array[i] = temp;
    heapify(array, i, 0);
  }
  return ;
}
```



### java模板

```java
// Java
static void heapify(int[] array, int length, int i) {
  int left = 2 * i + 1, right = 2 * i + 2；
  int largest = i;
  if (left < length && array[left] > array[largest]) {
    largest = left;
  }
  if (right < length && array[right] > array[largest]) {
    largest = right;
  }
  if (largest != i) {
    int temp = array[i];
    array[i] = array[largest];
    array[largest] = temp;
    heapify(array, length, largest);
  }
}

public static void heapSort(int[] array) {
  if (array.length == 0) return;
  int length = array.length;
  for (int i = length / 2-1; i >= 0; i-)
    heapify(array, length, i);
  for (int i = length - 1; i >= 0; i--) {
    int temp = array[0]; array[0] = array[i]; array[i] = temp;
    heapify(array, i, 0);
  }
}
```



### python模板

```python
def heapify(parent_index, length, nums):
  temp = nums[parent_index]
  child_index = 2*parent_index+1
  while child_index < length:
    if child_index+1 < length and nums[child_index+1] > nums[child_index]:
      child_index = child_index+1
    if temp > nums[child_index]:
      break
    nums[parent_index] = nums[child_index]
    parent_index = child_index
    child_index = 2*parent_index + 1
  nums[parent_index] = temp

def heapsort(nums):
  for i in range((len(nums)-2)//2, -1, -1): 
    heapify(i, len(nums), nums)
  for j in range(len(nums)-1, 0, -1):
    nums[j], nums[0] = nums[0], nums[j]
    heapify(0, j, nums)
```



# 动态规划



## 字典树、并查集

## 字典树模板

### Golang模板

```go
type Trie struct {
  root *node}
/** Initialize your data structure here. */
func Constructor() Trie {
  return Trie{root: &node{child: make(map[uint8]*node)}}
}
/** Inserts a word into the trie. */
func (this *Trie) Insert(word string) {
  this.find(word, true, true)
}
/** Returns if the word is in the trie. */
func (this *Trie) Search(word string) bool {
  return this.find(word, true, false)
}
/** Returns if there is any word in the trie that starts with the given prefix. */
func (this *Trie) StartsWith(prefix string) bool {
  return this.find(prefix, false, false)
}

type node struct {
  count int
  child map[uint8]*node
}

func (this *Trie) find(s string, exactMatch, insertIfNotExist bool) bool {
  curr := this.root;
  for i := 0; i < len(s); i++ {
    c := s[i]
    if _, ok := curr.child[c]; !ok {
      if !insertIfNotExist {
        return false
      }
      curr.child[c] = &node{child: make(map[uint8]*node)}
    }
    curr = curr.child[c]
  }
  if insertIfNotExist {
    curr.count++
  }
  return !exactMatch || curr.count > 0}

/**
* Your Trie object will be instantiated and called as such:
* obj := Constructor();
* obj.Insert(word);
* param_2 := obj.Search(word);
* param_3 := obj.StartsWith(prefix);
*/
```



### C++模板

```c++
//C/C++
class Trie {
  public: 
  /** Initialize your data structure here. */
  Trie() {
    root = new Node();
  }
  
  /** Inserts a word into the trie. */ 
  void insert(string word) {
    find(word, true, true);
  }
  
  /** Returns if the word is in the trie. */
  bool search(string word) {
    return find(word, true, false);
  }
  
  /** Returns if there is any word in the trie that starts with the given prefix. */
  bool startsWith(string prefix) {
    return find(prefix, false, false);
  }
  
  private:
  	struct Node {
      int count;
      unordered_map<char, Node*> child;
      Node(): count(0) {}
    };
  Node* root;
  
  bool find(const string& s, bool exact_match, bool insert_if_not_exist) {
    Node* curr = root;
    for (char c : s) {
      if (curr->child.find(c) == curr->child.end()) {
        if (!insert_if_not_exist) return false;
        curr->child[c] = new Node();
      }
      curr = curr->child[c];
    }
    if (insert_if_not_exist) curr->count++; 
    return exact_match ? curr->count > 0 : true;    }};
```



### java模板

```java
//Java
import java.util.HashMap;

class Trie {
  /** Initialize your data structure here. */ 
  public Trie() {
    root = new Node();
  }
  /** Inserts a word into the trie. */
  public void insert(String word) {
    find(word, true, true);
  }
  /** Returns if the word is in the trie. */
  public boolean search(String word) { 
    return find(word, true, false);
  }
  /** Returns if there is any word in the trie that starts with the given prefix. */
  public boolean startsWith(String prefix) {
    return find(prefix, false, false);
  }
  class Node {
    public int count;
    public HashMap<Character, Node> child;
    public Node() { count = 0; child = new HashMap<>(); }
  }
  Node root;
  
  boolean find(String s, boolean exact_match, boolean insert_if_not_exist) {
    Node curr = root;
    for (Character c : s.toCharArray()) {
      if (!curr.child.containsKey(c)) {
        if (!insert_if_not_exist) return false;
        curr.child.put(c, new Node());
      }
      curr = curr.child.get(c);
    }
    if (insert_if_not_exist) curr.count++;
    return exact_match ? curr.count > 0 : true; 
  }
}
```



### python模板

```python
# Python
class Trie:
  def __init__(self):
    """
    Initialize your data structure here.        
    """
    self.root = [0, {}]
    # [count, child]
  def insert(self, word: str) -> None:
    """        Inserts a word into the trie.        """
    self.find(word, True, True)
  def search(self, word: str) -> bool:
    """        Returns if the word is in the trie.        """
    return self.find(word, True, False)
  def startsWith(self, prefix: str) -> bool:
    """        Returns if there is any word in the trie that starts with the given prefix.        """
    return self.find(prefix, False, False)
  def find(self, s, exact_match, insert_if_not_exist):
    curr = self.root
    for ch in s:
      if ch not in curr[1]:
        if not insert_if_not_exist:
          return False
        curr[1][ch] = [0, {}]
      curr = curr[1][ch]
    if insert_if_not_exist:
      curr[0] += 1 
    return curr[0] > 0 if exact_match else True
```





## 并查集模板

### Golang模板

```go
// Golang
type DisjointSet struct {
  fa []int}func Construct(n int) DisjointSet {
  s := DisjointSet{fa: make([]int, n)}
  for i := 0; i < n; i++ {
    s.fa[i] = i
  }
  return s
}

func (s *DisjointSet) Find(x int) int {
  if s.fa[x] != x {
    s.fa[x] = s.Find(s.fa[x])
  }
  return s.fa[x]
}

func (s *DisjointSet) Join(x, y int) {
  x, y = s.Find(x), s.Find(y)
  if x != y {
    s.fa[x] = y
  }
}
```



### C++模板

```c++
//C/C++
class DisjointSet {
  public:   
  	DisjointSet(int n) {
      fa = vector<int>(n, 0);
      for (int i = 0; i < n; i++) fa[i] = i;
    } 
  int find(int x) {
    if (x == fa[x]) return x;
    return fa[x] = find(fa[x]);
  }
  void unionSet(int x, int y) {
    x = find(x), y = find(y);
    if (x != y) fa[x] = y;
  }
  
private:
  vector<int> fa;
};
```



### java模板

```java
// Java
class DisjointSet {
  public DisjointSet(int n) {
    fa = new int[n];
    for (int i = 0; i < n; i++) fa[i] = i;
  }
  public int find(int x) {
    if (x == fa[x]) return x;
    return fa[x] = find(fa[x]);
  } 
  public void unionSet(int x, int y) {
    x = find(x);
    y = find(y);
    if (x != y) fa[x] = y;
  }
  int[] fa;
};
```



### python模板

```python
class DisjointSet:
  def __init__(self, n):
    self.fa = [i for i in range(n)]
    
  def find(self, x):
    if x == self.fa[x]:
      return x
    self.fa[x] = self.find(self.fa[x])
    return self.fa[x]
  def unionSet(self, x, y):
    x = self.find(x)
    y = self.find(y)
    if x != y:
      self.fa[x] = y
```





# 图论算法

## Bellman-Ford 求最短路代码模板

### Golang模板

```go
// LeetCode 743
func networkDelayTime(times [][]int, n int, k int) int {
  dist := make([]int, n + 1)
  for i := 1; i <= n; i++ {
    dist[i] = 1e9
  }
  dist[k] = 0;
  for iteration := 1; iteration < n; iteration++ {
    updated := false
    for i := 0; i < len(times); i++ {
      x := times[i][0]
      y := times[i][1]
      z := times[i][2]
      if dist[y] > dist[x] + z {
        dist[y] = dist[x] + z
        updated = true
      }
    }
    if !updated {
      break
    }
  }
  ans := 0
  for i := 1; i <= n; i++ {
    if ans < dist[i] {
      ans = dist[i]
    }
  }
  if ans == 1e9 {
    ans = -1
  }
  return ans
}
```



### C++模板

```c++
// C/C++
// LeetCode 743
class Solution {
  public:
  	int networkDelayTime(vector<vector<int>>& times, int n, int k) {
      vector<int> dist(n + 1, 1e9);
      dist[k] = 0;
      for (int iteration = 1; iteration < n; iteration++) {
        bool updated = false;
        for (int i = 0; i < times.size(); i++) {
          int x = times[i][0];
          int y = times[i][1];
          int z = times[i][2];
          if (dist[y] > dist[x] + z) {
            dist[y] = dist[x] + z;
            updated = true;
          }
        }
        if (!updated) break;
      }
      int ans = 0;
      for (int i = 1; i <= n; i++)
        ans = max(ans, dist[i]);
      if (ans == 1e9) ans = -1;
      return ans;
    }
};
```



### java模板

```java
// Java
// LeetCode 743
class Solution {
  public int networkDelayTime(int[][] times, int n, int k) {
    int[] dist = new int[n + 1];
    for (int i = 1; i <= n; i++) dist[i] = (int)1e9;
    dist[k] = 0;
    for (int iteration = 1; iteration < n; iteration++) {
      boolean updated = false;
      for (int i = 0; i < times.length; i++) { 
        int x = times[i][0];
        int y = times[i][1];
        int z = times[i][2];
        if (dist[y] > dist[x] + z) {
          dist[y] = dist[x] + z;
          updated = true;
        }
      }
      if (!updated) break;
    }
    int ans = 0;
    for (int i = 1; i <= n; i++)
      ans = Math.max(ans, dist[i]);
    if (ans == 1e9) ans = -1;
    return ans;
  }
}
```



### python模板

```python
# Python
# LeetCode 743
class Solution:
  def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
    dist = [1e9] * (n + 1)
    dist[k] = 0
    for iteration in range(n - 1):
      updated = False
      for i in range(len(times)):
        x = times[i][0]
        y = times[i][1]
        z = times[i][2]
        if dist[y] > dist[x] + z:
          dist[y] = dist[x] + z
          updated = True
      if not updated:
        break
    ans = 0
    for i in range(1, n + 1):
      ans = max(ans, dist[i])
    if ans == 1e9:
      ans = -1
    return ans
```



## Dijkstra 求最短路代码模板

### Golang模板

```go
// Go
// https://www.acwing.com/problem/content/description/852/
package main
import (
  "fmt"
)

type Pair struct {
  A, B int
}

type PriorityQueue struct {
  Data []Pair
}
func (q *PriorityQueue) Add (p Pair) {
  q.Data = append(q.Data, p)
  cur := len(q.Data) - 1
  for {
    if cur != 0 && q.Data[cur].A < q.Data[(cur - 1) / 2].A {
      q.Data[cur], q.Data[(cur - 1) / 2] = q.Data[(cur - 1) / 2], q.Data[cur]
      cur = (cur - 1) / 2
    } else {
      return
    }
  }
}

func (q *PriorityQueue) Pop() (res Pair) {
  res = q.Data[0]
  q.Data[0] = q.Data[len(q.Data) - 1]
  q.Data = q.Data[: len(q.Data) - 1]
  cur := 0
  for {
    l, r := cur * 2 + 1, cur * 2 + 2 
    if l < len(q.Data) && q.Data[l].A < q.Data[cur].A && (r >= len(q.Data) || q.Data[l].A <= q.Data[r].A) {
      q.Data[cur], q.Data[l] = q.Data[l], q.Data[cur]
      cur = l
    } else if r < len(q.Data) && q.Data[r].A < q.Data[cur].A &&
    		(l >= len(q.Data) || q.Data[r].A <= q.Data[l].A) {
          q.Data[cur], q.Data[r] = q.Data[r], q.Data[cur]
          cur = r
    } else {
          return
    }
  }
}

const maxN = 150005
type Edge struct {
  V, W, Nxt int
}
var e = make([]Edge, maxN)
var head = make([]int, maxN)
var cnt = 1

func AddEdge(u, v, w int) {
  e[cnt].V = v
  e[cnt].W = w
  e[cnt].Nxt = head[u]
  head[u] = cnt
  cnt++
}
var n, m intvar dis = make([]int, maxN)
var q = PriorityQueue{Data: make([]Pair, 0)}

func main() {
  for i := range dis {
    dis[i] = -1
  }
  fmt.Scanf("%d %d", &n, &m)
  for ; m > 0; m-- {
    var u, v, w int
    fmt.Scanf("%d %d %d", &u, &v, &w)
    AddEdge(u, v, w)
  }
  dis[1] = 0
  q.Add(Pair{A: 0, B: 1})
  for len(q.Data) > 0 {
    cur := q.Pop()
    if cur.A != dis[cur.B] {
      continue
    } else if cur.B == n {
      break
    }
    for j := head[cur.B]; j != 0; j = e[j].Nxt {
      if dis[e[j].V] == -1 || cur.A + e[j].W < dis[e[j].V] {
        dis[e[j].V] = cur.A + e[j].W 
        q.Add(Pair{A: dis[e[j].V], B: e[j].V})
      }
    }
  }
  fmt.Println(dis[n])}
```



### C++模板

```c++
// C/C++
// https://www.acwing.com/problem/content/description/852/
#include<bits/stdc++.h>
using namespace std;
const int MAX_N = 150005, MAX_M = 150005;
vector<int> ver[MAX_N]; // 出边数组 - 另一端点
vector<int> edge[MAX_N]; // 出边数组 - 边权
int n, m, d[MAX_N];
bool v[MAX_N];
// pair<-dist[x], x>
priority_queue<pair<int, int>> q;
// 插入一条从x到y长度z的有向边
void add(int x, int y, int z) {
  ver[x].push_back(y);
  edge[x].push_back(z);
}

int main() {
  cin >> n >> m;
  for (int i = 1; i <= m; i++) {
    int x, y, z;
    scanf("%d%d%d", &x, &y, &z);
    add(x, y, z);
  }
  memset(d, 0x7f, sizeof(d));
  d[1] = 0;
  q.push(make_pair(0, 1));
  while (!q.empty()) {
    int x = q.top().second;
    q.pop();
    if (v[x]) continue;
    v[x] = true;
    for (int i = 0; i < ver[x].size(); i++) {
      int y = ver[x][i], z = edge[x][i];
      if (d[y] > d[x] + z) {
        d[y] = d[x] + z;
        q.push(make_pair(-d[y], y));
      }
    }
  }
  if (d[n] == 0x7f7f7f7f) puts("-1");
  else cout << d[n] << endl;}
```



### java模板

```java
// Java
// https://www.acwing.com/problem/content/description/852/
import java.io.*;
import java.util.*;
public class Main {
  public static void main(String args[]) throws Exception {
    Scanner input = new Scanner(System.in);
    int n = input.nextInt();
    int m = input.nextInt();
    // 模板：出边数组初始化
    // 初态：[[], [], ... []]
    List<List<Integer>> ver = new ArrayList<List<Integer>>(); // 另一端点
    List<List<Integer>> edge = new ArrayList<List<Integer>>(); // 边权
    boolean[] v = new boolean[n + 1];
    int[] dist = new int[n + 1];
    for (int i = 0; i <= n; i++) {
      ver.add(new ArrayList<Integer>());
      edge.add(new ArrayList<Integer>()); 
      v[i] = false;
      dist[i] = (int)1e9;
    }
    for (int i = 1; i <= m; i++) {
      int x = input.nextInt();
      int y = input.nextInt();
      int z = input.nextInt();
      // 出边数组 addEdge 模板
      ver.get(x).add(y);
      edge.get(x).add(z);
    }
    // Dijkstra算法
    dist[1] = 0;
    // 堆，每个结点是长度为2的数组 [点，dist]
    PriorityQueue<int[]> q = new PriorityQueue<>((a,b) -> {return a[1] - b[1];});
    q.offer(new int[]{1, 0});
    while(!q.isEmpty()){
      int[] top = q.poll();
      int x = top[0];
      if (v[x]) continue; 
      v[x] = true;
      for (int i = 0; i < ver.get(x).size(); i++) {
        int y = ver.get(x).get(i);
        int z = edge.get(x).get(i);
        if (dist[y] > dist[x] + z) {
          dist[y] = dist[x] + z;
          q.offer(new int[]{y, dist[y]});
        }
      }
    }
    System.out.println(dist[n] == 1e9 ? -1 : dist[n]);
  }
}
```



### python模板

```python
# Python
# https://www.acwing.com/problem/content/description/852/
from heapq import *
if __name__ == "__main__":
  n, m = map(int,input().split())
  ver = [[] for i in range(n + 1)] # 0~n
  edge = [[] for i in range(n + 1)] # 0~n
  dist = [1e9] * (n + 1)
  v = [False] * (n + 1)
  # 出边数组建图
  for i in range(m):
    x, y, z = map(int,input().split())
    ver[x].append(y)  # 另一端点
    edge[x].append(z) # 边权
  heap = []
  heappush(heap, (0, 1)) # (距离, 点)
  dist[1] = 0
  # Dijkstra 算法
  while heap:
    distance, x = heappop(heap)
    if v[x]:
      continue
    v[x] = True
    for i in range(len(ver[x])):
      y, z = ver[x][i], edge[x][i]
      if dist[y] > dist[x] + z:
        dist[y] = dist[x] + z 
        heappush(heap, (dist[y], y))
  print(dist[n] if dist[n] != 1e9 else -1)
```



## Kruskal 求最小生成树代码模板

### Golang模板

```go
// Go
// LeetCode 1584

type unionFind struct {
  parent, rank []int
}

func newUnionFind(n int) *unionFind {
  parent := make([]int, n)
  rank := make([]int, n)
  for i := range parent {
    parent[i] = i
    rank[i] = 1
  }
  return &unionFind{parent, rank}
}

func (uf *unionFind) find(x int) int {
  if uf.parent[x] != x {
    uf.parent[x] = uf.find(uf.parent[x])
  }
  return uf.parent[x]
}

func (uf *unionFind) union(x, y int) bool {
  fx, fy := uf.find(x), uf.find(y)
  if fx == fy {
    return false
  }
  if uf.rank[fx] < uf.rank[fy] {
    fx, fy = fy, fx
  }
  uf.rank[fx] += uf.rank[fy]
  uf.parent[fy] = fx
  return true
}

func dist(p, q []int) int {
  return abs(p[0]-q[0]) + abs(p[1]-q[1])}func minCostConnectPoints(points [][]int) (ans int) {
  n := len(points)
  type edge struct{ v, w, dis int }
  edges := []edge{}
  for i, p := range points {
    for j := i + 1; j < n; j++ {
      edges = append(edges, edge{i, j, dist(p, points[j])})
    }
  } 
  sort.Slice(edges, func(i, j int) bool {
    return edges[i].dis < edges[j].dis
  })
  
  uf := newUnionFind(n)
  left := n - 1
  for _, e := range edges {
    if uf.union(e.v, e.w) {
      ans += e.dis
      left--
      if left == 0 {
        break
      }
    } 
  } 
  return
}

func abs(x int) int {
  if x < 0 {
    return -x
  }
  return x
}
```



### C++模板

```c++
// C/C++
// LeetCode 1584
class Solution {
  public:
  	int minCostConnectPoints(vector<vector<int>>& points) {
      // 构造出边
      vector<vector<int>> edges;
      for (int i = 0; i < points.size(); i++)
        for (int j = i + 1; j < points.size(); j++) 
          edges.push_back({i, j, abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])});
      // 按照边权排序
      sort(edges.begin(), edges.end(),
           [](const vector<int>& a, const vector<int>&b) {
             return a[2] < b[2];
           });
      // Kruskal算法
      for (int i = 0; i < points.size(); i++) fa.push_back(i);
      int ans = 0;
      for (int i = 0; i < edges.size(); i++) {
        int x = edges[i][0];
        int y = edges[i][1];
        int z = edges[i][2];
        if (find(x) != find(y)) {
          ans += z;
          fa[find(x)] = find(y);
        }
      }
      return ans;
    }
  
  private:
  	vector<int> fa;
  	int find(int x) {
      if (x == fa[x]) return x;
      return fa[x] = find(fa[x]);
    }
};
```



### java模板

```java
// Java
// LeetCode 1584
class Solution {
  public int minCostConnectPoints(int[][] points) {
    // 构造出边
    List<int[]> edges = new ArrayList<>();
    for (int i = 0; i < points.length; i++)
      for (int j = i + 1; j < points.length; j++)
        edges.add(new int[]{i, j, Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1])});        
    // 按照边权排序
    edges.sort((a, b) -> { return a[2] - b[2]; });
    // Kruskal算法
    fa = new int[points.length];
    for (int i = 0; i < points.length; i++) fa[i] = i;
    int ans = 0;
    for (int i = 0; i < edges.size(); i++) {
      int x = edges.get(i)[0];
      int y = edges.get(i)[1];
      int z = edges.get(i)[2];
      if (find(x) != find(y)) {
        ans += z;
        fa[find(x)] = find(y);
      }
    }
    return ans;
  }
  int[] fa;
  int find(int x) {
    if (x == fa[x]) return x;
    return fa[x] = find(fa[x]);
  }
}
```



### python模板

```python
# Python
# LeetCode 1584
class Solution:
  def minCostConnectPoints(self, points: List[List[int]]) -> int:
    # 构造出边
    edges = []
    n = len(points)
    for i in range(n):
      for j in range(i + 1, n):
        edges.append([i, j, abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])])
        # 按边权排序
    edges.sort(key=lambda e: e[2])
    # Kruskal算法
    self.fa = []
    for i in range(n):
      self.fa.append(i)
    ans = 0
    for e in edges:
      x, y, z = self.find(e[0]), self.find(e[1]), e[2]
      if x != y:
        self.fa[x] = y
        ans += z 
    return ans
  
  def find(self, x):
    if x == self.fa[x]: 
      return x
    self.fa[x] = self.find(self.fa[x])
    return self.fa[x]
```



## Floyd 求最短路代码模板

### Golang模板

```go
// Go
// LeetCode 1334
func findTheCity(n int, edges [][]int, distanceThreshold int) int {
  // 最短路径的状态数组
  var dp [][]int
  // 先初始化
  for i := 0; i < n; i++ {
    var tmp []int
    for j := 0; j < n; j++ {
      if i == j {
        tmp = append(tmp, 0)
      } else {
        tmp = append(tmp, -1)
      } 
    } 
    dp = append(dp, tmp)
  }
  // 填出边长
  for i := 0; i < len(edges); i++ {
    from := edges[i][0] 
    to := edges[i][1] 
    weight := edges[i][2]
    // 无向图
    dp[from][to] = weight
    dp[to][from] = weight
  } 
  // dp状态转移方程
  // k放在第一层是因为后面的k要依赖前面的值
  for k := 0; k < n; k++ {
    // 从i到j
    for i := 0; i < n; i++ {
      for j := 0; j < n; j++ { 
        // 相同的节点不考虑 
        if i == j || i == k || j == k {
          continue
        } 
        // 不通的路也不考虑
        if dp[i][k] == -1 || dp[k][j] == -1 { 
          continue 
        }
        tmp := dp[i][k] + dp[k][j]
        if dp[i][j] == -1 || dp[i][j] > tmp { 
          dp[i][j] = tmp
          dp[j][i] = tmp
        } 
      }  
    } 
  }  
  
  // 统计小于阈值的路径数
  min := n  
  idx := 0  
  for i := 0; i < n; i++ {
    cnt := 0
    for j := 0; j < n; j++ {
      if i == j {
        continue
      }  
      if dp[i][j] <= distanceThreshold {
        cnt++ 
      }  
    } 
    if cnt <= min {
      min = cnt
      idx = i 
    } 
  } 
  return idx
}
```



### C++模板

```c++
// C/C++
// LeetCode 1334
class Solution {
  public:
  	int findTheCity(int n, vector<vector<int>>& edges, int distanceThreshold) {
      // 邻接矩阵初值：i到i长度为0，没有边长度为INF，其余为输入的边 
      vector<vector<int>> d(n, vector<int>(n, 1e9));
      for (auto& edge : edges) {
        int x = edge[0], y = edge[1], z = edge[2];
        d[x][y] = d[y][x] = z;
      }
      for (int i = 0; i < n; i++) d[i][i] = 0;
      // Floyd算法 
      for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
          for (int j = 0; j < n; j++)
            d[i][j] = min(d[i][j], d[i][k] + d[k][j]); 
      // 统计答案
      int ansCount = n, ans;
      for (int i = 0; i < n; i++) { 
        int count = 0; 
        for (int j = 0; j < n; j++)
          if (i != j && d[i][j] <= distanceThreshold) count++;
        if (count <= ansCount) {
          ansCount = count;
          ans = i;
        }
      }
      return ans;
    }
};
```



### java模板

```java
// Java
// LeetCode 1334
class Solution {
  public int findTheCity(int n, int[][] edges, int distanceThreshold) {
    // 邻接矩阵初值：i到i长度为0，没有边长度为INF，其余为输入的边
    int[][] d = new int[n][n];
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        d[i][j] = (int)1e9;
    for (int[] edge : edges) { 
      int x = edge[0], y = edge[1], z = edge[2];
      d[x][y] = d[y][x] = z;
    }
    for (int i = 0; i < n; i++) d[i][i] = 0;
    // Floyd算法
    for (int k = 0; k < n; k++)
      for (int i = 0; i < n; i++) 
        for (int j = 0; j < n; j++) 
          d[i][j] = Math.min(d[i][j], d[i][k] + d[k][j]);
    // 统计答案
    int ansCount = n, ans = 0;
    for (int i = 0; i < n; i++) {
      int count = 0;
      for (int j = 0; j < n; j++)
        if (i != j && d[i][j] <= distanceThreshold) count++; 
      if (count <= ansCount) {
        ansCount = count;
        ans = i;
      }
    } 
    return ans;
  }
}
```



### python模板

```python
# Python
# LeetCode 1334
class Solution:
  def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
    # 邻接矩阵初值：i到i长度为0，没有边长度为INF，其余为输入的边 
    d = [[1e9] * n for i in range(n)]
    for edge in edges:
      x, y, z = edge[0], edge[1], edge[2]
      d[x][y] = d[y][x] = z
    for i in range(n):
      d[i][i] = 0
    # Floyd算法
    for k in range(n): 
      for i in range(n):
        for j in range(n):
          d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    # 统计答案
    ansCount, ans = n, 0
    for i in range(n):
      count = 0
      for j in range(n):
        if i != j and d[i][j] <= distanceThreshold:
          count += 1
          if count <= ansCount:
            ansCount = count
            ans = i
      return ans
```





# 字符串处理

## Rabin-Karp 字符串哈希模板





### Golang模板

```go
// Golang
// LeetCode 28 实现strStr
func strStr(s, t string) int {
  if t == "" {
    return 0
  }
  n, m := len(s), len(t)
  s = " " + s
  t = " " + t
  p := int64(1e9 + 7) // 10^9+7 是一个质数
  var tHash int64 = 0
  for i := 1; i <= m; i++ {
    tHash = (tHash * 131 + (int64(t[i]) - 'a' + 1)) % p 
  }
  // 模板：预处理前缀Hash
  sHash := make([]int64, n + 1)
  sHash[0] = 0
  p131 := make([]int64, n + 1) // 131的次幂
  p131[0] = 1;
  for i := 1; i <= n; i++ {
    sHash[i] = (sHash[i - 1] * 131 + (int64(s[i]) - 'a' + 1)) % p
    p131[i] = p131[i - 1] * 131 % p
  } 
  // hello
  // ll
  for i := m; i <= n; i++ { // 滑动窗结尾
    // s[i-m+1 ~ i] 与 t[1..m] 是否相等
    if calcHash(sHash, p131, p, i - m + 1, i) == tHash &&
    s[i - m + 1: i + 1] == t[1: ] {
      return i - m // 下标变回0开始
    }
  }
  return -1
}
// 模板：O(1)得到子串[l..r]的Hash值
func calcHash(H, p131 []int64, p int64, l, r int) int64 {
  // hello 的子串ll的hash值
  //  hell
  // -he00
  // =  ll
  return ((H[r] - H[l - 1] * p131[r - l + 1]) % p + p) % p
}
```



### C++模板

```c++
// C/C++
// LeetCode 28 实现strStr
class Solution {
  public:
  	int strStr(string haystack, string needle) {
      if (needle.empty()) return 0;
      int n = haystack.size();
      int m = needle.size();
      haystack = " " + haystack;
      needle = " " + needle;
      H.push_back(0);
      for (int i = 1; i <= n; i++)
        H.push_back(H[i - 1] * 131 + haystack[i] - 'a' + 1);
      unsigned int val = 0;
      p131.push_back(1);
      for (int i = 1; i <= m; i++) {
        val = val * 131 + needle[i] - 'a' + 1;
        p131.push_back(p131[i - 1] * 131);
      }
      for (int i = m; i <= n; i++) { // 滑动窗结尾
        if (calcHash(i - m + 1, i) == val &&
            haystack.substr(i - m + 1, m) == needle.substr(1, m))
          return i - m; // 下标变回0开始
      }
      return -1;
    }
  // 模板：O(1)得到子串[l..r]的Hash值
  unsigned int calcHash(int l, int r) {
    return H[r] - H[l - 1] * p131[r - l + 1];
  }
  
private:
  vector<unsigned int> H;
  vector<unsigned int> p131;};
```



### java模板

```java
// Java
// LeetCode 28 实现strStr
class Solution {
  public int strStr(String s, String t) {
    if (t.length() == 0) return 0;
    int n = s.length();
    int m = t.length();
    s = " " + s;
    t = " " + t;
    int p = (int)1e9 + 7; // 10^9+7 是一个质数
    long tHash = 0;
    for (int i = 1; i <= m; i++)
      tHash = (tHash * 131 + (t.charAt(i) - 'a' + 1)) % p;
    // 模板：预处理前缀Hash
    long[] sHash = new long[n + 1];
    sHash[0] = 0;
    long[] p131 = new long[n + 1]; // 131的次幂
    p131[0] = 1;
    for (int i = 1; i <= n; i++) {
      sHash[i] = (sHash[i - 1] * 131 + s.charAt(i) - 'a' + 1) % p;
      p131[i] = p131[i - 1] * 131 % p;
    }
    // hello
    // ll
    for (int i = m; i <= n; i++) { // 滑动窗结尾
      // s[i-m+1 ~ i] 与 t[1..m] 是否相等
      if (calcHash(sHash, p131, p, i - m + 1, i) == tHash &&
          s.substring(i - m + 1, i + 1).equals(t.substring(1))) {
        return i - m; // 下标变回0开始 
      }
    } 
    return -1;
  }
  
  // 模板：O(1)得到子串[l..r]的Hash值
  private long calcHash(long[] H, long[] p131, int p, int l, int r) {
    // hello 的子串ll的hash值
    //  hell
    // -he00
    // =  ll
    return ((H[r] - H[l - 1] * p131[r - l + 1]) % p + p) % p;
  }
}
```



### python模板

```python
# Python
# LeetCode 28 实现strStr
class Solution:
  def strStr(self, s: str, t: str) -> int:
    if len(t) == 0:
      return 0
    n, m = len(s), len(t)
    s = " " + s 
    t = " " + t
    
    p = int(1e9 + 7)
    tHash = 0
    for i in range(1, m + 1): 
      tHash = (tHash * 13331 + ord(t[i])) % p 
      # 模板：预处理前缀Hash
      sHash = [0] * (n + 1)
      p13331 = [1] + [0] * n
      for i in range(1, n + 1):
        sHash[i] = (sHash[i - 1] * 13331 + ord(s[i])) % p
        p13331[i] = p13331[i - 1] * 13331 % p 
        # 模板：O(1)得到子串[l..r]的Hash值
        # hello 的子串ll的hash值
        #  hell
        # -he00
        # =  ll
        calcHash = lambda l, r: ((sHash[r] - sHash[l - 1] * p13331[r - l + 1]) % p + p) % p
        for i in range(m, n + 1): # 滑动窗结尾
          print(calcHash(i - m + 1, i)) 
          # s[i-m+1 ~ i] 与 t[1..m] 是否相等
          if calcHash(i - m + 1, i) == tHash and s[i - m + 1 : i + 1] == t[1:]:
            return i - m; # 下标变回0开始
        return -1
```



## 字符串转整数代码示例

### Golang模板

```go
//C/C++
class Solution {
  public:
  	int myAtoi(string s) {
      // i----->
      int index = 0;
      // 1. while 丢弃前导空格
      while (index < s.length() && s[index] == ' ') index++;
      // 2. if 判断符号
      int sign = 1;
      if (index < s.length() && (s[index] == '+' || s[index] == '-')) {
        if (s[index] == '-') sign = -1;
        index++;
      }
      // 3. while 处理数字
      int val = 0;
      // ASCII table
      // ASCII码 '0'-'9'是相连的
      while (index < s.length() && (s[index] >= '0' && s[index] <= '9')) {
        //    (a) if 数值范围 
        // if (val * 10 + (s[index] - '0') > 2147483647) 移项
        if (val > (2147483647 - (s[index] - '0')) / 10) {
          if (sign == -1) return -2147483648;
          else return 2147483647;
        }
        val = val * 10 + (s[index] - '0');
        index++;
      }
      // 4. 终止条件：遇到非数字停止
      // 已经体现在while循环中
      return val * sign;
    }
};
```



### C++模板

```c++
//C/C++
class Solution {
  public:
  	int myAtoi(string s) {
      // i----->
      int index = 0;
      // 1. while 丢弃前导空格
      while (index < s.length() && s[index] == ' ') index++;
      // 2. if 判断符号
      int sign = 1;
      if (index < s.length() && (s[index] == '+' || s[index] == '-')) {
        if (s[index] == '-') sign = -1;
        index++;
      }
      // 3. while 处理数字
      int val = 0;
      // ASCII table
      // ASCII码 '0'-'9'是相连的
      while (index < s.length() && (s[index] >= '0' && s[index] <= '9')) {
        //    (a) if 数值范围 
        // if (val * 10 + (s[index] - '0') > 2147483647) 移项
        if (val > (2147483647 - (s[index] - '0')) / 10) {
          if (sign == -1) return -2147483648;
          else return 2147483647;
        }
        val = val * 10 + (s[index] - '0');
        index++;
      }
      // 4. 终止条件：遇到非数字停止
      // 已经体现在while循环中
      return val * sign;
    }
};
```



### java模板

```java
// Java
public int myAtoi(String str) {
  int index = 0, sign = 1, total = 0;
  //1. Empty string
  if(str.length() == 0) return 0;
  //2. Remove Spaces
  while(str.charAt(index) == ' ' && index < str.length())
    index ++;
  //3. Handle signs
  if(str.charAt(index) == '+' || str.charAt(index) == '-'){
    sign = str.charAt(index) == '+' ? 1 : -1;
    index ++;
  }
  //4. Convert number and avoid overflow
  while(index < str.length()){
    int digit = str.charAt(index) - '0';
    if(digit < 0 || digit > 9) break;
    //check if total will be overflow after 10 times and add digit
    if(Integer.MAX_VALUE/10 < total ||
       Integer.MAX_VALUE/10 == total && Integer.MAX_VALUE %10 < digit)
      return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
    total = 10 * total + digit;
    index ++;
  }
  return total * sign;
}
```



### python模板

```python
# Python
class Solution(object):
  def myAtoi(self, s):
    if len(s) == 0 : return 0
    ls = list(s.strip())
    sign = -1 if ls[0] == '-' else 1
    if ls[0] in ['-','+'] : del ls[0]
      
     ret, i = 0, 0
    while i < len(ls) and ls[i].isdigit() :
      ret = ret*10 + ord(ls[i]) - ord('0')
      i += 1
    return max(-2**31, min(sign * ret,2**31-1))
```





# 高级搜索

##  KMP 字符串匹配模板

### Golang模板

```go
func strStr(haystack string, needle string) int {
  n,m:= len(haystack),len(needle)
  if m==0{
    return 0
  }
  next:=make([]int,m)
  for i,j := 1,0; i < m; i++ {
    for j>0&&needle[i]!=needle[j] {
      j=next[j-1]
    }
    if needle[i]==needle[j]{
      j++
    }
    next[i]=j
  }
  
  for i,j:= 0,0; i < n; i++ {
    for j>0&&needle[j]!=haystack[i] {
      j=next[j-1]
    }
    
    if needle[j]==haystack[i]{
      j++;
    }
    if j==m{
      return i-m+1
    }
  }
  return -1
}
```



### C++模板

```c++
// LeetCode 28 实现strStr
class Solution {
  public:
  	int strStr(string haystack, string needle) {
      if (needle.empty()) return 0;
      int n = haystack.length();
      int m = needle.length();
      vector<int> next(m, -1); // 下标从0开始，初值-1；下标从1开始，初值0。
      for (int i = 1, j = -1; i < m; i++) {
        while (j >= 0 && needle[j + 1] != needle[i]) j = next[j];
        if (needle[j + 1] == needle[i]) j++;
        next[i] = j;
      }
      
      for (int i = 0, j = -1; i < n; i++) {
        while (j >= 0 && needle[j + 1] != haystack[i]) j = next[j];
        if (j + 1 < m && needle[j + 1] == haystack[i]) j++;
        if (j + 1 == m) return i - m + 1;
      }
      return -1;
    }
};
```



### java模板

```java
// Java
// LeetCode 28 实现strStr
class Solution {
  public int strStr(String haystack, String needle) {
    if (needle.isEmpty()) return 0;
    int n = haystack.length();
    int m = needle.length();
    int[] next = new int[m];
    for (int i = 0; i < m; i++) next[i] = -1; // 下标从0开始，初值-1；下标从1开始，初值0。
    for (int i = 1, j = -1; i < m; i++) {
      while (j >= 0 && needle.charAt(j + 1) != needle.charAt(i)) j = next[j];
      if (needle.charAt(j + 1) == needle.charAt(i)) j++;
      next[i] = j;
    } 
    for (int i = 0, j = -1; i < n; i++) {
      while (j >= 0 && needle.charAt(j + 1) != haystack.charAt(i)) j = next[j];
      if (j + 1 < m && needle.charAt(j + 1) == haystack.charAt(i)) j++;
      if (j + 1 == m) return i - m + 1;
    }
    return -1;
  }
}
```



### python模板

```python
# Python
# LeetCode 28 实现strStr
class Solution:
  def strStr(self, haystack: str, needle: str) -> int:
    if len(needle) == 0:
      return 0
    n, m = len(haystack), len(needle)
    next = [-1] * m  # 下标从0开始，初值-1；下标从1开始，初值0。
    j = -1
    for i in range(1, m):
      while j >= 0 and needle[j + 1] != needle[i]:
        j = next[j]
      if needle[j + 1] == needle[i]:
        j += 1
        next[i] = j
    j = -1
    for i in range(n):
      while j >= 0 and needle[j + 1] != haystack[i]:
        j = next[j]
      if j + 1 < m and needle[j + 1] == haystack[i]:
        j += 1
      if j + 1 == m:
        return i - m + 1
   return -1
```





## 二叉平衡树

### Golang模板

```go

```



### C++模板

```c++
// lc773 滑动谜题
class Solution {
public:
    int slidingPuzzle(vector<vector<int>>& board) {
        // 2*3 => 1*6
        vector<int> list;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                list.push_back(board[i][j]);
        int start = zip(list);
        int target = 123450;
        q.push(make_pair(-evaluate(list), start));
        dist[start] = 0;
        while (!q.empty()) {
            int now = q.top().second;
            q.pop();
            if (now == target) return dist[target];
            auto a = unzip(now);
            int pos = getZeroIndex(a);
            if (pos != 0 && pos != 3 /*非最左侧*/) insert(pos, pos - 1, a, now);
            if (pos != 2 && pos != 5 /*非最右侧*/) insert(pos, pos + 1, a, now);
            if (pos >= 3) insert(pos, pos - 3, a, now);
            if (pos < 3) insert(pos, pos + 3, a, now);
        }
        return -1;
    }

    void insert(int pos, int newPos, vector<int>& a, int now) {
        swap(a[pos], a[newPos]);
        int next = zip(a);
        if (dist.find(next) == dist.end() || dist[next] > dist[now] + 1) {
            dist[next] = dist[now] + 1;
            q.push(make_pair(-dist[next] - evaluate(a) ,next));
        }
        swap(a[pos], a[newPos]);
    }

    int evaluate(vector<int>& a) {
        static int targetx[6] = {-1, 0, 0, 0, 1, 1};
        static int targety[6] = {-1, 0, 1, 2, 0, 1};
        int res = 0;
        for (int i = 0; i < 6; i++) {
            if (a[i] == 0) continue;
            int x = i/3, y = i%3;
            int tx = targetx[a[i]], ty = targety[a[i]];
            res += abs(x - tx) + abs(y - ty);
        }
        return res;
    }

    int getZeroIndex(vector<int>& a) {
        for (int i = 0; i < 6; i++)
            if (a[i] == 0) return i;
        return -1;
    }

    // [1*6]数组 => 6位整数 Hash
    int zip(vector<int>& a) {
        int res = 0;
        for (int i = 0; i < 6; i++)
            res = res * 10 + a[i];
        return res;
    }

    // 复原：6位整数 => [1*6]数组
    vector<int> unzip(int state) {
        vector<int> a(6, 0);
        for (int i = 5; i >= 0; i--) {
            a[i] = state % 10;
            state /= 10; 
        }
        return a;
    }

    // <dist+eval, state>
    priority_queue<pair<int, int>> q;
    unordered_map<int, int> dist;
};
```



### java模板

```java

```



### python模板

```python

```





## 跳表模板

### Golang模板

```go

```



### C++模板

```c++

```



### java模板

```java

```



### python模板

```python

```





# 树状数组、线段树

## 树状数组代码模板

### Golang模板

```go
type NumArray struct {
  data []int
}

func Constructor(nums []int) NumArray {
  a := NumArray{
    data: make([]int, len(nums) + 1),
  }
  for i := range nums {
    a.Update(i, nums[i])
  }
  return a
}

func (this *NumArray) Update(index int, val int) {
  x := this.SumRange(index, index)
  index++	for index < len(this.data) {
    this.data[index] += val - x
    index += index & (-index)
  }
}

func (this *NumArray) SumRange(left int, right int) int {
  right++	l, r := 0, 0
  for left > 0 {
    l += this.data[left]
    left -= left & (-left)
  }	for right > 0 {
    r += this.data[right]
    right -= right & (-right)
  }
  return r - l
}
```



### C++模板

```c++
class BinaryIndexedTree {
  public:
  	BinaryIndexedTree(vector<int>& nums) {
      n = nums.size();
      a = c = vector<int>(n + 1, 0);
      for (int i = 1; i <= n; i++) {
        a[i] = nums[i - 1];
        add(i, a[i]);
      }
    }
  
  void add(int pos, int delta) {
    for (; pos <= n; pos += pos & -pos) c[pos] += delta;
  }
  
  int get(int index) {
    return a[index];
  }
  
  void set(int index, int value) {
    a[index] = value;
  }
  
  int sumPrefix(int pos) {
    int ans = 0;
    for (; pos > 0; pos -= pos & -pos) ans += c[pos];
    return ans;
  }
  
  private:
  	int n;
  	vector<int> a;
  	vector<int> c;
};

class NumArray {
  public:
  	NumArray(vector<int>& nums)  : tree(BinaryIndexedTree(nums)) {    }
  	void update(int index, int val) {
      index++;
      tree.add(index, val - tree.get(index));
      tree.set(index, val);
    }
  int sumRange(int left, int right) {
    left++, right++;
    return tree.sumPrefix(right) - tree.sumPrefix(left - 1);
  }
  BinaryIndexedTree tree;
};
/** 
* Your NumArray object will be instantiated and called as such:
* NumArray* obj = new NumArray(nums);
* obj->update(index,val);
* int param_2 = obj->sumRange(left,right);
*/
```



### java模板

```java
class NumArray {
  public NumArray(int[] nums) {
    n = nums.length;
    a = new int[n + 1]; // 1~n
    c = new int[n + 1];
    for (int i = 1; i <= n; i++) {
      a[i] = nums[i - 1]; // 下标变为从1开始
      add(i, a[i]);
    }
  }
  
  public void update(int index, int val) {
    index++; // 下标从1开始
    int delta = val - a[index];
    add(index, delta);
    a[index] = val;
  }
  
  public int sumRange(int left, int right) {
    left++; right++; // 下标从1开始
    return query(right) - query(left - 1);
  }
  
  int query(int x) {
    int ans = 0;
    for (; x > 0; x -= lowbit(x)) ans += c[x];
    return ans;
  }
  
  void add(int x, int delta) {
    for (; x <= n; x += lowbit(x)) c[x] += delta;
  } 
  
  int lowbit(int x) {
    return x & (-x);
  } 
  
  int n;
  int[] a;
  int[] c;}
/** 
* Your NumArray object will be instantiated and called as such:
* NumArray obj = new NumArray(nums);
* obj.update(index,val);
* int param_2 = obj.sumRange(left,right);
*/
```



### python模板

```python
class BinaryIndexedTree:
  def __init__(self, nums):
    self.n = len(nums) 
    self.a = [0] * (self.n + 1)
    self.c = [0] * (self.n + 1)
    for i in range(1, self.n + 1): 
      self.a[i] = nums[i - 1]
      self.add(i, self.a[i])
  def add(self, pos, delta):
    while pos <= self.n:
      self.c[pos] += delta 
      pos += pos & -pos
      
  def get(self, index):
    return self.a[index]
  
  def set(self, index, value):
    self.a[index] = value
    
  def sumPrefix(self, pos):
    ans = 0
    while pos > 0:
      ans += self.c[pos]
      pos -= pos & -pos
      return ans
class NumArray:
  def __init__(self, nums: List[int]):
    self.tree = BinaryIndexedTree(nums)
    
  def update(self, index: int, val: int) -> None:
    index += 1
    self.tree.add(index, val - self.tree.get(index))
    self.tree.set(index, val)
    
  def sumRange(self, left: int, right: int) -> int:
    left += 1
    right += 1
    return self.tree.sumPrefix(right) - self.tree.sumPrefix(left - 1)
  
  # Your NumArray object will be instantiated and called as such:
  # obj = NumArray(nums)
  # obj.update(index,val)
  # param_2 = obj.sumRange(left,right)
```





## 线段树代码模板

### Golang模板

```go
package st
import (
  "errors"
  "fmt"
)

type SegmentTree struct {
  tree []int //线段树
  a    []int //数组数据
}

func leftChild(i int) int {
  return 2*i + 1
}

// 传入一个数组arrs和一个功能函数func,根据功能函数返回一个线段树
func NewSegmentTree(arrs []int) *SegmentTree {
  length := len(arrs)
  tree := &SegmentTree{
    tree: make([]int, length*4),
    a:    arrs,
  }
  tree.build(0, 0, length-1)
  return tree
}

// 在tree的index位置创建 arrs [ l 到 r ]  的线段树
func (tree *SegmentTree) build(index, l, r int) int {
  // 递归终止条件
  if l == r {
    tree.tree[index] = tree.a[l]
    return tree.a[l]
  }
  // 递归过程
  leftI := leftChild(index)
  rightI := leftI + 1
  mid := l + (r-l)/2
  leftResp := tree.build(leftI, l, mid)
  rightResp := tree.build(rightI, mid+1, r)
  tree.tree[index] = leftResp + rightResp
  return tree.tree[index]
}

// 查询arrs范围queryL到queryR 的结果
func (tree *SegmentTree) Query(queryL, queryR int) (int, error) {
  length := len(tree.a)
  if queryL < 0 || queryL > queryR || queryR >= length {
    return 0, errors.New("index is illegal")
  }
  return tree.queryrange(0, 0, length-1, queryL, queryR), nil
}

// 在以index为根的线段树中[l...r]范围里，搜索区间[queryL...queryR]的值
func (tree *SegmentTree) queryrange(index, l, r, queryL, queryR int) int {
  if l == queryL && r == queryR {
    return tree.tree[index]
  }
  leftI := leftChild(index)
  rightI := leftI + 1
  mid := l + (r-l)/2
  if queryL > mid {
    return tree.queryrange(rightI, mid+1, r, queryL, queryR)
  }
  if queryR <= mid { 
    return tree.queryrange(leftI, l, mid, queryL, queryR)
  }
  leftResp := tree.queryrange(leftI, l, mid, queryL, mid)
  rightResp := tree.queryrange(rightI, mid+1, r, mid+1, queryR)
  return leftResp + rightResp
}

// 更新a中索引k的值为v
func (tree *SegmentTree) Change(k, v int) {
  length := len(tree.a)
  if k < 0 || k >= length {
    return
  }
  tree.set(0, 0, length-1, k, v)
}

// 在以treeIndex为根的线段树中更新index的值为e
func (tree *SegmentTree) set(treeIndex, l, r, k, v int) {
  if l == r {
    tree.tree[treeIndex] = v 
    return
  }
  leftI := leftChild(treeIndex)
  rightI := leftI + 1
  midI := l + (r-l)/2
  if k > midI {
    tree.set(rightI, midI+1, r, k, v)
  } else {
    tree.set(leftI, l, midI, k, v)
  }
  tree.tree[treeIndex] = tree.tree[leftI] + tree.tree[rightI]
}

func (tree *SegmentTree) Print() {
  fmt.Println(tree.tree)
}
```



### C++模板

```c++
class SegmentTree {
  public:
  	SegmentTree(vector<int>& nums) {
      n = nums.size();
      a = vector<Node>(4 * n);
      build(1, 0, n - 1, nums);
    }
  
  void Change(int index, int val) {
    change(1, index, val);
  }
  
  int Query(int left, int right) {
    return query(1, left, right);
  }
  
private:
  struct Node {
    int l, r;
    int sum;
    int mark;
    // 标记：曾经想加mark，还没加，之后需要填坑
  };
  
  // 递归建树
  void build(int curr, int l, int r, vector<int>& nums) {
    a[curr].l = l;
    a[curr].r = r;
    a[curr].mark = 0;
    // 递归边界：叶子
    if (l == r) {
      a[curr].sum = nums[l];
      return;
    }
    int mid = (l + r) / 2;
    // 分两半，递归 
    build(curr * 2, l, mid, nums);
    build(curr * 2 + 1, mid + 1, r, nums);
    // 回溯时，自底向上统计信息
    a[curr].sum = a[curr * 2].sum + a[curr * 2 + 1].sum;
  }
  
  // 单点修改：先递归找到，然后自底向上统计信息
  void change(int curr, int index, int val) {
    // 递归边界：叶子[index, index]
    if (a[curr].l == a[curr].r) {
      a[curr].sum = val;
      return;
    }
    spread(curr);
    int mid = (a[curr].l + a[curr].r) / 2;
    if (index <= mid) change(curr * 2, index, val); 
    else change(curr * 2 + 1, index, val);
    a[curr].sum = a[curr * 2].sum + a[curr * 2 + 1].sum;
  }
  
  // 递归求区间和
  // 完全包含：直接返回
  // 否则：左右划分
  int query(int curr, int l, int r) {
    // 查询的是  [l     ,     r]
    // curr结点是[a[curr].l, a[curr].r]
    // l  a[curr].l  a[curr].r  r
    if (l <= a[curr].l && r >= a[curr].r) return a[curr].sum; 
    spread(curr); 
    int mid = (a[curr].l + a[curr].r) / 2;
    int ans = 0;
    if (l <= mid) ans += query(curr * 2, l, r);
    if (r > mid) ans += query(curr * 2 + 1, l, r);
    return ans;
  }
  
  // 区间修改
  void change(int curr, int l, int r, int delta) {
    // 完全包含
    if (l <= a[curr].l && r >= a[curr].r) {
      // 修改这个被完全包含的区间的信息 
      a[curr].sum += delta * (a[curr].r - a[curr].l + 1);
      // 子树不改了，有bug，标记一下 
      a[curr].mark += delta;
      return;
    }
    spread(curr);
    int mid = (a[curr].l + a[curr].r) / 2;
    if (l <= mid) change(curr * 2, l, r, delta);
    if (r > mid) change(curr * 2 + 1, l, r, delta);
    a[curr].sum = a[curr * 2].sum + a[curr * 2 + 1].sum;
  } 
  
  void spread(int curr) {
    if (a[curr].mark != 0) {
      // 有bug标记 
      a[curr * 2].sum += a[curr].mark * (a[curr * 2].r - a[curr * 2].l + 1);
      a[curr * 2].mark += a[curr].mark;
      a[curr * 2 + 1].sum += a[curr].mark * (a[curr * 2 + 1].r - a[curr * 2 + 1].l + 1);
      a[curr * 2 + 1].mark += a[curr].mark;
      a[curr].mark = 0;
    }
  }
  int n;    vector<Node> a; // 长4N的数组，存线段树
};

class NumArray {
  public:
  	NumArray(vector<int>& nums) : tree(SegmentTree(nums)) {    }
  
  	void update(int index, int val) {
      tree.Change(index, val);
    } 
  
  	int sumRange(int left, int right) {
      return tree.Query(left, right);
    }
  	SegmentTree tree;
};
/**
* Your NumArray object will be instantiated and called as such: 
* NumArray* obj = new NumArray(nums);
* obj->update(index,val);
* int param_2 = obj->sumRange(left,right);
*/
```



### java模板

```java
class SegmentTree {
  public SegmentTree(int[] nums) { 
    n = nums.length;
    a = new Node[4 * n];
    build(1, 0, n - 1, nums);
  }
  
  public void Change(int index, int val) {
    change(1, index, val);
  }
  
  public int Query(int left, int right) {
    return query(1, left, right);
  }
  
  public class Node { 
    int l, r;
    int sum; 
    int mark; // 标记：曾经想加mark，还没加，之后需要填坑 
  };
  
  // 递归建树
  void build(int curr, int l, int r, int[] nums) {
    a[curr] = new Node();
    a[curr].l = l; 
    a[curr].r = r;
    a[curr].mark = 0;
    // 递归边界：叶子
    if (l == r) {
      a[curr].sum = nums[l];
      return;
    } 
    int mid = (l + r) / 2;
    // 分两半，递归
    build(curr * 2, l, mid, nums); 
    build(curr * 2 + 1, mid + 1, r, nums); 
    // 回溯时，自底向上统计信息
    a[curr].sum = a[curr * 2].sum + a[curr * 2 + 1].sum;
  }
  // 单点修改：先递归找到，然后自底向上统计信息
  void change(int curr, int index, int val) {
    // 递归边界：叶子[index, index]
    if (a[curr].l == a[curr].r) { 
      a[curr].sum = val; 
      return; 
    }
    spread(curr); 
    int mid = (a[curr].l + a[curr].r) / 2; 
    if (index <= mid) change(curr * 2, index, val);
    else change(curr * 2 + 1, index, val);
    a[curr].sum = a[curr * 2].sum + a[curr * 2 + 1].sum;
  }
  // 递归求区间和
  // 完全包含：直接返回
  // 否则：左右划分
  int query(int curr, int l, int r) {
    // 查询的是  [l     ,     r]  
    // curr结点是[a[curr].l, a[curr].r]
    // l  a[curr].l  a[curr].r  r 
    if (l <= a[curr].l && r >= a[curr].r) return a[curr].sum;
    spread(curr);
    int mid = (a[curr].l + a[curr].r) / 2; 
    int ans = 0; 
    if (l <= mid) ans += query(curr * 2, l, r);
    if (r > mid) ans += query(curr * 2 + 1, l, r);
    return ans;
  } 
  // 区间修改
  void change(int curr, int l, int r, int delta) { 
    // 完全包含 
    if (l <= a[curr].l && r >= a[curr].r) {
      // 修改这个被完全包含的区间的信息
      a[curr].sum += delta * (a[curr].r - a[curr].l + 1);
      // 子树不改了，有bug，标记一下
      a[curr].mark += delta;
      return; 
    }
    spread(curr);
    int mid = (a[curr].l + a[curr].r) / 2; 
    if (l <= mid) change(curr * 2, l, r, delta);
    if (r > mid) change(curr * 2 + 1, l, r, delta);
    a[curr].sum = a[curr * 2].sum + a[curr * 2 + 1].sum;
  } 
  
  void spread(int curr) {
    if (a[curr].mark != 0) {
      // 有bug标记
      a[curr * 2].sum += a[curr].mark * (a[curr * 2].r - a[curr * 2].l + 1); 
      a[curr * 2].mark += a[curr].mark;
      a[curr * 2 + 1].sum += a[curr].mark * (a[curr * 2 + 1].r - a[curr * 2 + 1].l + 1);
      a[curr * 2 + 1].mark += a[curr].mark;
      a[curr].mark = 0;
    } 
  }
  
  int n;  
  Node[] a; // 长4N的数组，存线段树
};

class NumArray {
  public NumArray(int[] nums) {
    tree = new SegmentTree(nums);
  } 
  public void update(int index, int val) {
    tree.Change(index, val);
  } 
  public int sumRange(int left, int right) {
    return tree.Query(left, right);
  }
  SegmentTree tree;
}
/**
* Your NumArray object will be instantiated and called as such:
* NumArray obj = new NumArray(nums);
* obj.update(index,val);
* int param_2 = obj.sumRange(left,right);
*/
```



### python模板

```python
class Node:
  def __init__(self): 
    self.l = self.r = self.sum = self.mark = 0
    
class SegmentTree:
  def __init__(self, nums): 
    n = len(nums)
    self.a = [Node() for i in range(4 * n)]
    self.build(1, 0, n - 1, nums)
    
  def Change(self, index, val):
    self.change(1, index, val) 
    
  def Query(self, left, right):
    return self.query(1, left, right)
  
  # 递归建树
  def build(self, curr, l, r, nums):
    self.a[curr].l = l
    self.a[curr].r = r
    self.a[curr].mark = 0 
    # 递归边界：叶子
    if l == r:
      self.a[curr].sum = nums[l]
      return 
    mid = (l + r) >> 1
    # 分两半，递归
    self.build(curr * 2, l, mid, nums)
    self.build(curr * 2 + 1, mid + 1, r, nums)
    # 回溯时，自底向上统计信息
    self.a[curr].sum = self.a[curr * 2].sum + self.a[curr * 2 + 1].sum
    
    # 单点修改：先递归找到，然后自底向上统计信息
  def change(self, curr, index, val):
    # 递归边界：叶子[index, index]
    if self.a[curr].l == self.a[curr].r: 
      self.a[curr].sum = val
      return
    self.spread(curr) 
    mid = (self.a[curr].l + self.a[curr].r) >> 1
    if index <= mid:
      self.change(curr * 2, index, val)
    else:
      self.change(curr * 2 + 1, index, val) 
      self.a[curr].sum = self.a[curr * 2].sum + self.a[curr * 2 + 1].sum 
     
  # 递归求区间和      
  # 完全包含：直接返回
  # 否则：左右划分 
  def query(self, curr, l, r): 
    # 查询的是  [l     ,     r] 
    # curr结点是[a[curr].l, a[curr].r]
    # l  a[curr].l  a[curr].r  r
    if l <= self.a[curr].l and r >= self.a[curr].r:
      return self.a[curr].sum
    self.spread(curr) 
    mid = (self.a[curr].l + self.a[curr].r) >> 1
    ans = 0 
    if l <= mid:
      ans += self.query(curr * 2, l, r)
    if r > mid: 
      ans += self.query(curr * 2 + 1, l, r)
    return ans 
  
  # 区间修改
  def changeRange(self, curr, l, r, delta):
    # 完全包含
    if l <= self.a[curr].l and r >= self.a[curr].r:
      # 修改这个被完全包含的区间的信息
      self.a[curr].sum += delta * (self.a[curr].r - self.a[curr].l + 1) 
      # 子树不改了，有bug，标记一下
      self.a[curr].mark += delta
      return
    self.spread(curr) 
    mid = (self.a[curr].l + self.a[curr].r) >> 1 
    if l <= mid: 
      self.changeRange(curr * 2, l, r, delta)  
    if r > mid: 
      self.changeRange(curr * 2 + 1, l, r, delta)
      self.a[curr].sum = self.a[curr * 2].sum + self.a[curr * 2 + 1].sum
      
  def spread(self, curr):
    if self.a[curr].mark != 0:
      self.a[curr * 2].sum += self.a[curr].mark * (self.a[curr * 2].r - self.a[curr * 2].l + 1)
      self.a[curr * 2].mark += self.a[curr].mark
      self.a[curr * 2 + 1].sum += self.a[curr].mark * (self.a[curr * 2 + 1].r - self.a[curr * 2 + 1].l + 1)
      self.a[curr * 2 + 1].mark += self.a[curr].mark
      self.a[curr].mark = 0
      
class NumArray:
  def __init__(self, nums: List[int]):
    self.tree = SegmentTree(nums)
    
  def update(self, index: int, val: int) -> None:
    self.tree.Change(index, val)
    
  def sumRange(self, left: int, right: int) -> int:
    return self.tree.Query(left, right)
  
# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)# obj.update(index,val)
# param_2 = obj.sumRange(left,right)
```





## 模板基础

### Golang模板

```go

```



### C++模板

```c++

```



### java模板

```java

```



### python模板

```python

```

