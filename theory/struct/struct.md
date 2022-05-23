# 基础模版

# 数组、队列、链表、栈

## 单调队列
### C++ 代码模板

```c++

// C/C++
// LeetCode 239 滑动窗口最大值
class Solution {
  public:    
  vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    vector<int> ans;
    // 双端队列，存下标（代表时间）
    deque<int> q;
    for (int i = 0; i < nums.size(); i++) {
      // 保证队头合法性
      while (!q.empty() && q.front() <= i - k) q.pop_front();
      // 维护队列单调性，插入新的选项
      while (!q.empty() && nums[q.back()] <= nums[i]) q.pop_back();
      q.push_back(i);            
      // 取队头更新答案            
      if (i >= k - 1) ans.push_back(nums[q.front()]);        
    }
    return ans;    
  }
};
/*1 3 [-1 -3 5] 3 6 7
时间：expire_time(-1) < expire_time(-3) < expire_time(5)
值大小：-1 < -3 < 5
求max
冗余：一个下标i一个下标j，如果i<j，并且nums[i]<=nums[j]，i是冗余
去除冗余：维护下标（时间）递增、nums递减（>=）的队列
队头最优，随着下标增长，队头expire后，后面的开始逐渐变成最优*/

```

### Golang 模板

```go
// Go// LeetCode 239 滑动窗口最大值
func maxSlidingWindow(nums []int, k int) []int {
  var q, ans []int;
  for i := range nums {
    // 保证队头合法性
    for len(q) > 0 && q[0] <= i - k {	
      q = q[1:]		
    }        
    // 维护队列单调性，插入新的选项
    for len(q) > 0 && nums[q[len(q) - 1]] <= nums[i] {
      q = q[: len(q) - 1]		
    }
    q = append(q, i) 
    // 取队头更新答案
    if (i >= k - 1) {
      ans = append(ans, nums[q[0]])
    }
  }
  return ans;}
```

### java模板

```java
// Java// LeetCode 239 滑动窗口最大值
class Solution {    
  public int[] maxSlidingWindow(int[] nums, int k) {
    int[] ans = new int[nums.length - k + 1];
    // 双端队列，存下标（代表时间）        
    Deque<Integer> q = new LinkedList<>();
    for (int i = 0; i < nums.length; i++) {
      // 保证队头合法性
      while (!q.isEmpty() && q.getFirst() <= i - k) q.removeFirst();
      // 维护队列单调性，插入新的选项
      while (!q.isEmpty() && nums[q.getLast()] <= nums[i]) q.removeLast();
      q.addLast(i);            
      // 取队头更新答案
      if (i >= k - 1) ans[i - (k - 1)] = nums[q.getFirst()];
    }        
    return ans;
  }
}
```

### python模板

```python
# Python# LeetCode 239 滑动窗口最大值
class Solution:    
  def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    ans = []        
    # 数组模拟双端队列，存下标（代表时间）
    l, r = 0, -1 # left, right
    q = [0] * len(nums) # left~right（包含两端）存储队列中的元素
    for i in range(len(nums)):
      # 保证队头合法性 
      while l <= r and q[l] <= i - k: 
        l += 1 
      # 维护队列单调性，插入新的选项
      while l <= r and nums[q[r]] <= nums[i]: 
          r -= 1
      r += 1 
      q[r] = i
      # 取队头更新答案
      if i >= k - 1:
          ans.append(nums[q[l]])
    return ans
```

## 单调栈模板

### Golang模板

```go
// Go// LeetCode 84 柱状图中最大的矩形
type Rect struct {
  Height int	
  Width int
}
func largestRectangleArea(heights []int) int {
  heights = append(heights, 0) 
  // 帮助我们在最后把栈清空	
  var s []Rect	ans := 0    
  // 每个柱子入栈、出栈各一次，2n=O(n)    
  // 第一步：for 每个元素    
  for _, h := range heights {		
    accumulatedWidth := 0
    // 第二步：while (栈顶不满足高度单调性) 累加宽度，出栈
    for len(s) > 0 && s[len(s) - 1].Height >= h {
      accumulatedWidth += s[len(s) - 1].Width;
      if ans < accumulatedWidth * s[len(s) - 1].Height {
        ans = accumulatedWidth * s[len(s) - 1].Height
      }            
      s = s[: len(s) - 1] 
    }        
    // 第三步：新元素入栈
    s = append(s, Rect{h, accumulatedWidth + 1})	
  }    
  return ans
}
```



### C++模板

```c++
// C/C++// LeetCode 84 柱状图中最大的矩形
class Solution {
  public:    
  int largestRectangleArea(vector<int>& heights) {
    heights.push_back(0);
    // 帮助我们在最后把栈清空
    stack<Rect> s;
    int ans = 0;
    // 每个柱子入栈、出栈各一次，2n=O(n)
    // 第一步：for 每个元素
    for (int h : heights) {
      int accumulated_width = 0;
      // 第二步：while (栈顶不满足高度单调性) 累加宽度，出栈
      while (!s.empty() && s.top().height >= h) {
        accumulated_width += s.top().width;
        ans = max(ans, accumulated_width * s.top().height);
        s.pop();
      } 
      // 第三步：新元素入栈
      s.push({h, accumulated_width + 1});
    }
    return ans;
  }
  private:
  struct Rect {
    int height;
    int width;
  };
};

```



### java模板

```java
// Java// LeetCode 84 柱状图中最大的矩形
class Solution {
  public int largestRectangleArea(int[] heights) {
    int n = heights.length;
    int[] heights_with_zero = Arrays.copyOf(heights, n + 1);
    heights_with_zero[n] = 0;
    // 帮助我们在最后把栈清空
    Stack<Rect> s = new Stack<Rect>();
    int ans = 0;
    // 每个柱子入栈、出栈各一次，2n=O(n)
    // 第一步：for 每个元素
    for (Integer h : heights_with_zero) {
      int accumulated_width = 0;
      // 第二步：while (栈顶不满足高度单调性) 累加宽度，出栈
      while (!s.empty() && s.peek().height >= h) {
        accumulated_width += s.peek().width;
        ans = Math.max(ans, accumulated_width * s.peek().height);
        s.pop();
      } 
      // 第三步：新元素入栈 
      Rect rect = new Rect();
      rect.height = h; 
      rect.width = accumulated_width + 1;
      s.push(rect);
    } 
    return ans;
  }
  private class Rect {
    public int height; 
    public int width;
  };
}
```



### python模板

```python
# Python# LeetCode 84 柱状图中最大的矩形
class Solution:
  def largestRectangleArea(self, heights: List[int]) -> int:
    heights.append(0)
    # 帮助我们在最后把栈清空
    stack = []
    # [[height, width], ...]
    ans = 0 
    # 每个柱子入栈、出栈各一次，2n=O(n)
    # 第一步：for 每个元素
    for h in heights:
      accumulated_width = 0
      # 第二步：while (栈顶不满足高度单调性) 累加宽度，出栈
      while stack and stack[-1][0] >= h:
        accumulated_width += stack[-1][1]
        ans = max(ans, accumulated_width * stack[-1][0]) 
        stack.pop()
        # 第三步：新元素入栈
      stack.append([h, accumulated_width + 1])
    return ans
```

## 双指针夹逼模板

### Golang模板

```go
// Go// LeetCode 167 两数之和 - 输入有序数组
func twoSum(numbers []int, target int) []int {
  j := len(numbers) - 1
  for i := range numbers {
    for i < j && numbers[i] + numbers[j] > target {
      j--		
    } 
    if (i < j && numbers[i] + numbers[j] == target) {
      return []int{i + 1, j + 1} 
    }
  } 
  return nil
}
```



### C++模板

```c++
// C/C++// LeetCode 167 两数之和 - 输入有序数组
class Solution {
  public:    
  vector<int> twoSum(vector<int>& numbers, int target) {
    int j = numbers.size() - 1;
    for (int i = 0; i < numbers.size(); i++) {
      while (i < j && numbers[i] + numbers[j] > target) j--;
      if (i < j && numbers[i] + numbers[j] == target) {
        return {i + 1, j + 1};
      } 
    }
    return {};
  }
  /*    
  for i = 0 ~ n - 1
  	for j = i + 1 ~ n - 1
  		if (numbers[i] + numbers[j] == target) 
  			return ...    
  固定i，找到j使得 numbers[j] = target - numbers[i]    
  移动i，j怎么变？   
  i增大，j单调减小*/
};
```



### java模板

```java
// Java// LeetCode 167 两数之和 - 输入有序数组
class Solution {
  public int[] twoSum(int[] numbers, int target) {
    int j = numbers.length - 1;
    for (int i = 0; i < numbers.length; i++) {
      while (i < j && numbers[i] + numbers[j] > target) j--;
      if (i < j && numbers[i] + numbers[j] == target) {
        return new int[]{i + 1, j + 1};
      } 
    }
    return null;
  }
}
```



### python模板

```python
# Python# LeetCode 167 两数之和 - 输入有序数组
class Solution:
  def twoSum(self, numbers: List[int], target: int) -> List[int]:
    j = len(numbers) - 1
    for i in range(len(numbers)):
      	while i < j and numbers[i] + numbers[j] > target:
        	j -= 1 
        if i < j and numbers[i] + numbers[j] == target:
          return [i + 1, j + 1]
     return
```

## 

## 前缀和、数组计数代码模板

### Golang模板

```go
// LeetCode 1248 统计优美子数组
// 给你一个整数数组 nums 和一个整数 k。如果某个连续子数组中恰好有 k 个奇数数字，我们就认为这个子数组是「优美子数组」。
func numberOfSubarrays(nums []int, k int) int {
  n := len(nums)
  s := make([]int, n+1) // 基数和
  cnt := make([]int, n+1)
  ans := 0
  cnt[s[0]]++;
  for i := 1; i <= n; i++ {
    s[i] = s[i-1] + nums[i-1]%2; // 如果是基数
    cnt[s[i]]++;    
  }
  for i := 1; i <= n; i++ {
    if s[i] >= k {
      ans += cnt[s[i]-k]
    }
  }
  return ans
}
```



### C++模板

```c++
// C/C++// LeetCode 1248 统计优美子数组
class Solution {
  public:    int numberOfSubarrays(vector<int>& nums, int k) {
    // 开头插入一个0，转化成下标1~n 
    int n = nums.size();
    nums.insert(nums.begin(), 0);
    // 前缀和，下标0~n        
    vector<int> sum(n + 1, 0);
    for (int i = 1; i <= n; i++)
      sum[i] = sum[i - 1] + nums[i] % 2;
    // 计数，下标0~n
    vector<int> count(n + 1, 0);
    for (int i = 0; i <= n; i++)
      count[sum[i]]++;
    int ans = 0;
    for (int i = 0; i < nums.size(); i++)
      ans += sum[i] >= k ? count[sum[i] - k] : 0; 
    return ans;    
  }
};
```



### java模板

```java
// Java// LeetCode 1248 统计优美子数组
class Solution {
  public int numberOfSubarrays(int[] nums, int k) {
    int n = nums.length;
    int[] s = new int[n + 1]; // 0~n
    int[] count = new int[n + 1];
    // s[0] = 0; 
    count[s[0]]++; 
    for (int i = 1; i <= n; i++) { 
      s[i] = s[i - 1] + nums[i - 1] % 2; 
      count[s[i]]++;
    }
    int ans = 0;
    for (int i = 1; i <= n; i++) { 
      // s[i] - s[j] = k, 求j的数量 
      // s[j] = s[i] - k
      if (s[i] - k >= 0) {
        ans += count[s[i] - k];
      }
    }
    return ans;
  }
}
```



### python模板

```python
# Python# LeetCode 1248 统计优美子数组
class Solution:
  def numberOfSubarrays(self, nums: List[int], k: int) -> int:
    nums = [0] + nums
    s = [0] * len(nums) 
    for i in range(1, len(nums)):
      s[i] = s[i - 1] + nums[i] % 2
      count = [0] * len(s)
    for i in range(len(s)): 
      count[s[i]] += 1 
      ans = 0  
    for i in range(1, len(s)): 
       if s[i] - k >= 0:  
         ans += count[s[i] - k]  
    return ans
```



## 二维前缀和代码模板

### Golang模板

```go
type NumMatrix struct {
  sums [][]int
}
func Constructor(matrix [][]int) NumMatrix {
  m := len(matrix) 
  if m == 0 {
    return NumMatrix{}
  }
  n := len(matrix[0]) 
  sums := make([][]int, m+1) 
  sums[0] = make([]int, n+1)
  for i, row := range matrix {
    sums[i+1] = make([]int, n+1)
    for j, v := range row {
      sums[i+1][j+1] = sums[i+1][j] + sums[i][j+1] - sums[i][j] + v 
    }
  }
  return NumMatrix{sums}
}
func (nm *NumMatrix) SumRegion(row1, col1, row2, col2 int) int {
  return nm.sums[row2+1][col2+1] - nm.sums[row1][col2+1] - nm.sums[row2+1][col1] + nm.sums[row1][col1]
}
```



### C++模板

```c++
// C/C++
// LeetCode 304 二维区域和检索 - 矩阵不可变
class NumMatrix {
  public:    NumMatrix(vector<vector<int>>& matrix) {
    sum.clear(); 
    for (int i = 0; i < matrix.size(); i++) {
      sum.push_back({}); 
      for (int j = 0; j < matrix[i].size(); j++) 
        sum[i].push_back(get_sum(i - 1, j) + get_sum(i, j - 1) - get_sum(i - 1, j - 1) + matrix[i][j]);
    }
  }
  int sumRegion(int row1, int col1, int row2, int col2) {
    return get_sum(row2, col2) - get_sum(row1 - 1, col2) - get_sum(row2, col1 - 1) + get_sum(row1 - 1, col1 - 1);
  }
  private:
  	int get_sum(int i, int j) {
      if (i >= 0 && j >= 0) return sum[i][j];
      return 0;
    }
  vector<vector<int>> sum;
};
```



### java模板

```java
// Java
// LeetCode 304 二维区域和检索 - 矩阵不可变
class NumMatrix {
  public NumMatrix(int[][] matrix) {
    sum = new int[matrix.length][matrix[0].length];
    for (int i = 0; i < matrix.length; i++) 
      for (int j = 0; j < matrix[i].length; j++) 
        sum[i][j] = get_sum(i - 1, j) + get_sum(i, j - 1) - get_sum(i - 1, j - 1) + matrix[i][j];
  }
  public int sumRegion(int row1, int col1, int row2, int col2) {
    return get_sum(row2, col2) - get_sum(row1 - 1, col2) - get_sum(row2, col1 - 1) + get_sum(row1 - 1, col1 - 1);
  }
  private int get_sum(int i, int j) {
    if (i >= 0 && j >= 0) return sum[i][j];
    return 0;
  }
  private int[][] sum;
}
```



### python模板

```python
# Python
# LeetCode 304 二维区域和检索 - 矩阵不可变
class NumMatrix:
  def __init__(self, matrix: List[List[int]]):
    self.sum = [[0] * len(matrix[0]) for i in range(len(matrix))]
    for i in range(len(matrix)):
      for j in range(len(matrix[i])):
        self.sum[i][j] = self.getSum(i - 1, j) + self.getSum(i, j - 1) - self.getSum(i - 1, j - 1) + matrix[i][j]    
  def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
    return self.getSum(row2, col2) - self.getSum(row1 - 1, col2) - self.getSum(row2, col1 - 1) + self.getSum(row1 - 1, col1 - 1)
  def getSum(self, i, j):
    return self.sum[i][j] if i >= 0 and j >= 0 else 0
```



## 差分代码模板

### Golang模板

```go
func corpFlightBookings(bookings [][]int, n int) []int {
  arr := make([]int , n+2)
  for _, book := range bookings {
    arr[book[0]] += book[2]
    arr[book[1]+1] -= book[2]
  } 
  for i := 1; i <= n; i++ {
    arr[i] += arr[i-1]
  } 
  return arr[1:n+1]
}
```



### C++模板

```c++
// C/C++// LeetCode 1109 航班预订统计
class Solution {
  public:
  vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
    vector<int> delta(n + 2, 0);
    // 差分要开0~n+1
    for (auto& booking : bookings) {
      int fir = booking[0];
      int last = booking[1];
      int seats = booking[2];
      // 差分模板
      delta[fir] += seats;
      delta[last + 1] -= seats;
    }
    vector<int> a(n + 1, 0);// 0~n
    // 1~n 
    for (int i = 1; i <= n; i++) a[i] = a[i - 1] + delta[i];
    // 0~n-1
    for (int i = 1; i <= n; i++) a[i - 1] = a[i]; 
    a.pop_back();
    return a;
  }
};
// 任何对于区间的操作，可以转化为两个关键点（事件）
// 事件的影响从l开始，在r+1处消失
// 累加影响得到答案
// l +d    r+1   -d
```



### java模板

```java
// Java// LeetCode 1109 航班预订统计
class Solution {
  public int[] corpFlightBookings(int[][] bookings, int n) {
    int[] delta = new int[n + 2];
    // 差分要开0~n+1
    Arrays.fill(delta, 0);
    for (int[] booking : bookings) {
      int fir = booking[0];
      int last = booking[1];
      int seats = booking[2];
      // 差分模板
      delta[fir] += seats; 
      delta[last + 1] -= seats;
    }
    int[] a = new int[n + 1]; // 0~n
    a[0] = 0;
    // 1~n
    for (int i = 1; i <= n; i++) a[i] = a[i - 1] + delta[i];
    // 0~n-1
    int[] ans = new int[n];
    for (int i = 1; i <= n; i++) ans[i - 1] = a[i];
    return ans;
  }
}
```



### python模板

```python
# Python# LeetCode 1109 航班预订统计
class Solution:
  def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
    	delta = [0] * (n + 2);
    	# 差分要开0~n+1
    	for booking in bookings:
      	fir = booking[0]
     		last = booking[1]
      	seats = booking[2]
      	# 差分模板
      	delta[fir] += seats
      	delta[last + 1] -= seats
      a = [0] * (n + 1)  # 0~n
      # 1~n
      for i in range(1, n + 1):
        a[i] = a[i - 1] + delta[i]
        # 0~n-1
      return a[1:]
```



# 哈希、集合、映射

## LRU模板

### Golang模板

```go
package lru
import (
  "container/list"
  "errors"
)
type LRU struct {
  size      int
  innerList *list.List
  innerMap  map[int]*list.Element
}

type entry struct {
  key   int
  value int
}

func NewLRU(size int) (*LRU, error) {
  if size <= 0 {
    return nil, errors.New("must provide a positive size")
  }
  c := &LRU{
    size:      size,
    innerList: list.New(),
    innerMap:  make(map[int]*list.Element),
  }
  return c, nil
}

func (c *LRU) Get(key int) (int, bool) {
  if e, ok := c.innerMap[key]; ok {
    c.innerList.MoveToFront(e)
    return e.Value.(*entry).value, true
  }
  return -1, false
}

func (c *LRU) Put(key int, value int) (evicted bool) {
  if e, ok := c.innerMap[key]; ok {
    c.innerList.MoveToFront(e)
    e.Value.(*entry).value = value
    return false
  } else {
    e := &entry{key, value}
    ent := c.innerList.PushFront(e)
    c.innerMap[key] = ent
    if c.innerList.Len() > c.size {
      last := c.innerList.Back()
      c.innerList.Remove(last)
      delete(c.innerMap, last.Value.(*entry).key)
      return true
    }
    return false
  }
}
```



### C++模板

```c++
//C/C++
class LRUCache {
  public:
  LRUCache(int capacity) : capacity(capacity) {
    head.next = &tail;
    tail.pre = &head;
  }
  
  int get(int key) {
    if (h.find(key) == h.end()) return -1;
    Node* item = h[key];
    removeFromList(item);
    insertToList(item);
    return item->value;
  }
  
  void put(int key, int value) {
    if (h.find(key) == h.end()) { 
      Node* item = new Node();
      item->key = key, item->value = value;
      insertToList(item);
      h[key] = item;
    } else {
      Node* item = h[key];
      item->value = value;
      removeFromList(item);
      insertToList(item);
    }
    if (h.size() > capacity) {
      Node* node = tail.pre;
      removeFromList(node);
      h.erase(node->key);
      delete node;
    }
  }
  private:
  	struct Node {
    	int key;
  	  int value;
    	Node* pre;
    	Node* next;
  	};
  	void removeFromList(Node* node) {
    	node->pre->next = node->next;
    	node->next->pre = node->pre;
  	}
  	void insertToList(Node* node) {
    	head.next->pre = node;
    	node->next = head.next;
    	head.next = node;
    	node->pre = &head;
  	}
  	int capacity;
  	unordered_map<int, Node*> h;
  	Node head, tail;
};
```



### java模板

```java
// Java
public class LRUCache {
  private class Node {
    public int key;
    public int value;
    public Node pre;
    public Node next;
  };
  private HashMap<Integer, Node> map;
  // 保护结点
  private Node head;
  private Node tail;
  private int capacity;
  public LRUCache(int capacity) {
    this.capacity = capacity;
    this.map = new HashMap<Integer, Node>();
    // 建立带有保护结点的空双向链表
    head = new Node();
    tail = new Node();
    head.next = tail;
    tail.pre = head;    
  }        
  public int get(int key) {
    if (!this.map.containsKey(key)) return -1;
    Node node = map.get(key);
    // 从链表和map中删掉
    this.removeFromList(node);
    // 重新插入到map、链表头部，维护时间顺序
    this.insertToListHead(node.key, node.value);
    return node.value;
  }
  public void put(int key, int value) { 
    if (this.map.containsKey(key)) {
      Node node = this.map.get(key);
      // 从链表中删掉
      this.removeFromList(node);
      // 重新插入到头部，维护时间顺序
      this.insertToListHead(key, value);
    } else {
      // 在链表中插入新结点，返回新结点引用
      this.insertToListHead(key, value); 
    } 
    if (this.map.size() > this.capacity) { 
      this.removeFromList(tail.pre);
    }
  }
  private void removeFromList(Node node) {
    node.pre.next = node.next;
    node.next.pre = node.pre;
    this.map.remove(node.key);
  } 
  private Node insertToListHead(int key, int value) {
    Node node = new Node();
    node.key = key;
    node.value = value;
    // node与head的下一个点之间建立联系
    node.next = head.next;
    head.next.pre = node;
    // node与head之间建立联系
    node.pre = head;
    head.next = node;
    // 建立映射关系
    this.map.put(key, node);
    return node;
  }
}
```



### python模板

```python
# Python 
class LRUCache(object):
  def __init__(self, capacity):
    self.dic = collections.OrderedDict()
    self.remain = capacity
  def get(self, key):
    if key not in self.dic:
      return -1
    v = self.dic.pop(key)
    self.dic[key] = v   # key as the newest one
    return v
  def put(self, key, value):
    if key in self.dic:
      self.dic.pop(key)
     else:
      if self.remain > 0:
        self.remain -= 1
        else:   # self.dic is full
          self.dic.popitem(last=False)
          self.dic[key] = value
```



# 递归、分治

## 子集递归模板

### Golang模板

```go
func subsets(nums []int) [][]int {
  var s []int
  var ans [][]int
  var findSubsets func(index int)
  findSubsets = func(index int) {
    if index == len(nums) {
      ans = append(ans, make([]int, 0))
      for _, i := range s {
        ans[len(ans) - 1] = append(ans[len(ans) - 1], i)
      }
      return
    }
    findSubsets(index + 1)
    s = append(s, nums[index])
    findSubsets(index + 1)
    s = s[: len(s) - 1]
  }
  findSubsets(0)
  return ans
}
```



### C++模板

```c++
class Solution {
  public:
  	vector<vector<int>> subsets(
      vector<int>& nums) {
      findSubsets(nums, 0);
      return ans;
    }
  
  	void findSubsets(vector<int>& nums, int index) {
      if (index == nums.size()) {
        ans.push_back(s);
        return;
      }
      findSubsets(nums, index + 1);
      s.push_back(nums[index]);
      findSubsets(nums, index + 1);
      s.pop_back();
    }
  private:
  vector<vector<int>> ans;
  vector<int> s;
};
```



### java模板

```java
class Solution {
  public List<List<Integer>> subsets(int[] nums) {
    ans = new ArrayList<List<Integer>>();
    s = new ArrayList<Integer>();
    findSubsets(nums, 0);
    return ans;
  }
  private void findSubsets(int[] nums, int index) {
    if (index == nums.length) {
      ans.add(new ArrayList<Integer>(s));
      return;
    } 
    findSubsets(nums, index + 1);
    s.add(nums[index]);
    findSubsets(nums, index + 1);
    s.remove(s.size() - 1);
  } 
  private List<List<Integer>> ans;
  private List<Integer> s;
}
```



### python模板

```python
class Solution:
  def subsets(self, nums: List[int]) -> List[List[int]]:
    self.ans = []
    self.s = []
    self.n = len(nums)
    self.findSubsets(nums, 0)
    return self.ans 
  def findSubsets(self, nums, index):
    if index == self.n:
      self.ans.append(self.s[:]) # make a copy
      return
    self.findSubsets(nums, index + 1)
    self.s.append(nums[index])
    self.findSubsets(nums, index + 1)
    self.s.pop()
```



## 组合递归模板



### Golang模板

```go
func combine(n int, k int) [][]int {
  ans := [][]int{}
  s := []int{}
  var findSubsets func(index int)
  findSubsets = func(index int) {
    if len(s) > k || len(s) + (n-index+1) < k{
      return
    } 
    if index == n + 1 {
      tmp := make([]int, k)
      copy(tmp, s)
      ans = append(ans, tmp)
      return
    }
    findSubsets(index+1)
    s = append(s, index)
    findSubsets(index+1)
    s = s[:len(s)-1]
  }
  
  findSubsets(1)
  return ans
}
```



### C++模板

```c++
class Solution {
  public:
  	vector<vector<int>> combine(int n, int k) {
      this->n = n;
      this->k = k;
      findSubsets(1);
      return ans;
    }
  	void findSubsets(int index) {
      // 已经选了超过k个，
      // 或者把剩下的全选上也不够k个，退出
      if (s.size() > k || s.size() + n - index + 1 < k) return;
      if (index == n + 1) {
        ans.push_back(s);
        return;
      }
      findSubsets(index + 1);
      s.push_back(index);
      findSubsets(index + 1);
      s.pop_back();
    }
  private:
  	vector<vector<int>> ans;
  	vector<int> s;
  	int n;
  	int k;};
```



### java模板

```java
class Solution {
  public List<List<Integer>> combine(int n, int k) {
    ans = new ArrayList<List<Integer>>();
    s = new ArrayList<Integer>();
    this.n = n;
    this.k = k;
    findSubsets(1);
    return ans;
  }
  private void findSubsets(int index) {
    // 已经选了超过k个，
    // 或者把剩下的全选上也不够k个，退出
    if (s.size() > k || s.size() + n - index + 1 < k) return;
    if (index == n + 1) {
      ans.add(new ArrayList<Integer>(s));
      return;
    }
    findSubsets(index + 1);
    s.add(index);
    findSubsets(index + 1);
    s.remove(s.size() - 1);
  }
  private List<List<Integer>> ans;
  private List<Integer> s;
  private int n;
  private int k;
}
```



### python模板

```python
class Solution:
  def combine(self, n: int, k: int) -> List[List[int]]:
    self.ans = []
    self.s = []
    self.n = n
    self.k = k
    self.findSubsets(1)
    return self.ans
  def findSubsets(self, index):
    # 已经选了超过k个，
    # 或者把剩下的全选上也不够k个，退出
    if len(self.s) > self.k or len(self.s) + self.n - index + 1 < self.k:
      return
    if index == self.n + 1:
      self.ans.append(self.s[:]) # make a copy
      return
    self.findSubsets(index + 1)
    self.s.append(index)
    self.findSubsets(index + 1)
    self.s.pop()
```



## 排列递归模板



### Golang模板

```go
func permute(nums []int) [][]int {
  res := [][]int{}
  used := make([]bool, len(nums))
  var find func(path []int, depth int)
  find = func(path []int, depth int) {
    if len(nums) == depth {
      tmp := make([]int, len(path))
      copy(tmp ,path)
      res = append(res, tmp)
      return
    }
    for i := 0; i < len(nums); i++ {
      if used[i] {continue;}
      used[i] = true;
      path = append(path, nums[i])
      find(path,depth+1)
      path = path[:len(path) - 1]
      used[i] = false;
    }
  }
  find([]int{}, 0)
  return res
}
```



### C++模板

```c++
class Solution {
  public:
  	vector<vector<int>> permute(vector<int>& nums) {
      used = vector<bool>(nums.size(), false);
      find(nums, 0);
      return ans;
    }
  
  	void find(vector<int>& nums, int count) {
      if (count == nums.size()) {
        ans.push_back(s);
        return;
      }
      for (int i = 0; i < nums.size(); i++)
        if (!used[i]) {
          used[i] = true;
          s.push_back(nums[i]);
          find(nums, count + 1);
          s.pop_back();
          used[i] = false;
        }
    }
  private:
  	vector<vector<int>> ans;
  	vector<int> s;
  	vector<bool> used;};
```



### java模板

```java
class Solution {
  public List<List<Integer>> permute(int[] nums) {
    n = nums.length;
    num = new int[n];
    for (int i = 0; i < n; i++) num[i] = nums[i];
    used = new boolean[n];
    per = new ArrayList<Integer>();
    ans = new ArrayList<List<Integer>>();
    dfs(0);
    return ans;
  }
  
  private void dfs(int depth) {
    if (depth == n) {
      ans.add(new ArrayList<Integer>(per));
      return;
    }
    for (int i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      per.add(num[i]);
      dfs(depth + 1);
      per.remove(per.size() - 1);
      used[i] = false;
    }
  }
  
  private int n;
  private int[] num;
  private boolean[] used;
  private List<Integer> per;
  private List<List<Integer>> ans;
}
```



### python模板

```python
class Solution:
  def permute(self, nums: List[int]) -> List[List[int]]:
    self.nums = nums
    self.ans = []
    self.per = []
    self.n = len(nums)
    self.used = [False] * self.n
    self.find(0)
    return self.ans
  # 依次考虑0,1,...,n-1位置放哪个数
  # “从还没用过的”数中选一个放在当前位置
  def find(self, index):
    if index == self.n:
      self.ans.append(self.per[:])  # make a copy
      return
    for i in range(self.n):
      if not self.used[i]:
        self.used[i] = True
        self.per.append(self.nums[i])
        self.find(index + 1)
        self.per.pop()
        self.used[i] = False
```



## 树与图（以下部分分散）



# 深度优先搜索、广度优先搜索

## BFS拓扑排序模板

### Golang模板

```go
//Go
// LeetCode210 课程表II
func topsort(outCome [][]int, inCome []int, numCourses int) {
  queue := make([]int,0)
  for i:=0; i<numCourses; i++ {
    if inCome[i] == 0 {
      queue = append(queue,i)
    }
  }
  rst := make([]int,0)
  for len(queue) > 0 {
    node := queue[0]
    queue = queue[1:]
    rst = append(rst,node)
    for _,v := range outCome[node] {
      inCome[v]--
      if inCome[v] == 0 {
        queue = append(queue,v)
      }
    }
  }
  // 如果没有环，则认为是true
  if len(rst) == numCourses {
    return rst
  }
  return nil
}

func findOrder(numCourses int, prerequisites [][]int) []int {
  outCome := make([][]int,numCourses)
  inCome := make([]int,numCourses)
  // 有向图由pre[i][1] -> pre[i][0]
  for i:=0; i<len(prerequisites); i++ {
    outCome[prerequisites[i][1]] = append(outCome[prerequisites[i][1]],prerequisites[i][0])	
    inCome[prerequisites[i][0]]++	
  }
  return topsort(outCome, inCome, numCourses)
}
```



### C++模板

```c++
//C/C++
// LeetCode210 课程表II
class Solution {
  public:
  	vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
      n = numCourses;
      // 出边数组初始化，n个空list
      edges = vector<vector<int>>(n, vector<int>());
      inDeg = vector<int>(n, 0);
      for (vector<int>& pre : prerequisites) {
        int ai = pre[0];
        int bi = pre[1];
        // 加边模板 
        addEdge(bi, ai);
      }
      auto ans = topsort();
      if (ans.size() < n) return {}; // 不能完成所有课程
      return ans;
    }
  private:
  // 有向图找环 模板（拓扑排序）
  // 返回学的课程数
  vector<int> topsort() {
    vector<int> order;
    // 拓扑排序基于BFS，需要队列
    queue<int> q;
    // 从所有零入度点出发
    for (int i = 0; i < n; i++)
      if (inDeg[i] == 0) q.push(i);
    // 执行BFS
    while (!q.empty()) {
      int x = q.front(); // 取队头（这门课学了）
      q.pop();
      order.push_back(x);
      // 考虑x的所有出边
      for (int y : edges[x]) {
        inDeg[y]--; // 去掉约束关系
        if (inDeg[y] == 0) {
          q.push(y);
        }
      }
    }
    return order;
  }
  void addEdge(int x, int y) {
    edges[x].push_back(y);
    inDeg[y]++;
  }
  int n;
  vector<vector<int>> edges;
  vector<int> inDeg; // in degree 入度};
```



### java模板

```java
// Java
// LeetCode210 课程表II
class Solution {
  public int[] findOrder(int numCourses, int[][] prerequisites) {
    // 初始化
    n = numCourses;
    edges = new ArrayList<List<Integer>>();
    inDeg = new int[n];
    for (int i = 0; i < n; i++) {
      edges.add(new ArrayList<Integer>());
      inDeg[i] = 0;
    }
    // 建图，加边 
    for (int[] pre : prerequisites) {
      int ai = pre[0];
      int bi = pre[1];
      addEdge(bi, ai);
    }
    // 拓扑排序
    return topsort();
  }
  int[] topsort() {
    int[] order = new int[n];
    int m = 0;
    Queue<Integer> q = new LinkedList<Integer>();
    // 零入度点入队
    for (int i = 0; i < n; i++)
      if (inDeg[i] == 0) q.offer(i);
    while (!q.isEmpty()) {
      Integer x = q.poll();
      order[m] = x;
      m++;
      // 扩展每个点
      for (Integer y : edges.get(x)) {
        inDeg[y]--;
        if (inDeg[y] == 0) q.offer(y);
      }
    } 
    // n个课程都进出过队列，说明可以完成
    if (m == n) return order;
    return new int[0];
  }
  
  private void addEdge(int x, int y) {
    edges.get(x).add(y);
    inDeg[y]++;
  }
  private int n;
  private List<List<Integer>> edges;
  private int[] inDeg;
}
```



### python模板

```python
# LeetCode210 课程表II
from collections import deque
class Solution:    
  def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    # 初始化
    self.n = numCourses
    self.edge = [[] for i in range(numCourses)]
    self.inDeg = [0] * numCourses
    # 加边
    for pre in prerequisites:
      ai, bi = pre[0], pre[1]
      self.addEdge(bi, ai)
      return self.topsort()
    
   def topsort(self): 
    order = []
    q = deque()
    for i in range(self.n): 
      if self.inDeg[i] == 0:
        q.append(i)
    while len(q) > 0:
      x = q.popleft()
      order.append(x)
      for y in self.edge[x]:
        self.inDeg[y] -= 1
        if self.inDeg[y] == 0:
           q.append(y)
     if len(order) == self.n:
        return order
     return []
    def addEdge(self, u, v):
      self.edge[u].append(v)
      self.inDeg[v] += 1
```



## BFS地图类模板

### Golang模板

```go
//Golang
func numIslands(grid [][]byte) int {
  m := len(grid)
  n := len(grid[0])
  visit := make([][]bool, 0)
  for i := 0; i < m; i++ {
    visit = append(visit, make([]bool, n))
  }
  dx := []int{-1, 0, 0, 1}
  dy := []int{0, -1, 1, 0}
  var bfs func(int, int)
  bfs = func(sx int, sy int) {
    q := make([][]int, 0)
    // 第一步：push起点 
    q = append(q, []int{sx, sy})
    for len(q) > 0 {
      now := q[0] 
      q = q[1:]
      x, y := now[0], now[1]
      // 扩展所有出边（四个方向）
      for i := 0; i < 4; i++ {
        nx := x + dx[i]
        ny := y + dy[i]
        // 任何时候访问数组前，判断合法性
        if nx < 0 || ny < 0 || nx >= m || ny >= n {
          continue
        }
        if grid[nx][ny] == '1' && !visit[nx][ny] {
          q = append(q, []int{nx, ny})
          // BFS：入队时标记visit 
          visit[nx][ny] = true
        }
      }
    }
  }
  ans := 0
  for i := 0; i < m; i++ {
    for j := 0; j < n; j++ {
      if grid[i][j] == '1' && !visit[i][j] {
        bfs(i, j)
        ans++
      }
    }
  }
  return ans;
};
```



### C++模板

```c++
//C/C++
class Solution {
  public:
  int numIslands(vector<vector<char>>& grid) {
    this->m = grid.size();
    this->n = grid[0].size();
    visit = vector<vector<bool>>(m, vector<bool>(n, false));
    int ans = 0;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        if (grid[i][j] == '1' && !visit[i][j]) { 
          bfs(grid, i, j);
          ans++;
        }
    return ans;
  }
  private:    
  // 从(sx,sy)出发bfs
  	void bfs(vector<vector<char>>& grid, int sx, int sy) {
      // 长度为2的list或者pair都可以
      queue<pair<int,int>> q;
      // 第一步：push起点
      q.push(make_pair(sx,sy));
      visit[sx][sy] = true;
      while (!q.empty()) { 
        int x = q.front().first;
        int y = q.front().second;
        q.pop();
        // 扩展所有出边（四个方向）
        for (int i = 0; i < 4; i++) {
          int nx = x + dx[i];
          int ny = y + dy[i];
          // 任何时候访问数组前，判断合法性
          if (nx < 0 || ny < 0 || nx >= m || ny >= n) continue;
          if (grid[nx][ny] == '1' && !visit[nx][ny]) {
            q.push(make_pair(nx, ny));
            // BFS：入队时标记visit
            visit[nx][ny] = true;
          }
        }
      }
    }
  int m;
  int n;
  vector<vector<bool>> visit;
  const int dx[4] = {-1, 0, 0, 1};
  const int dy[4] = {0, -1, 1, 0};
};
```



### java模板

```java
//Java
class Solution {
  public int numIslands(char[][] grid) {
    m = grid.length;
    n = grid[0].length;
    visit = new boolean[m][n];
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        visit[i][j] = false;
    int ans = 0;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        if (grid[i][j] == '1' && !visit[i][j]) {
          bfs(grid, i, j);
          ans++;
        }
    return ans;
  }
  // 从(sx,sy)出发bfs
  private void bfs(char[][] grid, int sx, int sy) {
    int[] dx = {-1, 0, 0, 1};
    int[] dy = {0, -1, 1, 0};
    Queue<Pair<Integer,Integer>> q = new LinkedList<Pair<Integer,Integer>>();
    // 第一步：push起点
    q.offer(new Pair<Integer,Integer>(sx,sy));
    visit[sx][sy] = true;
    while (!q.isEmpty()) {
      int x = q.peek().getKey();
      int y = q.poll().getValue();
      // 扩展所有出边（四个方向）
      for (int i = 0; i < 4; i++) { 
        int nx = x + dx[i];
        int ny = y + dy[i];
        // 任何时候访问数组前，判断合法性
        if (nx < 0 || ny < 0 || nx >= m || ny >= n) continue;
        if (grid[nx][ny] == '1' && !visit[nx][ny]) {
          q.offer(new Pair<Integer,Integer>(nx, ny));
          // BFS：入队时标记visit
          visit[nx][ny] = true;
        }
      }
    }
  }
  private int m;
  private int n;
  private boolean[][] visit;
```



### python模板

```python
# Python
from collections import deque
class Solution:
  def numIslands(self, grid: List[List[str]]) -> int:
    self.m = len(grid)
    self.n = len(grid[0])
    self.visit = [[False] * self.n for i in range(self.m)]
    ans = 0
    for i in range(self.m):
      for j in range(self.n):
        if grid[i][j] == '1' and not self.visit[i][j]:
          self.bfs(grid, i, j)
          ans += 1
    return ans
  
  def bfs(self, grid, sx, sy):
    dx = [-1, 0, 0, 1]
    dy = [0, -1, 1, 0]
    q = deque()
    # 第一步：push起点
    q.append([sx, sy])
    self.visit[sx][sy] = True
    while len(q) > 0:
      now = q.popleft()
      x, y = now[0], now[1]
      # 扩展所有出边（四个方向）
      for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        # 任何时候访问数组前，判断合法性
        if nx < 0 or ny < 0 or nx >= self.m or ny >= self.n:
          continue 
        if grid[nx][ny] == '1' and not self.visit[nx][ny]:
          q.append([nx, ny])
          # BFS：入队时标记visit 
          self.visit[nx][ny] = True
```



## DFS无向图找环模板

### Golang模板

```go
// Golang
// LeetCode684 冗余连接
// 本题有更高效解法，本解法主要练习DFS找环
func findRedundantConnection(edges [][]int) []int {
  n := 0
  for _, e  := range edges {
    if n < e[0] {
      n = e[0]
    }
    if n < e[1] {
      n = e[1]
    }
  }
  // 模板：出边数组初始化
  edge := make([][]int, n + 1)
  visit := make([]bool, n + 1)
  hasCycle := false
  // 模板：加边
  addEdge := func(u, v int) {
    edge[u] = append(edge[u], v)
  }
  // 模板：DFS无向图找环
  var dfs func(int, int)
  dfs = func(x, fa int) {
    visit[x] = true
    for _, y := range edge[x] {
      if y == fa {
        continue
      }
      if visit[y] {
        hasCycle = true
      } else {
        dfs(y, x)
      } 
    }
  }
  for _, e := range edges {
    u, v := e[0], e[1]
    addEdge(u, v)
    addEdge(v, u)
    visit = make([]bool, n + 1)
    dfs(u, -1)
    if (hasCycle) {
      return e
    }
  }
  return nil
};
```



### C++模板

```c++
//C/C++
// LeetCode684 冗余连接
// 本题有更高效解法，本解法主要练习DFS找环
class Solution {
  public:
  	vector<int> findRedundantConnection(vector<vector<int>>& edges) {
      for (vector<int>& e : edges) {
        int u = e[0], v = e[1];
        n = max(n, u);
        n = max(n, v);
      }
      // 模板：出边数组初始化
      edge = vector<vector<int>>(n + 1, vector<int>());
      visit = vector<bool>(n + 1, false);
      hasCycle = false;
      for (vector<int>& e : edges) {
        int u = e[0], v = e[1];
        addEdge(u, v);
        addEdge(v, u);
        dfs(u, 0);
        if (hasCycle) return e;
      }
      return {};
    }
  private:
  // 模板：DFS无向图找环
  	void dfs(int x, int fa) {
      visit[x] = true;
      for (int y : edge[x]) {
        if (y == fa) continue;
        if (!visit[y]) dfs(y, x);
        else hasCycle = true;
      }
      visit[x] = false;
    }
  // 模板：加边
  void addEdge(int x, int y) {
    edge[x].push_back(y);
  }
  int n;
  vector<vector<int>> edge;
  vector<bool> visit;
  bool hasCycle;
};
```



### java模板

```java
// Java
// LeetCode684 冗余连接
// 本题有更高效解法，本解法主要练习DFS找环
class Solution {
  public int[] findRedundantConnection(int[][] input) { 
    // 出现过的最大的点就是n
    n = 0;
    for (int[] edge : input) {
      int u = edge[0];
      int v = edge[1];
      n = Math.max(u, n);
      n = Math.max(v, n);
    }
    // 模板：出边数组初始化
    // 初态：[[], [], ... []]
    edges = new ArrayList<List<Integer>>();
    // [false, false, ...]
    visit = new boolean[n + 1];
    for (int i = 0; i <= n; i++) {
      edges.add(new ArrayList<Integer>());
      visit[i] = false;
    }
    hasCycle = false;
    // 加边
    for (int[] edge : input) {
      int u = edge[0];
      int v = edge[1];
      // 无向图看作双向边的有向图
      addEdge(u, v);
      addEdge(v, u); 
      // 每加一条边，看图中是否多了环c
      for (int i = 0; i <= n; i++) visit[i] = false;
      dfs(u, -1);
      if (hasCycle) return edge;
    }
    return null;
  }
  
  // 模板：无向图深度优先遍历找环
  // visit数组，避免重复访问
  // fa是第一次走到x的点
  private void dfs(int x, int fa) {
    // 第一步：标记已访问
    visit[x] = true;
    // 第二步：遍历所有出边
    for (Integer y : edges.get(x)) {
      if (y == fa) continue; // 返回父亲，不是环
      if (visit[y]) hasCycle = true;
      else dfs(y, x);
    }
  }
  // 模板：加边
  private void addEdge(int x, int y) {
    edges.get(x).add(y);
  }
  // 出边数组
  int n;
  private List<List<Integer>> edges;
  boolean hasCycle;
  private boolean[] visit;
}
```



### python模板

```python
# Python
# LeetCode684 冗余连接
# 本题有更高效解法，本解法主要练习DFS找环
class Solution:
  def findRedundantConnection(self, input: List[List[int]]) -> List[int]:
    # 模板：出边数组初始化
    self.edge = [[] for i in range(1001)]
    # max n is 1000
    self.hasCycle = False
    for e in input:
      u, v = e[0], e[1]
      self.addEdge(u, v)
      self.addEdge(v, u)
      self.visit = [False] * 1001
      self.dfs(u, -1) 
      if self.hasCycle:
        return e
      return []
    # 模板：DFS无向图找环
    def dfs(self, x, fa): 
      self.visit[x] = True
      for y in self.edge[x]:
        if y == fa:
          continue
        if self.visit[y]:
          self.hasCycle = True
        else:
          self.dfs(y, x)
    # 模板：加边
    def addEdge(self, u, v):
      self.edge[u].append(v)
```





# 二叉堆、二叉搜索树、二分

## 二叉搜索树模板

### Golang模板

```go
// 插入，保证val不存在
// 返回插入以后的新树根
func insertIntoBST(root *TreeNode, val int) *TreeNode {
  if root == nil {
    return &TreeNode{Val: val}
  }
  if val < root.Val {
    root.Left = insertIntoBST(root.Left, val)
  } else {
    root.Right = insertIntoBST(root.Right, val)
  }
  return root;
}

// 求val的后继
func findSucc(root *TreeNode, val int) *TreeNode {
  var ans *TreeNode
  for root != nil {
    if val == root.Val {
      if root.Right != nil {
        p := root.Right 
        for p.Left != nil {
          p = p.Left
        }
        return p
      }
      break 
    }
    if root.Val > val && (ans == nil || ans.Val > root.Val) {
      ans = root
    }
    if val < root.Val {
      root = root.Left
    } else { 
      root = root.Right
    } 
  }
  return ans
}

// 在以root为根的子树中删除key，返回新的根
func deleteNode(root *TreeNode, key int) *TreeNode {
  if root == nil {
    return nil
  }
  if root.Val == key {
    if root.Left == nil {
      return root.Right // 没有左子树，让right代替root的位置
    }
    if root.Right == nil {
      return root.Left // 没有右子树
    }
    next := root.Right
    for next.Left != nil {
      next = next.Left // 找后继：右子树一路向左
    }
    root.Right = deleteNode(root.Right, next.Val)
    root.Val = next.Val
    return root
  }
  if key < root.Val {
    root.Left = deleteNode(root.Left, key)
  } else {
    root.Right = deleteNode(root.Right, key)
  } 
  return root
}
```



### C++模板

```c++
// C/C++
// 插入，保证val不存在
// 返回插入以后的新树根
TreeNode* insertIntoBST(TreeNode* root, int val) {
  if (root == nullptr) {
    return new TreeNode(val);
  }
  if (val < root->val) {
    root->left = insertIntoBST(root->left, val);
  } else {
    root->right = insertIntoBST(root->right, val);
  }
  return root;
}

// 求val的后继
TreeNode* findSucc(TreeNode* root, int val) {
  TreeNode* ans = nullptr;
  while (root != nullptr) {
    if (val == root->val) {
      if (root->right != nullptr) {
        TreeNode* p = root->right;
        while (p->left != nullptr) p = p->left;
        return p;
      }
      break;
    }
    if (root->val > val && (ans == nullptr || ans->val > root->val))
      ans = root;
    if (val < root->val) root = root->left;
    else root = root->right;
  }
  return ans;
}

// 在以root为根的子树中删除key，返回新的根
TreeNode* deleteNode(TreeNode* root, int key) {
  if (root == nullptr) return nullptr;
  if (root->val == key) {
    if (root->left == nullptr) return root->right;
    // 没有左子树，让right代替root的位置
    if (root->right == nullptr) return root->left;
    // 没有右子树
    TreeNode* next = root->right;
    while (next->left != nullptr) next = next->left;
    // 找后继：右子树一路向左
    root->right = deleteNode(root->right, next->val);
    root->val = next->val;
    return root;
  }
  if (key < root->val) {
    root->left = deleteNode(root->left, key);
  } else {
    root->right = deleteNode(root->right, key);
  }
  return root;
}
```



### java模板

```java
// Java
// 插入，保证val不存在
// 返回插入以后的新树根
TreeNode insertIntoBST(TreeNode root, int val) {
  
  if (root == null) {
    return new TreeNode(val);
  }
  if (val < root.val) {
    root.left = insertIntoBST(root.left, val);
  } else {
    root.right = insertIntoBST(root.right, val);
  }
  return root;
}
// 求val的后继
TreeNode findSucc(TreeNode root, int val) {
  TreeNode curr = root;
  TreeNode ans = null;
  while (curr != null) {
    if (curr.val > val) {
      // case2：当后继存在于经过的点中（找到一个>val的最小点）
      // 含义：
      ans=min(ans, curr.val);
      if (ans == null || ans.val > curr.val)
        ans = curr;
    } 
    if (curr.val == val) {
      if (curr.right != null) {
        // case1：检索到val且右子树存在，右子树一路向左
        curr = curr.right;
        while (curr.left != null) curr = curr.left;
        return curr;
      }
      break;
    } 
    if (val < curr.val) curr = curr.left;
    else curr = curr.right;
  }
  return ans;
}
// 在以root为根的子树中删除key，返回新的根
TreeNode deleteNode(TreeNode root, int key) {
  if (root == null) return null;
  if (root.val == key) {
    if (root.left == null) return root.right;
    // 没有左子树，让right代替root的位置 
    if (root.right == null) return root.left; // 没有右子树
    TreeNode next = root.right;
    while (next.left != null) next = next.left; // 找后继：右子树一路向左
    root.right = deleteNode(root.right, next.val);
    root.val = next.val;
    return root;
  }
  if (key < root.val) {
    root.left = deleteNode(root.left, key);
  } else {
    root.right = deleteNode(root.right, key);
  } 
  return root;
}
```



### python模板

```python
# Python
# 插入，保证val不存在
# 返回插入以后的新树
class Solution:
  def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
    if root is None:
      return TreeNode(val)
    if val < root.val:
      root.left = self.insertIntoBST(root.left, val)
    else:
      root.right = self.insertIntoBST(root.right, val)
    return root
  
  # 求val的后继
  def findSucc(self, root, val):
    ans = None
    while root:
      if val == root.val:
        if root.right:
          p = root.right
          while p.left:
            p = p.left
          return p
        break
      if root.val > val:
        if ans is None or ans.val > root.val:
          ans = root
        if val < root.val:
          root = root.left
        else:
          root = root.right
    return ans
  
  # 在以root为根的子树中删除key，返回新的根 
  def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
    if root is None:
      return None
    if root.val == key:
      # 没有左子树，让right代替root的位置
      if root.left is None:
        return root.right
      # 没有右子树
      if root.right is None:
        return root.left
      # 找后继：右子树一路向左
      next = root.right
      while next.left: 
        next = next.left
      root.right = self.deleteNode(root.right, next.val)
      root.val = next.val
      return root
    if key < root.val:
      root.left = self.deleteNode(root.left, key)
    else:
      root.right = self.deleteNode(root.right, key)
    return root
```



## 二分查找模板

### Golang模板

```go
// 普通二分查找
func binarySearch(nums []int, target int) int {
  var i, j = 0, len(nums) - 1
  for i <= j {
    var mid = i + (j - i) / 2
    if nums[mid] == target {
      return mid 
    }
    if target > nums[mid] {
      i = mid + 1
    } else {
      j = mid - 1
    }
  }
  return -1
}

// 更通用的二分模板
func searchRange(nums []int, target int) []int {
  left, right := 0, len(nums)-1
  for {
    if left > right {
      return []int{-1, -1}
    }
    mid := (left+right)/2 
    if nums[mid] == target {
      leftBound := findLeftBound(nums, left, mid, target)
      rightBound := findRightBound(nums, mid, right, target)
      return []int{leftBound, rightBound}
    } else if nums[mid] > target {
      right = mid - 1
    } else {
      left = mid + 1
    }
  }
}

func findLeftBound(nums []int, left, right, target int) int {
  for {
    if left > right {
      return left
    }
    mid := (left+right)/2
    if nums[mid] == target {
      right = mid - 1
    } else {
      left = mid + 1
    }
  }
}

func findRightBound(nums []int, left, right, target int) int {
  for {
    if left > right {
      return right
    }
    mid := (left+right)/2
    if nums[mid] == target {
      left = mid + 1
    } else {
      right = mid - 1
    }
  }
}

func realSqrt(x int, eps float64) float64 {
  left := 0.0
  right := x
  for (right - left > eps) {
    var mid = (left + right) / 2
    if mid * mid <= x {
      left = mid 
    } else {
      right = mid 
    }
  }
  return right
}
```



### C++模板

```c++
// C/C++
// 普通二分查找
int binarySearch(const vector<int>& nums, int target) {
  int left = 0, right = (int)nums.size()-1;
  while (left <= right) {
    int mid = left + (right - left)/ 2;
    if (nums[mid] == target) return mid;
    else if (nums[mid] < target) left = mid + 1;
    else right = mid - 1;
  }
  return -1;
}

// 更通用的二分模板
// LeetCode34 在排序数组中查找元素的第一个和最后一个位置
class Solution {
  public:    
  	vector<int> searchRange(vector<int>& nums, int target) {
      vector<int> ans;
      // 开始位置（lower_bound）：查询第一个>=target的数
      // 范围 [0 .. n-1 ] + [n表示不存在]
      int left = 0, right = nums.size();
      while (left < right) {
        int mid = (left + right) / 2;
        if (nums[mid] >= target) right = mid;
        else left = mid + 1;
      }
      ans.push_back(right);
      //第一个>=target的数的下标（不存在为n）
      // 结束位置：查询最后一个<=target的数
      // 范围 [-1表示不存在] + [0 .. n-1 ] 
      left = -1, right = nums.size() - 1; 
      while (left < right) {
        int mid = (left + right + 1) / 2;
        if (nums[mid] <= target) left = mid;
        else right = mid - 1;
      } 
      ans.push_back(right);
      
      //最后一个<=target的数（不存在为-1）
      // target出现在[ans[0], ans[1]]
      if (ans[0] > ans[1]) ans = {-1, -1};
      return ans;
    }
};

// 实数二分模板
// ans = realSqrt(x)
// 如果要求4位小数，就多算2~4位，到1e-6或1e-8，保证精确
double realSqrt(double x, double eps = 1e-6) {
  double left = 0, right = max(x, 1.0);
  while (right - left > eps) {
    double mid = (left + right) / 2;
    if (mid * mid <= x) {
      left = mid;
    } else {
      right = mid;
    }
  }
  return right;
}
```



### java模板

```java
// 普通二分查找
public int binarySearch(int[] array, int target) {
  int left = 0, right = array.length - 1, mid;
  while (left <= right) {
    mid = (right - left) / 2 + left;
    if (array[mid] == target) {
      return mid;
    } else if (array[mid] > target) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  return -1;
}

// 更通用的二分模板
// LeetCode34 在排序数组中查找元素的第一个和最后一个位置
class Solution { 
  public int[] searchRange(int[] nums, int target) {
    int[] ans = new int[2];
    // 开始位置（lower_bound）：查询第一个>=target的数
    // 范围 [0 .. n-1 ] + [n表示不存在]
    int left = 0, right = nums.length;
    while (left < right) {
      int mid = (left + right) / 2;
      if (nums[mid] >= target) right = mid;
      else left = mid + 1;
    }
    ans[0] = right;
    //第一个>=target的数的下标（不存在为n）
    // 结束位置：查询最后一个<=target的数
    // 范围 [-1表示不存在] + [0 .. n-1 ]
    left = -1; right = nums.length - 1;
    while (left < right) {
      int mid = (left + right + 1) / 2;
      if (nums[mid] <= target) left = mid;
      else right = mid - 1;
    }
    ans[1] = right;
    //最后一个<=target的数（不存在为-1）
    // target出现在[ans[0], ans[1]]
    if (ans[0] > ans[1]) ans = new int[]{-1, -1};
    return ans;
  }
}

// 实数二分模板
// ans = realSqrt(x, 1e-6)
// 如果要求4位小数，就多算2~4位，到1e-6或1e-8，保证精确
double realSqrt(double x, double eps) {
  double left = 0, right = Math.max(x, 1);
  while (right - left > eps) {
    double mid = (left + right) / 2;
    if (mid * mid <= x) {
      left = mid;
    } else {
      right = mid;
    }
  }
}
```



### python模板

```python
# 普通二分查找
def searchRange(self, nums: List[int], target: int) -> List[int]:  
  left, right = 0, len(array) - 1 
	while left <= right:
  	mid = (left + right) / 2
    if array[mid] == target:
      # find the target!!
      break or return result
    elif array[mid] < target:
      left = mid + 1 	 
    else:
      right = mid - 1
      
      
# 更通用的二分模板
# LeetCode34 在排序数组中查找元素的第一个和最后一个位置
class Solution:
	def searchRange(self, nums: List[int], target: int) -> List[int]:
    ans = [-1, -1]
    # 开始位置（lower_bound）：查询第一个>=target的数
    # 范围 [0 .. n-1 ] + [n表示不存在]
    left = 0
    right = len(nums)
    while left < right:
      mid = (left + right) >> 1
      if nums[mid] >= target:
        right = mid
      else:
        left = mid + 1
    ans[0] = right #第一个>=target的数的下标（不存在为n）
    # 结束位置：查询最后一个<=target的数
    # 范围 [-1表示不存在] + [0 .. n-1 ]
    left = -1
    right = len(nums) - 1
    while left < right:
      mid = (left + right + 1) >> 1
      if nums[mid] <= target:
        left = mid
      else:
        right = mid - 1
    ans[1] = right #最后一个<=target的数（不存在为-1）
    # target出现在[ans[0], ans[1]]
    if ans[0] > ans[1]:
      ans = [-1, -1]
      return ans
    
# 实数二分模板
# ans = realSqrt(x, 1e-6)
# 如果要求4位小数，就多算2~4位，到1e-6或1e-8，保证精确
def realSqrt(x, eps=1e-6):
  left, right = 0, max(x, 1)
  while right - left > eps:
    mid = (left + right) / 2
    if mid * mid <= x:
      left = mid
    else:
      right = mid
  return right
```



## 二叉堆代码模板

### Golang模板

```go
func mergeKLists(lists []*ListNode) *ListNode {
  k := len(lists)	if k == 0 {
    return nil
  } else if k == 1 {
    return lists[0]
  } else {
    heap := NewMinHeap()
    var result *ListNode
    node := result
    for _, list := range lists {
      heap.Push(list)
    }
    for heap.length != 0 {
      tmp, err := heap.Pop()
      if err != nil {
        break
      }
      if node == nil {
        result = tmp
        node = tmp
      } else {
        node.Next = tmp
        node = node.Next
      }
      heap.Push(tmp.Next)
    }
    return result
  }
}

// 初始化小顶堆
type MinHeap struct {
  data   []*ListNode
  length int
}

func NewMinHeap() *MinHeap {
  return &MinHeap{data: []*ListNode{}, length: 0}
}

func (this *MinHeap) Pop() (*ListNode, error) {
  if this.length == 0 {
    return nil, errors.New("heap has no elements")
  } else if this.length == 1 {
    result := this.data[0]
    this.length--
    this.data = this.data[1:]
    return result, nil
  } else {
    result := this.data[0]
    this.data[0], this.data[this.length-1] = this.data[this.length-1], this.data[0]
    this.data = this.data[:this.length-1]
    this.length--
    this.data = minHeapifyFromUp(this.data, 0, this.length-1)
    return result, nil	
  }
}

func (this *MinHeap) Push(num *ListNode) {
  if num == nil {
    return
  }
  this.data = append(this.data, num)
  this.length++
  this.data = minHeapifyFromDown(this.data, this.length-1, 0)
}

// 小顶堆自上向下堆化

func heapifyUp(nums []*ListNode, start, end int) []*ListNode {
  left := 2*start + 1
  for left <= end {
    tmp := left
    right := left + 1
    if right <= end && nums[right].Val < nums[tmp].Val {
      tmp = right
    }
    if nums[tmp].Val < nums[start].Val {
      nums[tmp], nums[start] = nums[start], nums[tmp]
      start = tmp
      left = start*2 + 1
    } else {
      break
    }
  }
  return nums
}

// 小顶堆自下向上堆化

func heapifyDown(nums []*ListNode, start, end int) []*ListNode {
  father := (start - 1) / 2
  for father >= end && father != start {
    if nums[start].Val < nums[father].Val {
      nums[father], nums[start] = nums[start], nums[father]
      start = father
      father = (start - 1) / 2
    } else {
      break
    }
  }
  return nums
}
```



### C++模板

```c++

```



### java模板

```java

```



### python模板

```python
# Python
# LeetCode23 合并K个升序链表
from collections import namedtuple

# 堆结点（key用于比较的关键码，listNode可以是任意的附带信息）
Node = namedtuple('Node', ['key', 'listNode'])

# 小根二叉堆
class BinaryHeap:
  def __init__(self):
    # 数组存储完全二叉树
    # 从索引0开始存
    self.heap = [];
  def empty(self):
    return len(self.heap) == 0
  def push(self, node):
    # 插入到尾部
    self.heap.append(node)
    # 向上调整
    self.heapifyUp(len(self.heap) - 1)
  def pop(self):
    ans = self.heap[0]
    # 末尾交换到头部，然后删除末尾
    self.heap[0] = self.heap[-1]
    self.heap.pop()
    # 向下调整
    self.heapifyDown(0) 
    return ans
  def heapifyUp(self, p):
    while p > 0:
      fa = (p - 1) // 2
      if self.heap[p].key < self.heap[fa].key: # 小根堆
        self.heap[p], self.heap[fa] = self.heap[fa], self.heap[p]
        p = fa
      else:
        break
  
  def heapifyDown(self, p):
    child = p * 2 + 1
    while child < len(self.heap):  # child未出界，说明p有合法的child，还不是叶子
      otherChild = p * 2 + 2
      # 先比较两个孩子，谁小就继续跟p比较
      # child存较小的孩子
      if otherChild < len(self.heap) and self.heap[child].key > self.heap[otherChild].key:
        child = otherChild
        # 让child跟p比较
        if self.heap[p].key > self.heap[child].key:  # 小根堆
          self.heap[p], self.heap[child] = self.heap[child], self.heap[p]
          p = child
          child = p * 2 + 1
          else:
            break
            
class Solution:
  def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    # O(元素个数*logK)
    # O(total*logK)
    q = BinaryHeap()
    for listNode in lists:
      if listNode != None:
        q.push(Node(listNode.val, listNode))
        head = ListNode()
        tail = head
    while not q.empty():
      # 取出k个指针指向的最小元素
      node = q.pop()
      # 在答案链表的末尾插入
      tail.next = node.listNode 
      tail = tail.next
      # 当最小被取出后，指针向后移动一位，可能需要插入新的元素
      p = node.listNode.next 
      if p: 
        q.push(Node(p.val, p))
    return head.next
```









