# 模版

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

前缀和、数组计数代码模板

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



## 二维前缀和代码模板

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







## 模版格式备份

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

