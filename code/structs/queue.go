package structs

import "fmt"

// Queue 队列 实现一种先进先出（last-in，first-out，LIFO）策略
type Queue struct {
	list []Any
}

//初始化队列
func NewQueue() *Queue {
	return &Queue{
		list: make([]Any, 0),
	}
}

func (s *Queue) Len() int {
	return len(s.list)
}

//判断栈是否空
func (s *Queue) IsEmpty() bool {
	if len(s.list) == 0 {
		return true
	} else {
		return false
	}
}

//入栈
func (s *Queue) Enqueue(x interface{}) {
	s.list = append(s.list, x)
}

//连续传入
func (s *Queue) DequeueList(x []Any) {
	s.list = append(s.list, x...)
}

//出栈
func (s *Queue) Dequeue() Any {
	if len(s.list) <= 0 {
		fmt.Println("Queue is Empty")
		return nil
	} else {
		// 这种方式可能会造成不必要的损耗
		ret := s.list[0]
		s.list = s.list[1:]
		return ret
	}
}

//返回栈顶元素，空栈返nil
func (s *Queue) Top() Any {
	if s.IsEmpty() == true {
		fmt.Println("Queue is Empty")
		return nil
	} else {
		return s.list[len(s.list)-1]
	}
}
