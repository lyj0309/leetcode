package main

import (
	"fmt"
	"strconv"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var tree = TreeNode{Val: 1, Right: &TreeNode{Val: 3, Left: &TreeNode{Val: 6}, Right: &TreeNode{Val: 7}}}
var tree1 = TreeNode{Val: 1, Left: &TreeNode{Val: 2, Left: &TreeNode{Left: &TreeNode{Val: 4}, Right: &TreeNode{Val: 5}}}, Right: &TreeNode{Val: 3, Left: &TreeNode{Val: 6}, Right: &TreeNode{Val: 7}}}

/*
  			1
		2		3
	 4	   5  6
*/

func main() {
	fmt.Println(isValid("({[)"))
	/*	t1 := time.Now()
		   fmt.Println(3 / 2)
		   fmt.Println(findDisappearedNumbers([]int{
			   4, 3, 2, 7, 8, 2, 3, 1,
		   }))
		   //fmt.Println(checkN(3))
		   elapsed := time.Since(t1)
		   fmt.Println("运行时间: ", elapsed)*/
}

func isValid(s string) bool {
	var queue []int32
	for _, b := range s {
		if len(queue) != 0 {
			switch queue[len(queue)-1] {
			case 40:
				if b == 41 {
					queue = queue[:len(queue)-1]
				} else {
					queue = append(queue, b)
				}
			case 91:
				if b == 93 {
					queue = queue[:len(queue)-1]
				} else {
					queue = append(queue, b)
				}

			case 123:
				if b == 125 {
					queue = queue[:len(queue)-1]
				} else {
					queue = append(queue, b)
				}
			}
		} else {
			queue = append(queue, b)
		}
	}
	return len(queue) == 0
}

func setZeroes(matrix [][]int) {
	var zeros []point
	for y, line := range matrix {
		for x, k := range line {
			if k == 0 {
				fmt.Println(x, y)
				zeros = append(zeros, point{
					x: x,
					y: y,
				})
			}
		}
	}
	fmt.Println(zeros)
	for _, zero := range zeros {
		for i := 0; i < len(matrix[zero.y]); i++ {
			matrix[zero.y][i] = 0
		}
		for i := 0; i < len(matrix); i++ {
			matrix[i][zero.x] = 0
		}
	}
}

type point struct {
	x int
	y int
}

func evalRPN(tokens []string) int {
	var queue []int
	for _, token := range tokens {
		switch token {
		case `+`:
			t := queue[len(queue)-2] + queue[len(queue)-1]
			queue = queue[:len(queue)-2]
			queue = append(queue, t)
		case `-`:
			t := queue[len(queue)-2] - queue[len(queue)-1]
			queue = queue[:len(queue)-2]
			queue = append(queue, t)
		case `*`:
			t := queue[len(queue)-2] * queue[len(queue)-1]
			queue = queue[:len(queue)-2]
			queue = append(queue, t)

		case `/`:
			t := queue[len(queue)-2] / queue[len(queue)-1]
			fmt.Println(t)
			queue = queue[:len(queue)-2]
			queue = append(queue, t)
		}
		k, _ := strconv.Atoi(token)
		queue = append(queue, k)
		fmt.Println(queue)
	}
	return queue[0]
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseBetween(head *ListNode, left int, right int) *ListNode {
	if head.Next == nil || left == right {
		return head
	}
	i := 1
	var leftleftnode *ListNode
	var lastnode *ListNode
	var leftnode *ListNode
	var rightnode *ListNode
	trav := func(node *ListNode) {}
	trav = func(node *ListNode) {
		if node.Next == nil {
			return
		}
		if i == left-1 {
			leftleftnode = node
		}
		if i == left {
			lastnode = node
			leftnode = node
		}
		if i == right {
			rightnode = node
			leftnode.Next = node.Next
		}
		fmt.Println(node.Val, i)
		if i > left && i <= right {
			n := node.Next
			node.Next = lastnode
			lastnode = node
			i++
			trav(n)
			return
		}
		i++
		trav(node.Next)
	}
	trav(head)
	if leftleftnode != nil {
		leftleftnode.Next = rightnode
	}
	return head
}

func generateMatrix(n int) [][]int {
	var res [][]int
	for i := 0; i < n; i++ {
		res = append(res, make([]int, n))
	}
	x, y, d := 0, 0, 0
	for i := 1; i <= n*n; i++ {
		res[y][x] = i
		switch d {
		case 0:
			if len(res[y])-1 == x || res[y][x+1] != 0 {
				y++
				d++
			} else {
				x++
			}
		case 1:
			if len(res)-1 == y || res[y+1][x] != 0 {
				x--
				d++
			} else {
				y++
			}
		case 2:
			if x == 0 || res[y][x-1] != 0 {
				y--
				d++
			} else {
				x--
			}
		case 3:
			if y == 0 || res[y-1][x] != 0 {
				x++
				d = 0
			} else {
				y--
			}
		}
	}
	return res
}

type MyHashMap struct {
	data map[int]int
}

/** Initialize your data structure here. */
func Constructor() MyHashMap {
	return MyHashMap{
		data: make(map[int]int),
	}
}

/** value will always be non-negative. */
func (this *MyHashMap) Put(key int, value int) {
	this.data[key] = value
}

/** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
func (this *MyHashMap) Get(key int) int {
	v, ok := this.data[key]
	if !ok {
		return -1
	}
	return v
}

/** Removes the mapping of the specified value key if this map contains a mapping for the key */
func (this *MyHashMap) Remove(key int) {
	delete(this.data, key)
}

/**
 * Your MyHashMap object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Put(key,value);
 * param_2 := obj.Get(key);
 * obj.Remove(key);
 */

func findDisappearedNumbers(nums []int) (ans []int) {
	n := len(nums)
	for _, v := range nums {
		v = (v - 1) % n
		nums[v] += n
	}
	for i, v := range nums {
		if v <= n {
			ans = append(ans, i+1)
		}
	}
	return
}

func rotate(matrix [][]int) {
	N := len(matrix)
	for i := 1; i < N; i++ { //对称一波
		for k := 0; k < i; k++ {
			b := matrix[i][k]
			matrix[i][k] = matrix[k][i]
			matrix[k][i] = b
		}
	}
	for i := 0; i < N; i++ { //镜像翻转一波
		for k := 0; k < N/2; k++ {
			b := matrix[i][k]
			matrix[i][k] = matrix[i][N-1-k]
			matrix[i][N-1-k] = b
		}
	}
}
func firstUniqChar(s string) int {
	a := make(map[string]int)
	for i := 0; i < len(s); i++ {
		a[s[i:i+1]] += 1
	}
	fmt.Println(a)
	return 1
}
