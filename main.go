package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type ListNode struct {
	Val  int
	Next *ListNode
}

var tree = TreeNode{Val: 1, Right: &TreeNode{Val: 3, Left: &TreeNode{Val: 6}, Right: &TreeNode{Val: 7}}}
var tree1 = TreeNode{Val: 1, Left: &TreeNode{Val: 2, Left: &TreeNode{Left: &TreeNode{Val: 4}, Right: &TreeNode{Val: 5}}}, Right: &TreeNode{Val: 3, Left: &TreeNode{Val: 6}, Right: &TreeNode{Val: 7}}}

/*
  			1
		2		3
	 4	   5  6
*/

func main() {
	/*	fmt.Println(mergeTwoLists(&ListNode{
			Val:  1,
			Next: &ListNode{
				Val:  2,
				Next: &ListNode{
					Val:  4,
					Next: nil,
				},
			},
		},&ListNode{
			Val:  1,
			Next: &ListNode{
				Val:  3,
				Next: &ListNode{
					Val:  4,
					Next: nil,
				},
			},
		}))*/

	/*	t1 := time.Now()
		   fmt.Println(3 / 2)
		   fmt.Println(findDisappearedNumbers([]int{
			   4, 3, 2, 7, 8, 2, 3, 1,
		   }))
		   //fmt.Println(checkN(3))
		   elapsed := time.Since(t1)
		   fmt.Println("运行时间: ", elapsed)*/
	fmt.Println(plusOne([]int{9, 8, 9}))
}
func plusOne(digits []int) []int {
	length := len(digits)
	for i := length - 1; i >= 0; i-- {
		digits[i]++
		digits[i] = digits[i] % 10
		if digits[i] != 0 {
			return digits
		}
	}
	digits = append(digits, 0)
	copy(digits[1:], digits[0:]) // 先后移动一位
	digits[0] = 1
	return digits
}
func plusOne(digits []int) []int {
	if digits[len(digits)-1] != 9 {
		digits[len(digits)-1] += 1
		return digits
	}
	digits[len(digits)-1] = 0
	p := true
	for i := len(digits) - 2; i >= 0; i-- {
		fmt.Println(digits)
		if p == true && digits[i] == 9 {
			digits[i] = 0
			continue
		}
		if p == true {
			p = false
			digits[i] += 1
		}
	}
	if p == true {
		digits = append([]int{1}, digits...)
	}
	return digits
}

func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}

	cur := head
	for cur.Next != nil {
		if cur.Next.Val == cur.Val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return head
}

func deleteDuplicates1(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}

	dummy := &ListNode{0, head} //哑巴节点

	cur := dummy
	for cur.Next != nil && cur.Next.Next != nil {
		if cur.Next.Val == cur.Next.Next.Val {
			x := cur.Next.Val
			for cur.Next != nil && cur.Next.Val == x {
				cur.Next = cur.Next.Next
			}
		} else {
			cur = cur.Next
		}
	}

	return dummy.Next
}

func countNegatives(grid [][]int) int {
	l := len(grid[0])
	r := 0
	for _, ints := range grid {
		for i := l; i > 0; i-- {
			if ints[i-1] < 0 {
				r++
			} else {
				break
			}
		}
	}
	return r
}
