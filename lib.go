package main

import (
	"strconv"
)

//获取十进制的每一位
func getBit(a int) (array []int) {
	count := 0
	for n := a; n != 0; n /= 10 {
		count++
	}

	k := 1
	for i := 1; i <= count; i++ {
		p := a / (1 * k) % 10
		k *= 10
		array = append([]int{p}, array...)
	}
	return array
}

func DecConvertToX(n, num int) string {
	if n < 0 {
		return strconv.Itoa(n)
	}
	result := ""
	h := map[int]string{
		0:  "0",
		1:  "1",
		2:  "2",
		3:  "3",
		4:  "4",
		5:  "5",
		6:  "6",
		7:  "7",
		8:  "8",
		9:  "9",
		10: "A",
		11: "B",
		12: "C",
		13: "D",
		14: "E",
		15: "F",
	}
	for ; n > 0; n /= num {
		lsb := h[n%num]
		result = lsb + result
	}
	return result
}
