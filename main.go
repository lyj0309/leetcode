package main

import (
	"fmt"
	"math"
)

func main() {
	fmt.Println(toD(20))
}

func toD(i int) int {
	cd := 30
	nc := 0
	var f int
	f = 5
	ha := (i - 1) / 8
	if i-ha*8 > 4 {
		f = 4
	}
	dest := int(math.Abs(float64(20 * ((i - ha*8) - f))))
	nc = dest + (-(i-24)/8)*20
	nd := nc + cd
	return nd
}

func DtoDest(area, i int) int {

}
