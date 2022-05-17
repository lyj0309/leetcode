package model

type User interface {
	Info()
}

type Student struct {
	Name string
}

func (s Student) Info() {

}
