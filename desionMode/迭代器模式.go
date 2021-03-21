package main

//集合
type collection interface {
	createIterator() iterator
}

//具体集合
type userCollection struct {
	users []*user
}

func (u *userCollection) createIterator() iterator {
	return &userIterator{
		users: u.users,
	}
}

//迭代器
type iterator interface {
	hasNext() bool
	getNext() *user
}

//具体迭代器
type userIterator struct {
	index int
	users []*user
}

func (u *userIterator) hasNext() bool {
	if u.index < len(u.users) {
		return true
	}
	return false

}
func (u *userIterator) getNext() *user {
	if u.hasNext() {
		user := u.users[u.index]
		u.index++
		return user
	}
	return nil
}

//客户端代码
type user struct {
	name string
	age  int
}
