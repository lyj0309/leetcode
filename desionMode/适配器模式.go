package main

import "fmt"

//客户端代码
type client struct {
}

func (c *client) insertLightningConnectorIntoComputer(com computer) {
	fmt.Println("Client inserts Lightning connector into computer.")
	com.insertIntoLightningPort()
}

//客户端接口
type computer interface {
	insertIntoLightningPort()
}

//服务
type mac struct {
}

func (m *mac) insertIntoLightningPort() {
	fmt.Println("Lightning connector is plugged into mac machine.")
}

//未知服务
type windows struct{}

func (w *windows) insertIntoUSBPort() {
	fmt.Println("USB connector is plugged into windows machine.")
}

//适配器
type windowsAdapter struct {
	windowMachine *windows
}

func (w *windowsAdapter) insertIntoLightningPort() {
	fmt.Println("Adapter converts Lightning signal to USB.")
	w.windowMachine.insertIntoUSBPort()
}
