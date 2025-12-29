import socket
from socket import socket,AF_INET,SOCK_DGRAM
import time


class UdpServer:
    def __init__(self, host='127.0.0.1', port=10000):
        self.ADDRESS = (host, port)
        self.BUFSIZE = 1024
        # 绑定服务端口和地址
        self.udpServerSocket = socket(AF_INET, SOCK_DGRAM)
        self.udpServerSocket.bind(self.ADDRESS)
        self.udpServerSocket.setblocking(0)
        #self.client_addr = None

    def run_once_rec(self):
        try:
            data, addr = self.udpServerSocket.recvfrom(self.BUFSIZE)
            #self.client_addr = addr
            data = data.decode("utf-8")
            #print(data.decode("utf-8"), type(data.decode("utf-8")))
            # print(list(data), list(data)[0])
            # print(float(data.split()[1]))
            #if data is not None:
                #print('data: ', data.decode("utf-8"))
        except:
            data = None
            addr = None
        return data, addr

    def run_once_send(self, send_data, addr):
        self.udpServerSocket.sendto(send_data.encode('utf-8'), addr)


if __name__ == '__main__':

    udp = UdpServer()
    while True:
        data, addr = udp.run_once_rec()
        print(type(data))
        #send_data_R = "P {} {} {}".format(100.11, 200.22, 300.33)
        time.sleep(0.2)
        send_data_R = "R {}".format(0.02)
        if addr is not None:
            udp.run_once_send(send_data_R, addr)
        #udp.run_once_send(send_data_P)
